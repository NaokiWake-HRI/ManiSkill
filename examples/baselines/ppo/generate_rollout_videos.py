"""Generate rollout videos from each outer-loop iteration's checkpoint.

Loads iter_NN_final.pt for each iteration, runs deterministic rollouts
in the same environment, and saves videos for visual comparison.

Usage:
    python generate_rollout_videos.py <run_dir> [--num_eval_envs 4] [--num_eval_steps 200] [--seed 0]

Example:
    python generate_rollout_videos.py runs/ppo-outer-loop-PickCube-v1-9351-PickCube-v1-20260211_124438
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from ppo_outer_loop import Agent
from reward_wrapper import RewardWrapper


def infer_env_id(run_name: str) -> str:
    """Infer env_id from run directory name."""
    # Pattern: ppo-outer-loop-{env_id}-{seed}-{env_id}-{timestamp}
    # or ppo-eureka-{env_id}-{seed}-{env_id}-{timestamp}
    for env_id in ["PickCube-v1", "PushCube-v1", "OpenCabinetDoor-v1", "OpenCabinetDrawer-v1",
                   "PickCubePandaAllegro-v1"]:
        if env_id in run_name:
            return env_id
    raise ValueError(f"Cannot infer env_id from run name: {run_name}")


def main():
    parser = argparse.ArgumentParser(description="Generate rollout videos from outer-loop checkpoints")
    parser.add_argument("run_dir", type=str, help="Path to the run directory")
    parser.add_argument("--env_id", type=str, default=None,
                        help="Environment ID (auto-detected from run name if not given)")
    parser.add_argument("--num_eval_envs", type=int, default=4,
                        help="Number of parallel eval envs (= number of videos per iter)")
    parser.add_argument("--num_eval_steps", type=int, default=200,
                        help="Steps per rollout episode (longer = more time to succeed)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Env reset seed for reproducibility across iterations")
    parser.add_argument("--control_mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <run_dir>/rollout_videos)")
    parser.add_argument("--iterations", type=str, default=None,
                        help="Comma-separated iteration numbers to evaluate, e.g. '1,3,5'. Default: all")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic actions (default: True)")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: run_dir not found: {run_dir}")
        sys.exit(1)

    # Infer env_id
    env_id = args.env_id or infer_env_id(run_dir.name)
    print(f"Environment: {env_id}")

    # Load outer loop history
    history_path = run_dir / "outer_loop_history.json"
    if not history_path.exists():
        print(f"Error: outer_loop_history.json not found in {run_dir}")
        sys.exit(1)
    with open(history_path) as f:
        history = json.load(f)
    print(f"Found {len(history)} iterations in history")

    # Determine which iterations to evaluate
    if args.iterations:
        iter_nums = [int(x) for x in args.iterations.split(",")]
    else:
        iter_nums = list(range(1, len(history) + 1))

    # Verify checkpoints exist
    for n in iter_nums:
        ckpt = run_dir / f"iter_{n:02d}_final.pt"
        if not ckpt.exists():
            print(f"Warning: checkpoint not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_base = Path(args.output_dir) if args.output_dir else run_dir / "rollout_videos"
    output_base.mkdir(parents=True, exist_ok=True)

    for iter_num in iter_nums:
        ckpt_path = run_dir / f"iter_{iter_num:02d}_final.pt"
        if not ckpt_path.exists():
            print(f"Skipping iter {iter_num}: checkpoint not found")
            continue

        entry = history[iter_num - 1]
        weights = entry["weights"]
        print(f"\n{'='*60}")
        print(f"Iteration {iter_num}")
        print(f"  Weights: {weights}")
        print(f"  Training success_once: {entry.get('success_once', 'N/A')}")
        print(f"  Training success_at_end: {entry.get('success_at_end', 'N/A')}")

        # Create environment
        env_kwargs = dict(
            obs_mode="state",
            render_mode="rgb_array",
            sim_backend="physx_cuda",
            reward_mode="none",
        )
        if args.control_mode:
            env_kwargs["control_mode"] = args.control_mode

        eval_envs = gym.make(
            env_id,
            num_envs=args.num_eval_envs,
            reconfiguration_freq=1,
            **env_kwargs,
        )

        if isinstance(eval_envs.action_space, gym.spaces.Dict):
            eval_envs = FlattenActionSpaceWrapper(eval_envs)

        eval_envs = RewardWrapper(eval_envs, env_id=env_id, weights=weights)

        video_dir = str(output_base / f"iter_{iter_num:02d}")
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=video_dir,
            save_trajectory=False,
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )

        eval_envs = ManiSkillVectorEnv(
            eval_envs,
            args.num_eval_envs,
            ignore_terminations=True,  # Don't reset on termination → full-length videos
            record_metrics=True,
        )

        # Load agent
        agent = Agent(eval_envs).to(device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        agent.load_state_dict(state_dict)
        agent.eval()

        # Run rollout
        obs, _ = eval_envs.reset(seed=args.seed)
        eval_metrics = defaultdict(list)
        num_episodes = 0

        for step in range(args.num_eval_steps):
            with torch.no_grad():
                action = agent.get_action(obs, deterministic=args.deterministic)
                obs, rew, terminations, truncations, infos = eval_envs.step(action)

                if "final_info" in infos:
                    mask = infos["_final_info"]
                    num_episodes += mask.sum().item()
                    for k, v in infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v)

        # Print metrics
        print(f"  Rollout results ({args.num_eval_steps} steps, {num_episodes} episodes):")
        for k, v in eval_metrics.items():
            mean = torch.stack(v).float().mean().item()
            print(f"    {k}: {mean:.4f}")

        print(f"  Videos saved to: {video_dir}")
        eval_envs.close()
        del agent

    print(f"\nAll videos saved to: {output_base}")
    print("Done!")


if __name__ == "__main__":
    main()
