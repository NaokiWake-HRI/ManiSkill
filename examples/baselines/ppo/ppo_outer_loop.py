"""
PPO Outer Loop: VLM/LLM-guided Reward Weight Optimization.

Unlike ppo_iterative.py which changes weights mid-training (inner loop),
this script runs full PPO training with fixed weights, then uses VLM/LLM
to suggest new weights, and restarts training from scratch.

Iteration 1 (random weights) serves as the baseline.
Iterations 2+ show VLM/LLM-guided improvement.

Modes:
    1. VLM + LLM mode (default): Uses video analysis + reward optimization
    2. Eureka mode (--eureka_mode): Pure LLM-only, uses learning curve data

Usage:
    # With VLM/LLM (default):
    export OPENAI_API_KEY=sk-... && python ppo_outer_loop.py --env_id PushCube-v1

    # Eureka mode (LLM-only, no VLM):
    export OPENAI_API_KEY=sk-... && python ppo_outer_loop.py --env_id PushCube-v1 --eureka_mode

    # Without VLM/LLM (debug):
    python ppo_outer_loop.py --env_id PushCube-v1 --skip_vlm_llm --num_outer_iters=2 --total_timesteps_per_iter=50000
"""

import inspect
import json
import os
import random as py_random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from ppo_common import (
    layer_init, Agent, Logger,
    crop_tiled_frame, extract_frames_from_video, build_vlm_prompt,
    generate_reward_plot_html, append_html_to_file, generate_random_weights,
)
from reward_wrapper import RewardWrapper, TASK_DEFAULTS, _resolve_task_id
from task_descriptions import get_llm_task_descs


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "PushCube-v1"
    """the id of the environment"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    reward_scale: float = 1.0
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    finite_horizon_gae: bool = False

    # Outer loop arguments
    num_outer_iters: int = 5
    """number of outer loop iterations (each is a full PPO training from scratch)"""
    total_timesteps_per_iter: int = 2_000_000
    """total timesteps per outer iteration"""
    initial_weights_file: Optional[str] = None
    """path to JSON file with initial weights (if None, generate random)"""
    weight_seed: int = 42
    """seed for random weight generation"""

    # VLM/LLM arguments
    vlm_model: str = "gpt-5.2"
    """VLM model for video analysis"""
    llm_model: str = "gpt-5.2"
    """LLM model for reward tuning"""
    vlm_max_frames: int = 8
    """max frames to send to VLM"""
    vlm_num_envs: int = 1
    """number of eval envs to show in VLM frames"""
    vlm_reward_plot: bool = False
    """if toggled, append per-step reward plot to VLM debug HTML"""
    rl_project_path: str = "/home/nwake/codes/RL_project"
    """path to RL_project for VLM/LLM imports"""
    skip_vlm_llm: bool = False
    """skip VLM/LLM calls (for testing)"""
    eureka_mode: bool = False
    """pure Eureka mode: use LLM only without VLM (learning curve based optimization)"""
    enable_function_code: bool = False
    """allow LLM to generate custom reward code (Eureka-style). When False, params-only mode."""

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# ============================================================================
# Single PPO training run
# ============================================================================

def run_ppo_training(
    args: Args,
    weights: Dict[str, float],
    outer_iter: int,
    run_dir: str,
    logger: Optional["Logger"],
    device: torch.device,
    global_step_offset: int,
) -> Dict[str, Any]:
    """Run a single PPO training from scratch with fixed weights.

    Returns:
        Dict with eval metrics, per-step reward data, and video path.
    """
    # Reset RNG so each iteration starts from identical state
    py_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps_per_iter // args.batch_size

    print(f"\n[Outer Iter {outer_iter+1}] Starting PPO training")
    print(f"  Weights: {weights}")
    print(f"  num_iterations={args.num_iterations}, batch_size={args.batch_size}")

    # --- Environment setup ---
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        reward_mode="none",
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs,
    )

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    # RewardWrapper with fixed weights
    reward_wrapper_train = RewardWrapper(envs, env_id=args.env_id, weights=weights)
    reward_wrapper_eval = RewardWrapper(eval_envs, env_id=args.env_id, weights=weights)
    envs = reward_wrapper_train
    eval_envs = reward_wrapper_eval

    # Video recording for eval
    eval_output_dir = f"runs/{run_dir}/videos/iter_{outer_iter+1:02d}"
    if args.capture_video:
        print(f"  Saving eval videos to {eval_output_dir}")
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=False,
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )

    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    # --- Agent (fresh initialization) ---
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- Storage ---
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # --- Training ---
    global_step = global_step_offset
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    action_space_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_space_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    latest_eval_metrics: Dict[str, Any] = {}
    latest_eval_step_rewards: List[torch.Tensor] = []
    latest_eval_reward_breakdowns: List[Dict[str, float]] = []

    # Collect learning curve: eval metrics at each eval point (Eureka-style)
    learning_curve: List[Dict[str, Any]] = []

    for iteration in range(1, args.num_iterations + 1):
        print(f"  Epoch: {iteration}/{args.num_iterations}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()

        # --- Evaluation ---
        is_last_iter = (iteration == args.num_iterations)
        if iteration % args.eval_freq == 1 or is_last_iter:
            print("  Evaluating")
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            step_rewards_list = []
            step_breakdowns_list = []
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(
                        agent.get_action(eval_obs, deterministic=True)
                    )
                    step_rewards_list.append(eval_rew.detach())
                    step_breakdowns_list.append(dict(reward_wrapper_eval._last_breakdown))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"  Evaluated {args.num_eval_steps * args.num_eval_envs} steps, {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"  eval_{k}_mean={mean}")
            latest_eval_metrics = {
                k: torch.stack(v).float().mean().item() for k, v in eval_metrics.items()
            }
            if isinstance(num_episodes, torch.Tensor):
                latest_eval_metrics["num_episodes"] = int(num_episodes.item())
            else:
                latest_eval_metrics["num_episodes"] = int(num_episodes)

            # Store per-step reward data
            latest_eval_step_rewards = step_rewards_list
            latest_eval_reward_breakdowns = step_breakdowns_list

            # Record learning curve point (Eureka-style)
            lc_point = {
                "step": global_step,
                "avg_return": latest_eval_metrics.get("return", 0.0),
                "success_at_end": latest_eval_metrics.get("success_at_end", 0.0),
                "success_once": latest_eval_metrics.get("success_once", 0.0),
                "success_rate": latest_eval_metrics.get("success_at_end", 0.0),
            }
            # Add per-component reward means (Eureka-style)
            if step_breakdowns_list:
                comp_means = {}
                for key in step_breakdowns_list[0]:
                    if key == "norm_scale":
                        continue
                    vals = [bd[key] for bd in step_breakdowns_list]
                    comp_means[key] = sum(vals) / len(vals)
                lc_point["reward_components"] = comp_means
            learning_curve.append(lc_point)

        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_dir}/iter_{outer_iter+1:02d}_ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)

        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # --- Rollout ---
        rollout_time = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    if logger is not None:
                        logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(
                        infos["final_observation"][done_mask]
                    ).view(-1)
        rollout_time = time.time() - rollout_time

        # --- GAE ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                if args.finite_horizon_gae:
                    if t == args.num_steps - 1:
                        lam_coef_sum = 0.0
                        reward_term_sum = 0.0
                        value_term_sum = 0.0
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done
                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values
                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
            returns = advantages + values

        # --- PPO Update ---
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if logger is not None:
            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)
            sps = int(global_step / (time.time() - start_time))
            logger.add_scalar("charts/SPS", sps, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)

    # Save final model for this iteration
    if args.save_model:
        model_path = f"runs/{run_dir}/iter_{outer_iter+1:02d}_final.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"  Model saved to {model_path}")

    # Close envs
    envs.close()
    eval_envs.close()

    return {
        "eval_metrics": latest_eval_metrics,
        "step_rewards": latest_eval_step_rewards,
        "reward_breakdowns": latest_eval_reward_breakdowns,
        "eval_video_dir": eval_output_dir,
        "final_global_step": global_step,
        "learning_curve": learning_curve,
    }


# ============================================================================
# Main: Outer Loop
# ============================================================================

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.exp_name}-{args.env_id}-{timestamp}"
    experiment_type = "eureka" if args.eureka_mode else "outer-loop"
    run_dir = f"{experiment_type}/{args.env_id}/{run_name}"

    # Seeding
    py_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # --- Initial weights ---
    if args.initial_weights_file:
        with open(args.initial_weights_file) as f:
            current_weights = json.load(f)
        print(f"Loaded initial weights from {args.initial_weights_file}: {current_weights}")
    else:
        current_weights = generate_random_weights(args.env_id, seed=args.weight_seed)
        print(f"Generated random weights (seed={args.weight_seed}): {current_weights}")

    initial_weights = dict(current_weights)

    # --- Logging ---
    logger = None
    if args.track:
        import wandb
        config = vars(args)
        config["initial_weights"] = initial_weights
        # Set tags based on mode
        tags = ["ppo", "outer-loop"]
        if args.eureka_mode:
            tags.extend(["eureka", "llm-only"])
        else:
            tags.extend(["vlm-llm"])
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group="PPO-OuterLoop",
            tags=tags,
        )
    writer = SummaryWriter(f"runs/{run_dir}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logger = Logger(log_wandb=args.track, tensorboard=writer)

    # --- VLM/LLM setup (once) ---
    vlm = None
    llm = None
    save_vlm_debug_html = None
    save_llm_debug_html = None
    if not args.skip_vlm_llm:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            # Mock stable_baselines3 to avoid dependency
            import types
            sb3_mocks = {
                "stable_baselines3": {},
                "stable_baselines3.common": {},
                "stable_baselines3.common.callbacks": {"BaseCallback": object, "CallbackList": object},
                "stable_baselines3.common.base_class": {"BaseAlgorithm": object},
                "stable_baselines3.common.vec_env": {"VecNormalize": object},
                "stable_baselines3.common.logger": {"configure": lambda *a, **kw: None},
            }
            for mod_name, attrs in sb3_mocks.items():
                if mod_name not in sys.modules:
                    mock = types.ModuleType(mod_name)
                    for attr_name, attr_val in attrs.items():
                        setattr(mock, attr_name, attr_val)
                    sys.modules[mod_name] = mock

            sys.path.insert(0, args.rl_project_path)
            from experiments.callbacks.episode_collector import VLMEvaluator, LLMRewardTuner
            from experiments.iterative_learner import (
                save_vlm_debug_html as _save_vlm_debug_html,
                save_llm_debug_html as _save_llm_debug_html,
            )
            save_vlm_debug_html = _save_vlm_debug_html
            save_llm_debug_html = _save_llm_debug_html

            # Initialize VLM only if not in Eureka mode
            if not args.eureka_mode:
                vlm = VLMEvaluator.from_openai(
                    api_key=api_key,
                    model=args.vlm_model,
                    prompt=build_vlm_prompt(args.env_id),
                    max_frames=args.vlm_max_frames,
                    cache_results=False,
                )
                print(f"[VLM/LLM] Initialized VLM: {args.vlm_model}")
            else:
                print("[Eureka Mode] Skipping VLM initialization (LLM-only mode)")

            # Always initialize LLM
            llm = LLMRewardTuner.from_openai(
                api_key=api_key,
                model=args.llm_model,
                enable_function_code=args.enable_function_code,
                max_param_change=2.0,
            )
            print(f"[VLM/LLM] Initialized LLM: {args.llm_model}")
        else:
            print("[warn] OPENAI_API_KEY not set, skipping VLM/LLM")
    else:
        print("[info] VLM/LLM disabled (--skip_vlm_llm)")

    print(f"\n{'='*60}")
    print(f"PPO Outer Loop: {args.num_outer_iters} iterations x {args.total_timesteps_per_iter} steps")
    print(f"Initial weights: {current_weights}")
    print(f"{'='*60}\n")

    # --- Outer Loop ---
    outer_loop_history = []
    global_step_offset = 0

    for outer_iter in range(args.num_outer_iters):
        print(f"\n{'='*60}")
        print(f"OUTER ITERATION {outer_iter+1}/{args.num_outer_iters}")
        print(f"Weights: {current_weights}")
        print(f"{'='*60}")

        # Log weights at start of iteration (use global_step_offset for monotonic wandb step)
        if logger is not None:
            for wk, wv in current_weights.items():
                logger.add_scalar(f"outer_iter/weights/{wk}", wv, global_step_offset)

        # Run full PPO training with fixed weights
        result = run_ppo_training(
            args=args,
            weights=current_weights,
            outer_iter=outer_iter,
            run_dir=run_dir,
            logger=logger,
            device=device,
            global_step_offset=global_step_offset,
        )
        global_step_offset = result["final_global_step"]

        eval_metrics = result["eval_metrics"]
        success_at_end = eval_metrics.get("success_at_end", 0.0)
        success_once = eval_metrics.get("success_once", 0.0)
        avg_return = eval_metrics.get("return", 0.0)

        print(f"\n[Outer Iter {outer_iter+1}] Results:")
        print(f"  Success at end: {success_at_end:.4f}")
        print(f"  Success once: {success_once:.4f}")
        print(f"  Avg return: {avg_return:.4f}")

        # Log outer iteration metrics (use global_step_offset so wandb step is monotonic)
        if logger is not None:
            logger.add_scalar("outer_iter/success_at_end", success_at_end, global_step_offset)
            logger.add_scalar("outer_iter/success_once", success_once, global_step_offset)
            logger.add_scalar("outer_iter/avg_return", avg_return, global_step_offset)
            logger.add_scalar("outer_iter/iteration", outer_iter, global_step_offset)

        # --- VLM/LLM analysis ---
        vlm_comment = None
        new_weights_suggestion = None

        # VLM analysis (skip in Eureka mode)
        if vlm is not None:
            print(f"\n[VLM] Analyzing iteration {outer_iter+1}...")

            # Extract frames from latest eval video
            video_dir = Path(result["eval_video_dir"])
            video_files = sorted(video_dir.glob("*.mp4"))

            if video_files:
                latest_video = video_files[-1]
                frames = extract_frames_from_video(
                    latest_video,
                    max_frames=args.vlm_max_frames,
                    num_total_envs=args.num_eval_envs,
                    num_show_envs=args.vlm_num_envs,
                )
                if frames:
                    episode_info = {
                        "return": avg_return,
                        "success_at_end": success_at_end,
                        "success_once": success_once,
                        "length": args.num_eval_steps,
                    }

                    vlm_score, vlm_comment, _ = vlm.evaluate(frames, episode_info)
                    print(f"[VLM] Analysis:\n{vlm_comment}")

                    # Save VLM debug HTML
                    debug_dir = Path(f"runs/{run_dir}/debug_html")
                    vlm_html_path = debug_dir / f"iter_{outer_iter+1:02d}_vlm.html"
                    save_vlm_debug_html(
                        frames=frames,
                        prompt=build_vlm_prompt(args.env_id),
                        episode_info=episode_info,
                        vlm_score=vlm_score,
                        vlm_comment=vlm_comment,
                        save_path=vlm_html_path,
                        max_frames=args.vlm_max_frames,
                    )

                    # Append per-step reward plot
                    if args.vlm_reward_plot and result["step_rewards"]:
                        step_rewards_tensor = torch.stack(result["step_rewards"])
                        plot_html = generate_reward_plot_html(
                            step_rewards_tensor, args.num_eval_envs,
                            breakdowns=result["reward_breakdowns"],
                        )
                        append_html_to_file(vlm_html_path, plot_html)
                else:
                    print("[warn] No frames extracted from video")
            else:
                print("[warn] No eval videos found")

        # LLM reward tuning (works with or without VLM)
        if llm is not None:
            if vlm_comment is not None or args.eureka_mode:
                # Build task description for LLM (appears in prompt via llm_task_description)
                task_id = _resolve_task_id(args.env_id)
                _llm_task_descs = get_llm_task_descs(args.env_id)

                # Compute per-component reward means from final eval (Eureka-style)
                reward_components = {}
                if result.get("reward_breakdowns"):
                    bds = result["reward_breakdowns"]
                    for key in bds[0]:
                        if key == "norm_scale":
                            continue
                        vals = [bd[key] for bd in bds]
                        reward_components[key] = sum(vals) / len(vals)

                # Get reward function source code for LLM context
                _reward_method_map = {
                    "PickCube": "_compute_pick_cube",
                    "PushCube": "_compute_push_cube",
                    "OpenCabinetDoor": "_compute_open_cabinet",
                    "OpenCabinetDrawer": "_compute_open_cabinet",
                    "UnitreeG1PlaceAppleInBowl": "_compute_unitree_place_apple",
                    "AnymalC": "_compute_anymalc_reach",
                    "PegInsertionSide": "_compute_peg_insertion",
                    "PushT": "_compute_push_t",
                }
                try:
                    method_name = _reward_method_map[task_id]
                    reward_fn_source = inspect.getsource(
                        getattr(RewardWrapper, method_name)
                    )
                except Exception as e:
                    print(f"[Warning] Could not get reward_fn_source: {e}")
                    reward_fn_source = "N/A"

                training_summary = {
                    "current_weights": dict(current_weights),
                    "initial_weights": initial_weights,
                    "current_iteration": outer_iter,
                    "total_iterations": args.num_outer_iters,
                    "total_timesteps": global_step_offset,
                    "avg_return": avg_return,
                    "success_rate": success_at_end,
                    "success_at_end": success_at_end,
                    "success_once": success_once,
                    "vlm_comments": [vlm_comment],
                    "num_episodes": int(eval_metrics.get("num_episodes", 0)),
                    "llm_task_description": _llm_task_descs.get(task_id, ""),
                    "reward_fn_source": reward_fn_source,
                    "learning_curve": result["learning_curve"],
                    "reward_components": reward_components,
                }

                # Add past iteration history for LLM context
                if outer_loop_history:
                    # Performance trend: how results changed over iterations
                    training_summary["performance_trend"] = [
                        {
                            "iteration": h["outer_iter"],
                            "weights": h["weights"],
                            "avg_return": h["avg_return"],
                            "success_rate": h["success_at_end"],
                            "success_at_end": h["success_at_end"],
                            "success_once": h["success_once"],
                            "learning_curve": h.get("learning_curve", []),
                        }
                        for h in outer_loop_history
                    ]
                    # Weight change history: what was changed and why
                    training_summary["history_summary"] = [
                        {
                            "iteration": h["outer_iter"],
                            "changes": h.get("rationale", "N/A"),
                            "result": {
                                "avg_return": h["avg_return"],
                                "success_rate": h["success_at_end"],
                                "success_at_end": h["success_at_end"],
                                "success_once": h["success_once"],
                            },
                            "rationale": h.get("rationale", "N/A"),
                            "vlm_comment": h.get("vlm_comment", "N/A"),
                        }
                        for h in outer_loop_history
                    ]

                suggestions = llm.suggest_parameters(training_summary)

                # Save LLM debug HTML
                query_info = llm.get_last_query_info() if hasattr(llm, 'get_last_query_info') else None
                llm_prompt = query_info.get("prompt", "(no prompt)") if query_info else "(no query info)"
                llm_response = query_info.get("response_text", "(no response)") if query_info else "(no query info)"
                debug_dir = Path(f"runs/{run_dir}/debug_html")
                save_llm_debug_html(
                    iteration=outer_iter,
                    prompt=llm_prompt,
                    response_text=llm_response,
                    suggestions=suggestions,
                    summary_for_llm=training_summary,
                    save_path=debug_dir / f"iter_{outer_iter+1:02d}_llm.html",
                )

                if suggestions and suggestions.get("type") == "params":
                    new_params = suggestions.get("params", {})
                    rationale = suggestions.get("rationale", "No rationale")
                    if new_params:
                        print(f"[LLM] Rationale: {rationale}")
                        print(f"[LLM] Suggested weights: {new_params}")
                        new_weights_suggestion = {
                            "params": new_params,
                            "rationale": rationale,
                        }
                    else:
                        print("[LLM] No weight changes suggested")
                else:
                    print(f"[LLM] Unexpected suggestion type: {suggestions}")

        # Record history
        iter_record = {
            "outer_iter": outer_iter,
            "weights": dict(current_weights),
            "success_at_end": success_at_end,
            "success_once": success_once,
            "avg_return": avg_return,
            "eval_metrics": eval_metrics,
            "vlm_comment": vlm_comment,
            "learning_curve": result["learning_curve"],
        }

        # Apply new weights for next iteration
        if new_weights_suggestion is not None:
            old_weights = dict(current_weights)
            for k, v in new_weights_suggestion["params"].items():
                if k in current_weights:
                    current_weights[k] = v
            iter_record["new_weights"] = dict(current_weights)
            iter_record["rationale"] = new_weights_suggestion["rationale"]
            print(f"\n[Outer Iter {outer_iter+1}] Weight update:")
            print(f"  Old: {old_weights}")
            print(f"  New: {current_weights}")
        else:
            iter_record["new_weights"] = dict(current_weights)
            iter_record["rationale"] = "No change"
            outer_loop_history.append(iter_record)
            print(f"\n[Outer Iter {outer_iter+1}] LLM suggested no weight change. "
                  f"Stopping early (no benefit to re-training with identical weights).")
            break

        outer_loop_history.append(iter_record)

    # --- Save final results ---
    history_path = f"runs/{run_dir}/outer_loop_history.json"
    with open(history_path, "w") as f:
        json.dump(outer_loop_history, f, indent=2, default=str)
    print(f"\nOuter loop history saved to {history_path}")

    final_weights_path = f"runs/{run_dir}/final_weights.json"
    with open(final_weights_path, "w") as f:
        json.dump(current_weights, f, indent=2)
    print(f"Final weights saved to {final_weights_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("OUTER LOOP SUMMARY")
    print(f"{'='*60}")
    for record in outer_loop_history:
        i = record["outer_iter"]
        s_end = record["success_at_end"]
        s_once = record["success_once"]
        ar = record["avg_return"]
        print(f"  Iter {i+1}: success_end={s_end:.4f}, success_once={s_once:.4f}, return={ar:.4f}, weights={record['weights']}")
    print(f"{'='*60}")

    logger.close()
