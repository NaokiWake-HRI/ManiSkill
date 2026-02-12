"""Scripted pick-and-place demo for PickCubePandaAllegro-v1.

Generates reference trajectories using a reactive phase-based heuristic.

Phases:
  approach  – move TCP above the cube (XY align + safe height)
  descend   – lower TCP to grasp height, pre-shape fingers
  grasp     – close fingers around the cube
  lift      – move TCP to goal position while maintaining grasp
  static    – hold still at the goal

Control mode: pd_ee_target_delta_pose
  arm:  6D normalised EE delta (pos 3D + rot 3D, [-1, 1] -> [-0.1, 0.1])
  hand: 16D absolute joint positions (rad)

  use_target=True keeps an internal target pose; rotation delta = 0
  means the wrist orientation is locked to the rest-keyframe value.

Usage
-----
    # Visualise the scripted policy:
    python -m mani_skill.examples.demo_scripted_pick_cube_allegro \\
        --render-mode human --shader rt-fast

    # Record trajectories (HDF5 + JSON, ManiSkill standard format):
    python -m mani_skill.examples.demo_scripted_pick_cube_allegro \\
        --num-episodes 100 --record-dir demos/PickCubePandaAllegro-v1

    # Record with video preview:
    python -m mani_skill.examples.demo_scripted_pick_cube_allegro \\
        --num-episodes 10 --record-dir demos/PickCubePandaAllegro-v1 \\
        --save-video

    # Convert recorded trajectories to a different control mode:
    python -m mani_skill.trajectory.replay_trajectory \\
        demos/PickCubePandaAllegro-v1/trajectory.h5 \\
        --save-traj -c pd_joint_delta_pos_coupled -o state
"""
from __future__ import annotations

import argparse

import numpy as np
import gymnasium as gym

import mani_skill.envs  # noqa: F401 – triggers env registration
from mani_skill.utils.wrappers import RecordEpisode

# =========================================================================== #
# Hand joint presets (16 DOF)
# Order: index[4] + middle[4] + ring[4] + thumb[4]
#
# Joint limits (from panda_allegro.urdf):
#   finger joints 0-11:
#     abduction [-0.47, 0.47]   MCP [-0.196, 1.61]
#     PIP       [-0.174, 1.709] DIP [-0.227, 1.618]
#   thumb joints 12-15:
#     rotation  [0.263, 1.396]  MCP [-0.105, 1.163]
#     PIP       [-0.189, 1.644] DIP [-0.162, 1.719]
# =========================================================================== #

HAND_OPEN = np.array(
    [
        # index
        0.0,  0.0,  0.0,  0.0,
        # middle
        0.0,  0.0,  0.0,  0.0,
        # ring
        0.0,  0.0,  0.0,  0.0,
        # thumb (pre-rotation matching rest keyframe)
        0.83, 0.0,  0.0,  0.0,
    ],
    dtype=np.float32,
)

# RL_project phase-1 deltas (0.7 x base pattern for fingers, thumb PIP/DIP only)
HAND_PRE_GRASP = np.array(
    [
        -0.14, 0.63, 0.70, 0.49,  # index:  0.7 * [-0.2, 0.9, 1.0, 0.7]
         0.00, 0.63, 0.70, 0.49,  # middle: 0.7 * [ 0.0, 0.9, 1.0, 0.7]
         0.14, 0.63, 0.70, 0.49,  # ring:   0.7 * [ 0.2, 0.9, 1.0, 0.7]
         0.83, 0.00, 0.40, 0.40,  # thumb:  rest + [0, 0, 0.4, 0.4]
    ],
    dtype=np.float32,
)

# RL_project full grasp (phase-1 + phase-2), clamped to URDF limits.
# thumb thj0: rest 0.83 + delta 1.4 = 2.23 -> clamp to URDF max 1.396
HAND_GRASP = np.array(
    [
        -0.20, 0.90, 1.00, 0.70,  # index:  [-0.2, 0.9, 1.0, 0.7]
         0.00, 0.90, 1.00, 0.70,  # middle: [ 0.0, 0.9, 1.0, 0.7]
         0.20, 0.90, 1.00, 0.70,  # ring:   [ 0.2, 0.9, 1.0, 0.7]
         1.396, 0.70, 0.40, 0.40, # thumb:  [max, 0.7, 0.4, 0.4]
    ],
    dtype=np.float32,
)

# =========================================================================== #
# Tuneable parameters
# =========================================================================== #

# Heights relative to cube centre (allegro_tcp is 7 cm below palm)
APPROACH_Z_OFFSET = 0.15   # m – safe height above cube for approach
GRASP_Z_OFFSET    = 0.1   # m – TCP height above cube centre for grasping
GRASP_XY_OFFSET   = np.array([-0.15, -0.05])  # m – TCP XY offset from cube (negative X = 手前)

# Proportional gain for arm EE position delta (normalised action space)
ARM_GAIN = 1.0
# (rotation delta = 0; pd_ee_target_delta_pose keeps orientation locked)

# Phase-transition distance thresholds (metres)
APPROACH_THRESH = 0.02
DESCEND_THRESH  = 0.015
GOAL_THRESH     = 0.03

# Finger closing schedule (steps)
GRASP_CLOSE_STEPS = 15   # steps to interpolate pre-grasp -> grasp
GRASP_HOLD_STEPS  = 5    # extra steps to consolidate the grip
DESCEND_BLEND     = 10   # steps to blend open -> pre-grasp during descent

# Hold at goal
STATIC_HOLD_STEPS = 10



# =========================================================================== #
# Scripted policy
# =========================================================================== #


def scripted_policy(env, ps: dict) -> np.ndarray:
    """Compute one action from environment internal state.

    Args:
        env: gymnasium environment (may be wrapped; unwrapped is accessed).
        ps:  mutable phase-state dict.  Must contain ``phase``, ``step``.

    Returns:
        action: np.ndarray of shape (22,) — [arm_pos(3) arm_rot(3) hand(16)].
    """
    base_env = env.unwrapped
    tcp_pose = base_env.agent.tcp_pose
    tcp = tcp_pose.p[0].cpu().numpy()
    cube = base_env.cube.pose.p[0].cpu().numpy()
    goal = base_env.goal_site.pose.p[0].cpu().numpy()

    phase = ps["phase"]
    step  = ps["step"]

    # ---- per-phase target & hand configuration ---- #
    if phase == "approach":
        target = np.array([cube[0] + GRASP_XY_OFFSET[0], cube[1] + GRASP_XY_OFFSET[1], cube[2] + APPROACH_Z_OFFSET])
        hand = HAND_OPEN.copy()
        if (np.linalg.norm(tcp[:2] - target[:2]) < APPROACH_THRESH
                and abs(tcp[2] - target[2]) < APPROACH_THRESH):
            ps.update(phase="descend", step=0)

    elif phase == "descend":
        target = np.array([cube[0] + GRASP_XY_OFFSET[0], cube[1] + GRASP_XY_OFFSET[1], cube[2] + GRASP_Z_OFFSET])
        t = min(1.0, step / float(DESCEND_BLEND))
        hand = HAND_OPEN + t * (HAND_PRE_GRASP - HAND_OPEN)
        if np.linalg.norm(tcp - target) < DESCEND_THRESH:
            ps.update(phase="grasp", step=0)
        else:
            ps["step"] += 1

    elif phase == "grasp":
        target = np.array([cube[0] + GRASP_XY_OFFSET[0], cube[1] + GRASP_XY_OFFSET[1], cube[2] + GRASP_Z_OFFSET])
        t = min(1.0, step / float(GRASP_CLOSE_STEPS))
        hand = HAND_PRE_GRASP + t * (HAND_GRASP - HAND_PRE_GRASP)
        ps["step"] += 1
        if step >= GRASP_CLOSE_STEPS + GRASP_HOLD_STEPS:
            ps.update(phase="lift", step=0)

    elif phase == "lift":
        target = goal.copy()
        hand = HAND_GRASP.copy()
        if np.linalg.norm(tcp - target) < GOAL_THRESH:
            ps.update(phase="static", step=0)
        else:
            ps["step"] += 1

    elif phase == "static":
        target = goal.copy()
        hand = HAND_GRASP.copy()
        ps["step"] += 1

    else:
        raise ValueError(f"Unknown phase: {phase}")

    # ---- arm position delta ---- #
    pos_delta = target - tcp
    arm_pos = np.clip(pos_delta * ARM_GAIN, -1.0, 1.0).astype(np.float32)

    # ---- arm rotation delta: zero keeps wrist locked (pd_ee_target_delta_pose) ---- #
    arm_rot = np.zeros(3, dtype=np.float32)

    return np.concatenate([arm_pos, arm_rot, hand])


# =========================================================================== #
# Main
# =========================================================================== #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scripted demo for PickCubePandaAllegro-v1"
    )
    parser.add_argument("--num-episodes", "-n", type=int, default=10)
    parser.add_argument("--record-dir", type=str, default=None)
    parser.add_argument(
        "--render-mode", type=str, default=None,
        choices=["human", "rgb_array"],
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shader", type=str, default="default")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    env_kwargs = dict(
        obs_mode="state_dict",
        control_mode="pd_ee_target_delta_pose",
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
    )
    env = gym.make("PickCubePandaAllegro-v1", **env_kwargs)

    if args.record_dir is not None:
        env = RecordEpisode(
            env,
            output_dir=args.record_dir,
            save_trajectory=True,
            save_video=args.save_video,
            source_type="motionplanning",
            source_desc="scripted phase-based pick-and-place for Allegro hand",
        )

    print(f"Action space : {env.action_space}")
    print(f"Control mode : {env.unwrapped.control_mode}")

    success_count = 0
    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + ep)

        ps = {"phase": "approach", "step": 0}

        for step_i in range(100):  # max_episode_steps = 100
            action = scripted_policy(env, ps)
            obs, reward, terminated, truncated, info = env.step(action)

            if args.render_mode == "human":
                env.render()

            if args.verbose:
                print(
                    f"  [{step_i:3d}] phase={ps['phase']:8s} "
                    f"step={ps['step']:3d}  r={float(reward):.3f}"
                )

            if terminated or truncated:
                break

        # Handle tensor / scalar success flag
        suc = info.get("success", False)
        if hasattr(suc, "item"):
            suc = suc.item()
        suc = bool(suc)

        if suc:
            success_count += 1
        tag = "OK" if suc else f"FAIL(phase={ps['phase']})"
        print(f"ep {ep:3d}: {tag}  steps={step_i + 1}")

    print(f"\nSuccess: {success_count}/{args.num_episodes}")
    env.close()


if __name__ == "__main__":
    main()
