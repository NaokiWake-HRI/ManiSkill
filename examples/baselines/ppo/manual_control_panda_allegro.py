"""
Keyboard manual control for PandaAllegro (Franka Panda + Allegro Hand).

Supports two control modes:
  1. pd_joint_target_delta_pos_arm_abs_hand (default for v2):
     - Arm: 7D joint delta (-1 to +1)
     - Hand: 16D normalized absolute joint pos (-1=lower_limit, +1=upper_limit)
     - Total: 23D flat action

  2. pd_ee_delta_pose (legacy):
     - Arm: 6D EE delta pose
     - Hand: 16D absolute joint pos (raw angles)
     - Total: 22D via action_dict

Usage:
    # v2 with new control mode (default):
    python manual_control_panda_allegro.py -e PickCubePandaAllegro-v2 --enable-sapien-viewer

    # v1 with legacy EE control:
    python manual_control_panda_allegro.py -e PickCubePandaAllegro-v1 -c pd_ee_delta_pose --enable-sapien-viewer

Controls:
    Arm:
        i/k  : joint 1 +/-     (or +x/-x in EE mode)
        j/l  : joint 2 +/-     (or +y/-y in EE mode)
        u/o  : joint 3 +/-     (or +z/-z in EE mode)
        1/2  : joint 4 +/-     (or rot x in EE mode)
        3/4  : joint 5 +/-     (or rot y in EE mode)
        5/6  : joint 6 +/-     (or rot z in EE mode)
        7/8  : joint 7 +/-     (joint mode only)

    Hand (coupled: index=middle=ring):
        g    : Close fingers (index+middle+ring curl)
        f    : Open fingers (index+middle+ring extend)
        t    : Close thumb
        y    : Open thumb

    Other:
        r    : Reset environment
        q    : Quit
"""
import argparse
import signal
import time

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt

signal.signal(signal.SIGINT, signal.SIG_DFL)

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common


# Coupled hand targets (8D: index[4] + thumb[4])
# Index finger open / closed
FINGERS_OPEN = np.array([0.0, 0.0, 0.0, 0.0])
FINGERS_CLOSED = np.array([0.3, 1.0, 1.0, 1.0])

# Thumb open / closed
THUMB_OPEN = np.array([0.83, 0.0, 0.0, 0.0])
THUMB_CLOSED = np.array([1.3, 0.7, 0.7, 1.2])  # joint12: max 1.396 (more pronation)

# Normalized versions [-1, +1] for pd_joint_target_delta_pos_arm_abs_hand mode
# These will be computed at runtime from joint limits
FINGERS_OPEN_NORM = None
FINGERS_CLOSED_NORM = None
THUMB_OPEN_NORM = None
THUMB_CLOSED_NORM = None

LERP_RATE = 0.1  # interpolation speed per frame


def flip_wrist(env):
    """Rotate wrist 180 degrees so palm faces up for easier viewing."""
    qpos = env.unwrapped.agent.robot.get_qpos().clone()
    qpos[..., 6] -= np.pi  # panda_joint7: 90deg -> -90deg
    env.unwrapped.agent.robot.set_qpos(qpos)


def coupled_8d_to_16d(fingers_4d, thumb_4d):
    """Expand 8D coupled hand target to 16D.
    index(0-3) = middle(4-7) = ring(8-11), thumb(12-15) independent."""
    return np.concatenate([fingers_4d, fingers_4d, fingers_4d, thumb_4d])


def raw_to_normalized(raw_16d, lower, upper):
    """Convert raw joint angles to normalized [-1, +1]."""
    return 2.0 * (raw_16d - lower) / (upper - lower) - 1.0


def compute_normalized_targets(env):
    """Compute normalized open/closed targets from joint limits."""
    global FINGERS_OPEN_NORM, FINGERS_CLOSED_NORM, THUMB_OPEN_NORM, THUMB_CLOSED_NORM
    qlimits = env.unwrapped.agent.robot.get_qlimits()[0].cpu().numpy()
    # Hand joints are indices 7:23
    hand_lower = qlimits[7:23, 0]
    hand_upper = qlimits[7:23, 1]

    open_16d = coupled_8d_to_16d(FINGERS_OPEN, THUMB_OPEN)
    closed_16d = coupled_8d_to_16d(FINGERS_CLOSED, THUMB_CLOSED)

    open_norm = raw_to_normalized(open_16d, hand_lower, hand_upper)
    closed_norm = raw_to_normalized(closed_16d, hand_lower, hand_upper)

    # Split back into finger/thumb parts (use first 4 of the 12 finger joints)
    FINGERS_OPEN_NORM = open_norm[:4]
    FINGERS_CLOSED_NORM = closed_norm[:4]
    THUMB_OPEN_NORM = open_norm[12:16]
    THUMB_CLOSED_NORM = closed_norm[12:16]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCubePandaAllegro-v2")
    parser.add_argument("-o", "--obs-mode", type=str, default="state")
    parser.add_argument("--reward-mode", type=str, default="dense")
    parser.add_argument("-c", "--control-mode", type=str, default=None,
                        help="Control mode. Default: auto-detect from env.")
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--enable-sapien-viewer", action="store_true")
    parser.add_argument("--record-dir", type=str)
    parser.add_argument("--arm-action-scale", type=float, default=0.5,
                        help="Scale for arm joint delta actions (-1 to +1)")
    parser.add_argument("--ee-action-scale", type=float, default=0.1,
                        help="Scale for EE delta actions (legacy mode)")
    return parser.parse_args()


def main():
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()

    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        render_mode=args.render_mode,
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    env: BaseEnv = gym.make(args.env_id, **env_kwargs)

    if args.record_dir:
        from mani_skill.utils.wrappers import RecordEpisode
        env = RecordEpisode(env, args.record_dir, render_mode=args.render_mode)

    control_mode = env.unwrapped.control_mode
    is_joint_mode = "joint" in control_mode and "ee" not in control_mode

    print("=" * 60)
    print(f"Env:          {args.env_id}")
    print(f"Control mode: {control_mode}")
    print(f"Action space: {env.action_space}")
    print(f"Mode:         {'Joint delta + Abs hand' if is_joint_mode else 'EE delta + Abs hand'}")
    print("=" * 60)
    print()
    if is_joint_mode:
        print("Arm (joint delta): i/k(j1) j/l(j2) u/o(j3) 1/2(j4) 3/4(j5) 5/6(j6) 7/8(j7)")
    else:
        print("Arm (EE delta):    i/k(±x) j/l(±y) u/o(±z) 1-6(rotation)")
    print("Hand (coupled: index=middle=ring):")
    print("      g(close fingers) f(open fingers)")
    print("      t(close thumb)   y(open thumb)")
    print("      r(reset) q(quit)")
    print("=" * 60)

    obs, _ = env.reset()
    after_reset = True

    # Compute normalized hand targets if using joint mode
    if is_joint_mode:
        compute_normalized_targets(env)
        fingers = FINGERS_OPEN_NORM.copy()
        thumb = THUMB_OPEN_NORM.copy()
    else:
        fingers = FINGERS_OPEN.copy()
        thumb = THUMB_OPEN.copy()

    # SAPIEN viewer (3D scene)
    if args.enable_sapien_viewer:
        env.render_human()

    # Check if agent has TouchLab sensors
    has_tl = hasattr(env.unwrapped.agent, 'tl_links') and len(env.unwrapped.agent.tl_links) > 0
    tl_labels = ["Index", "Middle", "Ring", "Thumb"]

    # Sensor bar chart (matplotlib)
    plt.ion()
    # Disable matplotlib default key shortcuts so they don't interfere
    for kmap in [
        "keymap.fullscreen", "keymap.home", "keymap.back",
        "keymap.forward", "keymap.pan", "keymap.zoom",
        "keymap.save", "keymap.grid", "keymap.yscale", "keymap.xscale",
    ]:
        for k in [c for c in plt.rcParams[kmap] if len(c) == 1 and c.islower()]:
            try:
                plt.rcParams[kmap].remove(k)
            except ValueError:
                pass

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.canvas.manager.set_window_title("TouchLab Sensors")
    bar_colors = ["#4285f4", "#ea4335", "#fbbc05", "#34a853"]
    bars = ax.bar(tl_labels, [0.0] * 4, color=bar_colors)
    ax.set_ylabel("Impulse (norm)")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    pressed_keys = set()

    def _on_key_press(event):
        pressed_keys.add(event.key)

    def _on_key_release(event):
        pressed_keys.discard(event.key)

    fig.canvas.mpl_connect("key_press_event", _on_key_press)
    fig.canvas.mpl_connect("key_release_event", _on_key_release)

    ARM_SCALE = args.arm_action_scale
    EE_SCALE = args.ee_action_scale
    control_timestep = env.unwrapped.control_timestep
    last_update_time = time.time()

    while True:
        current_time = time.time()
        if current_time - last_update_time < control_timestep:
            time.sleep(0.001)
            plt.pause(0.001)
            continue
        last_update_time = current_time

        if args.enable_sapien_viewer:
            env.render_human()

        pressed = pressed_keys

        if is_joint_mode:
            # --- Arm joint delta (7D), values in [-1, +1] ---
            arm_action = np.zeros(7)
            if "i" in pressed: arm_action[0] = ARM_SCALE
            if "k" in pressed: arm_action[0] = -ARM_SCALE
            if "j" in pressed: arm_action[1] = ARM_SCALE
            if "l" in pressed: arm_action[1] = -ARM_SCALE
            if "u" in pressed: arm_action[2] = ARM_SCALE
            if "o" in pressed: arm_action[2] = -ARM_SCALE
            if "1" in pressed: arm_action[3] = ARM_SCALE
            elif "2" in pressed: arm_action[3] = -ARM_SCALE
            if "3" in pressed: arm_action[4] = ARM_SCALE
            elif "4" in pressed: arm_action[4] = -ARM_SCALE
            if "5" in pressed: arm_action[5] = ARM_SCALE
            elif "6" in pressed: arm_action[5] = -ARM_SCALE
            if "7" in pressed: arm_action[6] = ARM_SCALE
            elif "8" in pressed: arm_action[6] = -ARM_SCALE
        else:
            # --- Arm EE delta pose (6D) ---
            arm_action = np.zeros(6)
            if "i" in pressed: arm_action[0] = EE_SCALE
            if "k" in pressed: arm_action[0] = -EE_SCALE
            if "j" in pressed: arm_action[1] = EE_SCALE
            if "l" in pressed: arm_action[1] = -EE_SCALE
            if "u" in pressed: arm_action[2] = EE_SCALE
            if "o" in pressed: arm_action[2] = -EE_SCALE
            if "1" in pressed: arm_action[3:6] = (1, 0, 0)
            elif "2" in pressed: arm_action[3:6] = (-1, 0, 0)
            elif "3" in pressed: arm_action[3:6] = (0, 1, 0)
            elif "4" in pressed: arm_action[3:6] = (0, -1, 0)
            elif "5" in pressed: arm_action[3:6] = (0, 0, 1)
            elif "6" in pressed: arm_action[3:6] = (0, 0, -1)

        # --- Hand (coupled) ---
        if is_joint_mode:
            f_open, f_closed = FINGERS_OPEN_NORM, FINGERS_CLOSED_NORM
            t_open, t_closed = THUMB_OPEN_NORM, THUMB_CLOSED_NORM
        else:
            f_open, f_closed = FINGERS_OPEN, FINGERS_CLOSED
            t_open, t_closed = THUMB_OPEN, THUMB_CLOSED

        if "g" in pressed:  # close fingers (index=middle=ring)
            fingers += (f_closed - fingers) * LERP_RATE
        if "f" in pressed:  # open fingers
            fingers += (f_open - fingers) * LERP_RATE
        if "t" in pressed:  # close thumb
            thumb += (t_closed - thumb) * LERP_RATE
        if "y" in pressed:  # open thumb
            thumb += (t_open - thumb) * LERP_RATE

        # --- Special keys ---
        if "r" in pressed:
            obs, _ = env.reset()
            if is_joint_mode:
                fingers = FINGERS_OPEN_NORM.copy()
                thumb = THUMB_OPEN_NORM.copy()
            else:
                fingers = FINGERS_OPEN.copy()
                thumb = THUMB_OPEN.copy()
            after_reset = True
            pressed_keys.discard("r")
            continue
        if "q" in pressed or "escape" in pressed:
            break

        # --- Build action and step ---
        hand_16d = coupled_8d_to_16d(fingers, thumb)

        if is_joint_mode:
            # Flat 23D action: arm(7) + hand(16)
            action = np.concatenate([arm_action, hand_16d])
            action = common.to_tensor(action).unsqueeze(0)
        else:
            # Dict action for EE mode
            action_dict = dict(arm=arm_action, hand=hand_16d)
            action_dict = common.to_tensor(action_dict)
            action = env.unwrapped.agent.controller.from_action_dict(action_dict)

        obs, reward, terminated, truncated, info = env.step(action)

        # Update sensor bar chart
        sensor_str = ""
        if has_tl:
            tl_impulse = env.unwrapped.agent.get_tl_impulse()
            tl_norms = torch.linalg.norm(tl_impulse, dim=-1)[0]  # first env
            for bar, val in zip(bars, tl_norms):
                bar.set_height(float(val))
            ymax = max(float(tl_norms.max()), 0.1) * 1.3
            ax.set_ylim(0, ymax)
            sensor_str = " TL[" + " ".join(
                f"{lbl}={v:.1f}" for lbl, v in zip(tl_labels, tl_norms)
            ) + "]"
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        # Compact status line
        info_str = " ".join(
            f"{k}={v.item() if hasattr(v, 'item') else v:.3f}"
            if isinstance(v, (float, int)) or hasattr(v, 'item') else f"{k}={v}"
            for k, v in info.items()
        )
        print(f"\rR={float(reward):.3f} | fingers={fingers} thumb={thumb}{sensor_str} | {info_str}",
              end="", flush=True)

    print()
    plt.close(fig)
    env.close()


if __name__ == "__main__":
    main()
