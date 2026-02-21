"""
Keyboard manual control for PandaAllegro (Franka Panda + Allegro Hand).

Hand coupling (same as pd_joint_delta_pos_coupled):
  - Index finger (joints 0-3) controls middle (4-7) and ring (8-11) too
  - Thumb (joints 12-15) is independent
  => Effective hand DOFs: 8 (index 4 + thumb 4)

Usage:
    python manual_control_panda_allegro.py
    python manual_control_panda_allegro.py --enable-sapien-viewer

Controls:
    Arm (EE delta pose):
        i/k  : +x / -x
        j/l  : +y / -y
        u/o  : +z / -z
        1/2  : rotate around x
        3/4  : rotate around y
        5/6  : rotate around z

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

LERP_RATE = 0.1  # interpolation speed per frame


def flip_wrist(env):
    """Rotate wrist 180 degrees so palm faces up for easier viewing."""
    qpos = env.unwrapped.agent.robot.get_qpos().clone()
    qpos[..., 6] -= np.pi  # panda_joint7: 90deg -> -90deg
    env.unwrapped.agent.robot.set_qpos(qpos)


def coupled_8d_to_16d(fingers_4d, thumb_4d):
    """Expand 8D coupled hand target to 16D absolute joint positions.
    index(0-3) = middle(4-7) = ring(8-11), thumb(12-15) independent."""
    return np.concatenate([fingers_4d, fingers_4d, fingers_4d, thumb_4d])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCubePandaAllegro-v1")
    parser.add_argument("-o", "--obs-mode", type=str, default="state")
    parser.add_argument("--reward-mode", type=str, default="dense")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--enable-sapien-viewer", action="store_true")
    parser.add_argument("--record-dir", type=str)
    parser.add_argument("--ee-action-scale", type=float, default=0.1)
    return parser.parse_args()


def main():
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()

    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
    )

    if args.record_dir:
        from mani_skill.utils.wrappers import RecordEpisode
        env = RecordEpisode(env, args.record_dir, render_mode=args.render_mode)

    print("=" * 60)
    print(f"Env:          {args.env_id}")
    print(f"Control mode: {env.control_mode}")
    print(f"Action space: {env.action_space}")
    print("=" * 60)
    print()
    print("Arm:  i/k(±x) j/l(±y) u/o(±z)  1-6(rotation)")
    print("Hand (coupled: index=middle=ring):")
    print("      g(close fingers) f(open fingers)")
    print("      t(close thumb)   y(open thumb)")
    print("      r(reset) q(quit)")
    print("=" * 60)

    obs, _ = env.reset()
    after_reset = True

    # Coupled hand state (8D: index[4] + thumb[4])
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

        # --- Arm EE delta pose (6D) ---
        ee_action = np.zeros(6)
        if "i" in pressed: ee_action[0] = EE_SCALE
        if "k" in pressed: ee_action[0] = -EE_SCALE
        if "j" in pressed: ee_action[1] = EE_SCALE
        if "l" in pressed: ee_action[1] = -EE_SCALE
        if "u" in pressed: ee_action[2] = EE_SCALE
        if "o" in pressed: ee_action[2] = -EE_SCALE
        if "1" in pressed: ee_action[3:6] = (1, 0, 0)
        elif "2" in pressed: ee_action[3:6] = (-1, 0, 0)
        elif "3" in pressed: ee_action[3:6] = (0, 1, 0)
        elif "4" in pressed: ee_action[3:6] = (0, -1, 0)
        elif "5" in pressed: ee_action[3:6] = (0, 0, 1)
        elif "6" in pressed: ee_action[3:6] = (0, 0, -1)

        # --- Hand (coupled) ---
        if "g" in pressed:  # close fingers (index=middle=ring)
            fingers += (FINGERS_CLOSED - fingers) * LERP_RATE
        if "f" in pressed:  # open fingers
            fingers += (FINGERS_OPEN - fingers) * LERP_RATE
        if "t" in pressed:  # close thumb
            thumb += (THUMB_CLOSED - thumb) * LERP_RATE
        if "y" in pressed:  # open thumb
            thumb += (THUMB_OPEN - thumb) * LERP_RATE

        # --- Special keys ---
        if "r" in pressed:
            obs, _ = env.reset()
            fingers = FINGERS_OPEN.copy()
            thumb = THUMB_OPEN.copy()
            after_reset = True
            pressed_keys.discard("r")
            continue
        if "q" in pressed or "escape" in pressed:
            break

        # --- Build action and step ---
        hand_16d = coupled_8d_to_16d(fingers, thumb)
        action_dict = dict(arm=ee_action, hand=hand_16d)
        action_dict = common.to_tensor(action_dict)
        action = env.agent.controller.from_action_dict(action_dict)

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
