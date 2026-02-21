"""
Action wrapper for PandaAllegro coupled hand control.

Converts an 8D action space (6 arm EE delta + 2 hand scalars) to the
full 22D flat Box action expected by pd_ee_delta_pose control mode.

ManiSkill's pd_ee_delta_pose uses CombinedController which flattens
dict(arm=Box(6), hand=Box(16)) into a single Box(22).
The first 6 dims are arm EE delta, the last 16 are hand joint positions.

Hand scalar mapping (same limits as manual_control_panda_allegro.py):
  action[6]: finger_scalar in [-1, 1]
    -1 = FINGERS_OPEN  = [0.0, 0.0, 0.0, 0.0]
    +1 = FINGERS_CLOSED = [0.3, 1.0, 1.0, 1.0]
    Replicated 3x for index/middle/ring = 12D

  action[7]: thumb_scalar in [-1, 1]
    -1 = THUMB_OPEN  = [0.83, 0.0, 0.0, 0.0]
    +1 = THUMB_CLOSED = [1.3, 0.7, 0.7, 1.2]
    = 4D

Total hand: 12D + 4D = 16D absolute joint positions.
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


# Same limits as manual_control_panda_allegro.py
FINGERS_OPEN = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
FINGERS_CLOSED = np.array([0.3, 1.0, 1.0, 1.0], dtype=np.float32)
THUMB_OPEN = np.array([0.83, 0.0, 0.0, 0.0], dtype=np.float32)
THUMB_CLOSED = np.array([1.3, 0.7, 0.7, 1.2], dtype=np.float32)

# Precompute for fast interpolation
FINGERS_RANGE = FINGERS_CLOSED - FINGERS_OPEN
THUMB_RANGE = THUMB_CLOSED - THUMB_OPEN


def expand_hand_scalars(finger_scalar, thumb_scalar):
    """Convert 2D hand scalars to 16D absolute joint positions.

    Args:
        finger_scalar: (batch,) tensor in [-1, 1]
        thumb_scalar: (batch,) tensor in [-1, 1]

    Returns:
        hand_16d: (batch, 16) tensor of absolute joint positions
    """
    # Map [-1, 1] -> [0, 1]
    f_alpha = (finger_scalar + 1.0) / 2.0  # (batch,)
    t_alpha = (thumb_scalar + 1.0) / 2.0   # (batch,)

    device = finger_scalar.device

    fingers_open = torch.tensor(FINGERS_OPEN, device=device)
    fingers_range = torch.tensor(FINGERS_RANGE, device=device)
    thumb_open = torch.tensor(THUMB_OPEN, device=device)
    thumb_range = torch.tensor(THUMB_RANGE, device=device)

    # Interpolate: OPEN + alpha * (CLOSED - OPEN)
    fingers_4d = fingers_open + f_alpha.unsqueeze(-1) * fingers_range  # (batch, 4)
    thumb_4d = thumb_open + t_alpha.unsqueeze(-1) * thumb_range        # (batch, 4)

    # Replicate fingers for index/middle/ring + thumb
    hand_16d = torch.cat([fingers_4d, fingers_4d, fingers_4d, thumb_4d], dim=-1)
    return hand_16d


class CoupledAllegroActionWrapper(gym.ActionWrapper):
    """Wraps pd_ee_delta_pose env to expose 8D action space.

    pd_ee_delta_pose uses CombinedController which produces a flat Box(22):
      [0:6]  = arm EE delta pose
      [6:22] = hand absolute joint positions

    This wrapper exposes Box(8):
      [0:6] = arm EE delta pose (passthrough)
      [6]   = finger group scalar [-1=open, 1=closed]
      [7]   = thumb scalar [-1=open, 1=closed]

    action() expands 8D -> 22D flat tensor for the env.
    """

    ARM_DIM = 6
    HAND_DIM = 16
    RAW_DIM = ARM_DIM + HAND_DIM  # 22

    def __init__(self, env):
        super().__init__(env)
        orig_space = env.action_space

        # --- Contract check: input must be Box(22) from CombinedController ---
        if not isinstance(orig_space, spaces.Box):
            raise ValueError(
                f"CoupledAllegroActionWrapper: expected Box action space, "
                f"got {type(orig_space).__name__}. "
                f"Ensure control_mode=pd_ee_delta_pose (CombinedController -> flat Box)."
            )
        actual_dim = orig_space.shape[-1]
        if actual_dim != self.RAW_DIM:
            raise ValueError(
                f"CoupledAllegroActionWrapper: expected action dim={self.RAW_DIM} "
                f"(arm={self.ARM_DIM} + hand={self.HAND_DIM}), got dim={actual_dim}. "
                f"Check control_mode and robot configuration."
            )

        # Extract arm bounds from first 6 dims
        arm_low = orig_space.low[..., :self.ARM_DIM]
        arm_high = orig_space.high[..., :self.ARM_DIM]

        # Build 8D action space: 6 arm + 2 hand scalars
        if orig_space.low.ndim == 1:
            # Single env: shape (22,) -> (8,)
            low = np.concatenate([arm_low, np.array([-1.0, -1.0], dtype=np.float32)])
            high = np.concatenate([arm_high, np.array([1.0, 1.0], dtype=np.float32)])
        else:
            # Batched env: shape (num_envs, 22) -> (num_envs, 8)
            n = orig_space.low.shape[0]
            scalar_low = np.full((n, 2), -1.0, dtype=np.float32)
            scalar_high = np.full((n, 2), 1.0, dtype=np.float32)
            low = np.concatenate([arm_low, scalar_low], axis=-1)
            high = np.concatenate([arm_high, scalar_high], axis=-1)

        self.action_space = spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )
        # single_action_space for ManiSkill vector env compatibility
        single_arm_low = orig_space.low.reshape(-1, orig_space.shape[-1])[0, :self.ARM_DIM]
        single_arm_high = orig_space.high.reshape(-1, orig_space.shape[-1])[0, :self.ARM_DIM]
        self.single_action_space = spaces.Box(
            low=np.concatenate([single_arm_low, np.array([-1.0, -1.0], dtype=np.float32)]).astype(np.float32),
            high=np.concatenate([single_arm_high, np.array([1.0, 1.0], dtype=np.float32)]).astype(np.float32),
            dtype=np.float32,
        )

    def action(self, action):
        """Convert 8D action to 22D flat tensor (arm 6D | hand 16D)."""
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()

        if action.ndim == 1:
            action = action.unsqueeze(0)

        arm_action = action[:, :self.ARM_DIM]
        finger_scalar = action[:, self.ARM_DIM]
        thumb_scalar = action[:, self.ARM_DIM + 1]

        hand_16d = expand_hand_scalars(finger_scalar, thumb_scalar)

        # Return flat tensor [arm(6) | hand(16)] = 22D
        return torch.cat([arm_action, hand_16d], dim=-1)
