"""
Reward wrapper for iterative VLM/LLM reward tuning (debug version).

Same as reward_wrapper.py but also exposes raw component values via info,
enabling replay buffer reward recomputation when weights change.

Supported tasks: PickCube, PushCube, OpenCabinetDoor/Drawer
"""

import gymnasium as gym
import torch
from typing import Dict, List, Optional


# Default weights per task
TASK_DEFAULTS = {
    "PickCube": {
        "w_reach": 1.0,
        "w_grasp": 1.0,
        "w_place": 1.0,
        "w_static": 1.0,
        "w_success": 5.0,
    },
    "PushCube": {
        "w_reach": 1.0,
        "w_push": 1.0,
        "w_z_keep": 1.0,
        "w_success": 4.0,
    },
    "OpenCabinetDoor": {
        "w_reach": 1.0,
        "w_open": 1.0,
        "w_static": 3.0,
        "w_success": 5.0,
    },
    "OpenCabinetDrawer": {
        "w_reach": 1.0,
        "w_open": 1.0,
        "w_static": 3.0,
        "w_success": 5.0,
    },
}


def _resolve_task_id(env_id: str) -> str:
    """Extract task name from env_id (e.g. 'PickCube-v1' -> 'PickCube')."""
    name = env_id.split("-")[0]
    for key in TASK_DEFAULTS:
        if key in name:
            return key
    raise ValueError(
        f"Unsupported task: {env_id}. "
        f"Supported: {list(TASK_DEFAULTS.keys())}"
    )


class RewardWrapper(gym.Wrapper):
    """
    Computes custom reward from env internal state with configurable weights.

    Works with ManiSkill GPU-parallel environments (batched tensors).
    Weights can be updated at runtime by LLM tuner.

    Must be placed BEFORE ManiSkillVectorEnv in the wrapper chain.
    Use with reward_mode="none".
    """

    def __init__(
        self,
        env,
        env_id: str,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(env)
        self.task_id = _resolve_task_id(env_id)
        self.weights = dict(TASK_DEFAULTS[self.task_id])
        if weights is not None:
            self.weights.update(weights)
        self._last_breakdown: Dict[str, float] = {}

        # Store initial non-success weight sum for normalization.
        # When LLM changes weights, we scale the reward so its magnitude
        # stays consistent with the initial scale (prevents PPO instability).
        self._initial_component_sum = self._component_weight_sum()

        # Select compute function and component key names (excluding w_success)
        self._compute_fn = {
            "PickCube": self._compute_pick_cube,
            "PushCube": self._compute_push_cube,
            "OpenCabinetDoor": self._compute_open_cabinet,
            "OpenCabinetDrawer": self._compute_open_cabinet,
        }[self.task_id]
        self._component_keys: List[str] = [
            k for k in self.weights if k != "w_success"
        ]
        # Last raw component values (num_envs,) each, for replay buffer storage
        self._last_components: Dict[str, torch.Tensor] = {}
        self._last_success: Optional[torch.Tensor] = None

    def _component_weight_sum(self) -> float:
        """Sum of non-success weights (used for normalization)."""
        return sum(v for k, v in self.weights.items() if k != "w_success")

    def _norm_scale(self) -> float:
        """Scale factor to keep reward magnitude consistent after weight changes."""
        current_sum = self._component_weight_sum()
        if current_sum < 1e-8:
            return 1.0
        return self._initial_component_sum / current_sum

    @property
    def component_keys(self) -> List[str]:
        """Ordered list of component weight names (excluding w_success)."""
        return self._component_keys

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        reward = self._compute_fn(info)
        info["reward_breakdown"] = self._last_breakdown
        info["reward_components"] = self._last_components
        info["reward_success"] = self._last_success
        return obs, reward, terminated, truncated, info

    def get_weights(self) -> Dict[str, float]:
        return dict(self.weights)

    def update_weights(self, new_weights: Dict[str, float]):
        for k, v in new_weights.items():
            if k in self.weights:
                self.weights[k] = v
        print(f"[RewardWrapper] Updated weights: {self.weights}")

    def compute_reward_from_components(
        self,
        components: Dict[str, torch.Tensor],
        success: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute reward from stored raw component values and current weights."""
        w = self.weights
        scale = self._norm_scale()
        reward = torch.zeros_like(success, dtype=torch.float32)
        for key in self._component_keys:
            reward += w[key] * components[key]
        reward *= scale
        # IMPORTANT: Apply scale to success reward too to maintain Q value stability
        reward[success] = w["w_success"] * scale
        return reward

    # --- PickCube ---
    def _compute_pick_cube(self, info: dict) -> torch.Tensor:
        base = self.env.unwrapped
        w = self.weights

        # reach: tcp -> cube distance
        tcp_to_obj_dist = torch.linalg.norm(
            base.cube.pose.p - base.agent.tcp_pose.p, axis=1
        )
        reach_r = 1 - torch.tanh(5 * tcp_to_obj_dist)

        # grasp
        grasp_r = info["is_grasped"].float()

        # place: cube -> goal distance (gated by grasp)
        obj_to_goal_dist = torch.linalg.norm(
            base.goal_site.pose.p - base.cube.pose.p, axis=1
        )
        place_r = (1 - torch.tanh(5 * obj_to_goal_dist)) * grasp_r

        # static: robot static & object placed
        static_r = info["is_robot_static"].float() * info["is_obj_placed"].float()

        # Store raw component values (before weighting)
        self._last_components = {
            "w_reach": reach_r, "w_grasp": grasp_r,
            "w_place": place_r, "w_static": static_r,
        }
        self._last_success = info["success"]

        scale = self._norm_scale()
        reward = scale * (
            w["w_reach"] * reach_r
            + w["w_grasp"] * grasp_r
            + w["w_place"] * place_r
            + w["w_static"] * static_r
        )
        reward[info["success"]] = w["w_success"] * scale

        self._last_breakdown = {
            "reach": (scale * w["w_reach"] * reach_r).mean().item(),
            "grasp": (scale * w["w_grasp"] * grasp_r).mean().item(),
            "place": (scale * w["w_place"] * place_r).mean().item(),
            "static": (scale * w["w_static"] * static_r).mean().item(),
            "norm_scale": scale,
        }
        return reward

    # --- PushCube ---
    def _compute_push_cube(self, info: dict) -> torch.Tensor:
        base = self.env.unwrapped
        w = self.weights

        # reach: tcp -> push pose (behind the cube)
        push_offset = torch.tensor(
            [-base.cube_half_size - 0.005, 0, 0], device=base.device
        )
        push_pose_p = base.obj.pose.p + push_offset
        tcp_to_push_dist = torch.linalg.norm(
            push_pose_p - base.agent.tcp.pose.p, axis=1
        )
        reach_r = 1 - torch.tanh(5 * tcp_to_push_dist)

        # push: obj xy -> goal xy (gated by reached)
        reached = (tcp_to_push_dist < 0.01).float()
        obj_to_goal_dist = torch.linalg.norm(
            base.obj.pose.p[..., :2] - base.goal_region.pose.p[..., :2], axis=1
        )
        push_r = (1 - torch.tanh(5 * obj_to_goal_dist)) * reached

        # z_keep: keep cube on table (gated by reached)
        z_deviation = torch.abs(base.obj.pose.p[..., 2] - base.cube_half_size)
        z_r = (1 - torch.tanh(5 * z_deviation)) * push_r  # scale with push progress

        self._last_components = {
            "w_reach": reach_r, "w_push": push_r, "w_z_keep": z_r,
        }
        self._last_success = info["success"]

        scale = self._norm_scale()
        reward = scale * (
            w["w_reach"] * reach_r
            + w["w_push"] * push_r
            + w["w_z_keep"] * z_r
        )
        reward[info["success"]] = w["w_success"] * scale

        self._last_breakdown = {
            "reach": (scale * w["w_reach"] * reach_r).mean().item(),
            "push": (scale * w["w_push"] * push_r).mean().item(),
            "z_keep": (scale * w["w_z_keep"] * z_r).mean().item(),
            "norm_scale": scale,
        }
        return reward

    # --- OpenCabinetDoor / OpenCabinetDrawer ---
    def _compute_open_cabinet(self, info: dict) -> torch.Tensor:
        base = self.env.unwrapped
        w = self.weights

        # reach: tcp -> handle position (already computed by evaluate())
        handle_pos = info["handle_link_pos"]
        tcp_to_handle_dist = torch.linalg.norm(
            base.agent.tcp.pose.p - handle_pos, axis=1
        )
        reach_r = 1 - torch.tanh(5 * tcp_to_handle_dist)

        # open: progress toward target_qpos
        amount_to_open_left = torch.div(
            base.target_qpos - base.handle_link.joint.qpos,
            base.target_qpos,
        )
        open_r = 2 * (1 - amount_to_open_left)
        # if joint opened even a little, replace reach with constant
        reach_r[amount_to_open_left < 0.999] = 2.0

        # static bonus when open enough
        static_r = info["open_enough"].float()

        self._last_components = {
            "w_reach": reach_r, "w_open": open_r, "w_static": static_r,
        }
        self._last_success = info["success"]

        scale = self._norm_scale()
        reward = scale * (
            w["w_reach"] * reach_r
            + w["w_open"] * open_r
            + w["w_static"] * static_r
        )
        reward[info["success"]] = w["w_success"] * scale

        self._last_breakdown = {
            "reach": (scale * w["w_reach"] * reach_r).mean().item(),
            "open": (scale * w["w_open"] * open_r).mean().item(),
            "static": (scale * w["w_static"] * static_r).mean().item(),
            "norm_scale": scale,
        }
        return reward
