"""
Reward wrapper for iterative VLM/LLM reward tuning.

Computes custom reward from env internal state + info dict with tunable weights.
Use with reward_mode="none" to disable built-in reward.

Supported tasks: PickCube, PushCube, OpenCabinetDoor/Drawer, UnitreeG1PlaceAppleInBowl, AnymalC-Reach, PegInsertionSide, PushT
"""

import gymnasium as gym
import sapien
import torch
from typing import Dict, Optional


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
    "UnitreeG1PlaceAppleInBowl": {
        "w_reach": 1.0,
        "w_grasp": 1.0,
        "w_place": 1.0,
        "w_release": 1.0,
        "w_success": 8.0,
    },
    "AnymalC": {
        "w_reach": 2.0,
        "w_vel_z_penalty": 2.0,
        "w_ang_vel_penalty": 0.05,
        "w_contact_penalty": 1.0,
        "w_qpos_penalty": 0.05,
    },
    "PegInsertionSide": {
        "w_reach": 1.0,
        "w_grasp": 1.0,
        "w_pre_insertion": 3.0,
        "w_insertion": 5.0,
        "w_success": 10.0,
    },
    "PushT": {
        "w_rotation": 0.5,
        "w_position": 0.5,
        "w_tcp_guide": 0.05,
        "w_success": 3.0,
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

        # Select compute function
        self._compute_fn = {
            "PickCube": self._compute_pick_cube,
            "PushCube": self._compute_push_cube,
            "OpenCabinetDoor": self._compute_open_cabinet,
            "OpenCabinetDrawer": self._compute_open_cabinet,
            "UnitreeG1PlaceAppleInBowl": self._compute_unitree_place_apple,
            "AnymalC": self._compute_anymalc_reach,
            "PegInsertionSide": self._compute_peg_insertion,
            "PushT": self._compute_push_t,
        }[self.task_id]

    def _component_weight_sum(self) -> float:
        """Sum of non-success weights (used for normalization)."""
        return sum(v for k, v in self.weights.items() if k != "w_success")

    def _norm_scale(self) -> float:
        """Scale factor to keep reward magnitude consistent after weight changes."""
        current_sum = self._component_weight_sum()
        if current_sum < 1e-8:
            return 1.0
        return self._initial_component_sum / current_sum

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        reward = self._compute_fn(info)
        info["reward_breakdown"] = self._last_breakdown
        return obs, reward, terminated, truncated, info

    def get_weights(self) -> Dict[str, float]:
        return dict(self.weights)

    def update_weights(self, new_weights: Dict[str, float]):
        for k, v in new_weights.items():
            if k in self.weights:
                self.weights[k] = v
        print(f"[RewardWrapper] Updated weights: {self.weights}")

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

        # static: continuous velocity penalty * object placed
        qvel = base.agent.robot.get_qvel()
        if base.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif base.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_r = (1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))) * info["is_obj_placed"].float()

        scale = self._norm_scale()
        reward = scale * (
            w["w_reach"] * reach_r
            + w["w_grasp"] * grasp_r
            + w["w_place"] * place_r
            + w["w_static"] * static_r
        )
        reward[info["success"]] = w["w_success"]

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

        scale = self._norm_scale()
        reward = scale * (
            w["w_reach"] * reach_r
            + w["w_push"] * push_r
            + w["w_z_keep"] * z_r
        )
        reward[info["success"]] = w["w_success"]

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

        scale = self._norm_scale()
        reward = scale * (
            w["w_reach"] * reach_r
            + w["w_open"] * open_r
            + w["w_static"] * static_r
        )
        reward[info["success"]] = w["w_success"]

        self._last_breakdown = {
            "reach": (scale * w["w_reach"] * reach_r).mean().item(),
            "open": (scale * w["w_open"] * open_r).mean().item(),
            "static": (scale * w["w_static"] * static_r).mean().item(),
            "norm_scale": scale,
        }
        return reward

    # --- UnitreeG1PlaceAppleInBowl ---
    def _compute_unitree_place_apple(self, info: dict) -> torch.Tensor:
        base = self.env.unwrapped
        w = self.weights

        # reach: tcp -> apple distance
        tcp_to_obj_dist = torch.linalg.norm(
            base.apple.pose.p - base.agent.right_tcp.pose.p, axis=1
        )
        reach_r = 1 - torch.tanh(5 * tcp_to_obj_dist)

        # grasp
        grasp_r = info["is_grasped"].float()

        # place: apple -> bowl distance (gated by grasp)
        obj_to_goal_dist = torch.linalg.norm(
            (base.bowl.pose.p + torch.tensor([0, 0, 0.15], device=base.device))
            - base.apple.pose.p,
            axis=1,
        )
        place_r = (1 - torch.tanh(5 * obj_to_goal_dist)) * grasp_r

        # release: encourage opening hand when above bowl
        obj_high_above_bowl = obj_to_goal_dist < 0.025
        grasp_release_reward = 1 - torch.tanh(
            base.agent.right_hand_dist_to_open_grasp()
        )
        release_r = grasp_release_reward * obj_high_above_bowl.float()

        scale = self._norm_scale()
        reward = scale * (
            w["w_reach"] * reach_r
            + w["w_grasp"] * grasp_r
            + w["w_place"] * place_r
            + w["w_release"] * release_r
        )
        reward[info["success"]] = w["w_success"]

        self._last_breakdown = {
            "reach": (scale * w["w_reach"] * reach_r).mean().item(),
            "grasp": (scale * w["w_grasp"] * grasp_r).mean().item(),
            "place": (scale * w["w_place"] * place_r).mean().item(),
            "release": (scale * w["w_release"] * release_r).mean().item(),
            "norm_scale": scale,
        }
        return reward

    # --- AnymalC-Reach ---
    def _compute_anymalc_reach(self, info: dict) -> torch.Tensor:
        base = self.env.unwrapped
        w = self.weights

        # reach: robot -> goal distance
        robot_to_goal_dist = info["robot_to_goal_dist"]
        reach_r = 1 - torch.tanh(1 * robot_to_goal_dist)

        # penalties
        lin_vel_z_l2 = torch.square(base.agent.robot.root_linear_velocity[:, 2])
        ang_vel_xy_l2 = (
            torch.square(base.agent.robot.root_angular_velocity[:, :2])
        ).sum(axis=1)

        # undesired contacts (knee links hitting ground)
        forces = base.agent.robot.get_net_contact_forces(
            base._UNDESIRED_CONTACT_LINK_NAMES
        )
        contact_penalty = (torch.norm(forces, dim=-1).max(-1).values > 1.0).float()

        # qpos deviation from default standing pose
        qpos_penalty = torch.linalg.norm(
            base.agent.robot.qpos - base.default_qpos, axis=1
        )

        scale = self._norm_scale()
        reward = (
            1.0
            + scale * w["w_reach"] * reach_r
            - w["w_vel_z_penalty"] * lin_vel_z_l2
            - w["w_ang_vel_penalty"] * ang_vel_xy_l2
            - w["w_contact_penalty"] * contact_penalty
            - w["w_qpos_penalty"] * qpos_penalty
        )
        reward[info["fail"]] = 0

        self._last_breakdown = {
            "reach": (scale * w["w_reach"] * reach_r).mean().item(),
            "vel_z_penalty": (w["w_vel_z_penalty"] * lin_vel_z_l2).mean().item(),
            "ang_vel_penalty": (w["w_ang_vel_penalty"] * ang_vel_xy_l2).mean().item(),
            "contact_penalty": (w["w_contact_penalty"] * contact_penalty).mean().item(),
            "qpos_penalty": (w["w_qpos_penalty"] * qpos_penalty).mean().item(),
            "norm_scale": scale,
        }
        return reward

    # --- PegInsertionSide ---
    def _compute_peg_insertion(self, info: dict) -> torch.Tensor:
        base = self.env.unwrapped
        w = self.weights

        # reach: gripper -> peg tail distance
        gripper_pos = base.agent.tcp.pose.p
        offset_pose = base.peg.pose * sapien.Pose([-0.06, 0, 0])
        gripper_to_peg_dist = torch.linalg.norm(
            gripper_pos - offset_pose.p, axis=1
        )
        reach_r = 1 - torch.tanh(4.0 * gripper_to_peg_dist)

        # grasp
        is_grasped = base.agent.is_grasping(base.peg, max_angle=20)
        grasp_r = is_grasped.float()

        # pre-insertion: align peg with hole (yz coordinates)
        peg_head_wrt_goal = base.goal_pose.inv() * base.peg_head_pose
        peg_head_wrt_goal_yz_dist = torch.linalg.norm(
            peg_head_wrt_goal.p[:, 1:], axis=1
        )
        peg_wrt_goal = base.goal_pose.inv() * base.peg.pose
        peg_wrt_goal_yz_dist = torch.linalg.norm(peg_wrt_goal.p[:, 1:], axis=1)

        pre_insertion_r = 1 - torch.tanh(
            0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist)
            + 4.5 * torch.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)
        )

        # insertion: insert peg into hole
        pre_inserted = (peg_head_wrt_goal_yz_dist < 0.01) & (
            peg_wrt_goal_yz_dist < 0.01
        )
        peg_head_wrt_hole = base.box_hole_pose.inv() * base.peg_head_pose
        insertion_r = (
            1 - torch.tanh(5.0 * torch.linalg.norm(peg_head_wrt_hole.p, axis=1))
        ) * (is_grasped & pre_inserted).float()

        scale = self._norm_scale()
        reward = scale * (
            w["w_reach"] * reach_r
            + w["w_grasp"] * grasp_r
            + w["w_pre_insertion"] * pre_insertion_r * grasp_r
            + w["w_insertion"] * insertion_r
        )
        reward[info["success"]] = w["w_success"]

        self._last_breakdown = {
            "reach": (scale * w["w_reach"] * reach_r).mean().item(),
            "grasp": (scale * w["w_grasp"] * grasp_r).mean().item(),
            "pre_insertion": (scale * w["w_pre_insertion"] * pre_insertion_r * grasp_r).mean().item(),
            "insertion": (scale * w["w_insertion"] * insertion_r).mean().item(),
            "norm_scale": scale,
        }
        return reward

    # --- PushT ---
    def _compute_push_t(self, info: dict) -> torch.Tensor:
        base = self.env.unwrapped
        w = self.weights

        # rotation alignment: cos similarity between tee and goal rotation
        tee_z_eulers = base.quat_to_z_euler(base.tee.pose.q)
        rot_rew = (tee_z_eulers - base.goal_z_rot).cos()
        rotation_r = ((rot_rew + 1) / 2) ** 2

        # position alignment: tee xy -> goal xy distance
        tee_to_goal_pose = base.tee.pose.p[:, 0:2] - base.goal_tee.pose.p[:, 0:2]
        tee_to_goal_dist = torch.linalg.norm(tee_to_goal_pose, axis=1)
        position_r = (1 - torch.tanh(5 * tee_to_goal_dist)) ** 2

        # tcp guidance: encourage end-effector to stay near tee
        tcp_to_tee = base.tee.pose.p - base.agent.tcp.pose.p
        tcp_to_tee_dist = torch.linalg.norm(tcp_to_tee, axis=1)
        tcp_guide_r = (1 - torch.tanh(5 * tcp_to_tee_dist)).sqrt()

        scale = self._norm_scale()
        reward = scale * (
            w["w_rotation"] * rotation_r
            + w["w_position"] * position_r
            + w["w_tcp_guide"] * tcp_guide_r
        )
        reward[info["success"]] = w["w_success"]

        self._last_breakdown = {
            "rotation": (scale * w["w_rotation"] * rotation_r).mean().item(),
            "position": (scale * w["w_position"] * position_r).mean().item(),
            "tcp_guide": (scale * w["w_tcp_guide"] * tcp_guide_r).mean().item(),
            "norm_scale": scale,
        }
        return reward
