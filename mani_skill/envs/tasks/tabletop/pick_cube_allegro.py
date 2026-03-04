"""PickCube task with Panda arm + Allegro dexterous hand."""
from typing import Any, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import PandaAllegro, PandaAllegroTouch, PandaAllegroTouchLab
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


# FSR tip indices within fsr_links (palm[0-3], thumb[4-6], index[7-9], middle[10-12], ring[13-15])
_FSR_TIP_THUMB = 6    # allegro_link_15.0_tip_fsr
_FSR_TIP_INDEX = 9    # allegro_link_3.0_tip_fsr
_FSR_TIP_MIDDLE = 12  # allegro_link_7.0_tip_fsr
_FSR_TIP_RING = 15    # allegro_link_11.0_tip_fsr
_FSR_TIP_FINGER_GROUP = [_FSR_TIP_INDEX, _FSR_TIP_MIDDLE, _FSR_TIP_RING]


@register_env("PickCubePandaAllegro-v1", max_episode_steps=100)
class PickCubePandaAllegroEnv(BaseEnv):
    """PickCube with Panda arm + Allegro dexterous hand.

    The task is the same as PickCube-v1 but uses the Allegro hand instead of
    a parallel jaw gripper. The episode is longer (100 steps) because
    dexterous grasping is harder and needs more time.
    """

    SUPPORTED_ROBOTS = ["panda_allegro", "panda_allegro_touch"]
    agent: Union[PandaAllegro, PandaAllegroTouch]
    goal_thresh = 0.025
    cube_half_size = 0.03
    cube_spawn_half_size = 0.1
    cube_spawn_center = (0, 0)
    max_goal_height = 0.3

    def __init__(self, *args, robot_uids="panda_allegro", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.7, 0.6], target=[0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # Initialize robot via table scene builder
            self.table_scene.initialize(env_idx)

            # Randomize cube position on the table
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # Randomize goal position
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
                tip_poses=self.agent.tip_poses.reshape(-1, 4 * 7),
                palm_pose=self.agent.palm_pose,
            )
        return obs

    # ------------------------------------------------------------------
    # Contact reward (RL_project inspired: opposing grasp detection)
    # ------------------------------------------------------------------
    def _compute_contact_reward(self):
        """Compute smooth contact reward based on opposing grasp groups.

        Groups (matching coupled finger synergy):
          - Thumb: thumb tip FSR (or tip link force)
          - Finger group: Index/Middle/Ring tip FSR (any one = contact)

        Returns:
            contact_r: (n_envs,) float tensor in [0, 1].
                0.0 = no contact, 0.5 = one group only, 1.0 = opposing grasp
        """
        if hasattr(self.agent, "fsr_links"):
            # FSR path: pair-wise contact between FSR tip pads and the cube
            # FSR links have pad-side-only collision, so this is direction-aware
            fsr_tip_indices = [_FSR_TIP_THUMB] + _FSR_TIP_FINGER_GROUP
            tip_forces = []
            for idx in fsr_tip_indices:
                forces = self.scene.get_pairwise_contact_forces(
                    self.agent.fsr_links[idx], self.cube
                )
                tip_forces.append(torch.linalg.norm(forces, axis=1))
            tip_mags = torch.stack(tip_forces, dim=-1)  # (n_envs, 4)
            threshold = 0.01
            has_thumb = (tip_mags[:, 0] > threshold).float()
            has_finger_group = (
                tip_mags[:, 1:].max(dim=-1).values > threshold
            ).float()
        else:
            # Fallback: use pairwise contact forces on tip links
            # tip_links order: [thumb_tip, index_tip, middle_tip, ring_tip]
            tip_forces = []
            for tip_link in self.agent.tip_links:
                forces = self.scene.get_pairwise_contact_forces(
                    tip_link, self.cube
                )
                tip_forces.append(torch.linalg.norm(forces, axis=1))
            tip_mags = torch.stack(tip_forces, dim=-1)  # (n_envs, 4)
            has_thumb = (tip_mags[:, 0] >= 0.5).float()
            has_finger_group = (tip_mags[:, 1:].max(dim=-1).values >= 0.5).float()

        return (has_thumb + has_finger_group) / 2.0

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        contact_r = self._compute_contact_reward()
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
            "contact_r": contact_r,
        }

    # ------------------------------------------------------------------
    # Dense reward (6-stage, RL_project inspired)
    # ------------------------------------------------------------------
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Stage 1: Reach - use fingertip distance (not TCP/palm)
        tip_positions = torch.stack(
            [link.pose.p for link in self.agent.tip_links], dim=1
        )  # (n_envs, 4, 3)
        tip_to_obj = tip_positions - self.cube.pose.p.unsqueeze(1)
        tip_dists = torch.linalg.norm(tip_to_obj, dim=-1)  # (n_envs, 4)
        min_tip_dist = tip_dists.min(dim=-1).values
        reaching_reward = 1 - torch.tanh(5 * min_tip_dist)

        # Stage 2: Contact (smooth, opposing-grasp aware)
        contact_r = info["contact_r"]

        # Stage 3: Lift gated by contact
        cube_z = self.cube.pose.p[:, 2]
        lift_height = cube_z - self.cube_half_size
        lift_r = torch.clamp(lift_height / 0.05, 0.0, 1.0)
        lift_reward = lift_r * contact_r

        # Stage 4: Place at goal (gated by grasp)
        is_grasped = info["is_grasped"]
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = (1 - torch.tanh(5 * obj_to_goal_dist)) * is_grasped

        # Stage 5: Be static when placed
        qvel = self.agent.robot.get_qvel()[..., :7]  # arm joints only
        static_reward = (
            1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        ) * info["is_obj_placed"]

        # Penalty: discourage palm contact with cube
        palm_forces = self.scene.get_pairwise_contact_forces(
            self.agent.palm_link, self.cube
        )
        palm_penalty = -0.5 * (torch.linalg.norm(palm_forces, dim=-1) > 0.1).float()

        reward = reaching_reward + contact_r + lift_reward + place_reward + static_reward + palm_penalty
        reward[info["success"]] = 6
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6


@register_env("PickCubePandaAllegroTouch-v1", max_episode_steps=100)
class PickCubePandaAllegroTouchEnv(PickCubePandaAllegroEnv):
    """PickCube with Panda arm + Allegro hand + FSR tactile sensors."""

    SUPPORTED_ROBOTS = ["panda_allegro_touch"]
    agent: PandaAllegroTouch

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 16,
                max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 4,
                found_lost_pairs_capacity=2**26,
            )
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="panda_allegro_touch", **kwargs)


@register_env("PickCubePandaAllegro-v2", max_episode_steps=600)
class PickCubePandaAllegroV2Env(BaseEnv):
    """PickCube with Panda + Allegro + TouchLab tactile sensors (4 fingertip patches).

    SimToolReal-style control and reward:
      - Arm: joint target delta pos (7D), Hand: absolute joint pos (16D) = 23D
      - sim_freq=120, control_freq=60 (matching SimToolReal)
      - Phase 1 (pre-lift): fingertip delta reward + lifting reward
      - Phase 2 (post-lift): place delta reward + success bonus
      - Action penalties: arm (0.03) >> hand (0.003)
    """

    SUPPORTED_ROBOTS = ["panda_allegro_touchlab"]
    agent: PandaAllegroTouchLab
    goal_thresh = 0.025
    cube_half_size = [0.02, 0.04, 0.02]
    cube_spawn_half_size = 0.1
    cube_spawn_center = (0, 0)
    max_goal_height = 0.3

    def __init__(self, *args, robot_uids="panda_allegro_touchlab",
                 robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if "control_mode" not in kwargs:
            kwargs["control_mode"] = "pd_joint_target_delta_pos_arm_abs_hand"
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=120,
            control_freq=60,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=max(2**21, self.num_envs * 1024),
                max_rigid_patch_count=max(2**21, self.num_envs * 128),
                found_lost_pairs_capacity=2**26,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.7, 0.6], target=[0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_box(
            self.scene,
            half_sizes=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size[2]]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    # --- SimToolReal-style reward parameters ---
    fingertip_delta_rew_scale = 50.0
    lifting_rew_scale = 20.0
    lifting_bonus = 300.0
    lifting_bonus_threshold = 0.15
    place_rew_scale = 200.0
    reach_goal_bonus = 1000.0
    success_steps = 10
    arm_actions_penalty_scale = 0.03
    hand_actions_penalty_scale = 0.003

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomize cube position (no rotation)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self.cube_half_size[2]
            self.cube.set_pose(Pose.create_from_pq(xyz))

            # Randomize goal position
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # Initialize SimToolReal-style tracking states
            n_envs = self.num_envs
            n_tips = 4  # thumb, index, middle, ring
            if not hasattr(self, "_closest_fingertip_dist"):
                self._closest_fingertip_dist = torch.full((n_envs, n_tips), float("inf"), device=self.device)
                self._lifted_object = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
                self._near_goal_steps = torch.zeros(n_envs, dtype=torch.long, device=self.device)
                self._object_init_z = torch.zeros(n_envs, device=self.device)
                self._closest_obj_to_goal_dist = torch.full((n_envs,), float("inf"), device=self.device)
            # Reset only the envs being initialized
            # Use -1 as sentinel; first reward step will initialize to actual distance
            self._closest_fingertip_dist[env_idx] = -1.0
            self._lifted_object[env_idx] = False
            self._near_goal_steps[env_idx] = 0
            self._object_init_z[env_idx] = xyz[:, 2]
            self._closest_obj_to_goal_dist[env_idx] = -1.0

    def _get_obs_extra(self, info: dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # === Phase 1: Fingertip delta reward (before lifting) ===
        # Per-finger distance to object center
        tip_positions = torch.stack(
            [link.pose.p for link in self.agent.tip_links], dim=1
        )  # (n_envs, 4, 3)
        tip_to_obj = tip_positions - self.cube.pose.p.unsqueeze(1)
        curr_fingertip_dist = torch.linalg.norm(tip_to_obj, dim=-1)  # (n_envs, 4)

        # On first step after reset, initialize closest to current (no free reward)
        first_step_mask = self._closest_fingertip_dist[:, 0] < 0
        self._closest_fingertip_dist[first_step_mask] = curr_fingertip_dist[first_step_mask]

        # Delta = improvement from historical closest distance per finger
        fingertip_deltas = self._closest_fingertip_dist - curr_fingertip_dist
        self._closest_fingertip_dist = torch.minimum(
            self._closest_fingertip_dist, curr_fingertip_dist
        )
        # Only reward positive improvement, clip to prevent explosion
        fingertip_deltas = torch.clamp(fingertip_deltas, 0, 10)
        fingertip_delta_rew = fingertip_deltas.sum(dim=-1)  # sum across 4 fingers
        # Disable after lifting
        fingertip_delta_rew = fingertip_delta_rew * (~self._lifted_object)
        fingertip_delta_rew = fingertip_delta_rew * self.fingertip_delta_rew_scale

        # === Phase 1: Lifting reward ===
        cube_z = self.cube.pose.p[:, 2]
        # +0.05 offset (from SimToolReal): gives small positive reward at rest,
        # so the agent gets a gradient signal toward "up" from the start.
        # Effective lift threshold = lifting_bonus_threshold - 0.05 = 0.10m actual rise.
        z_lift = 0.05 + cube_z - self._object_init_z
        lifting_rew = torch.clamp(z_lift, 0, 0.5)

        # Check if object crossed lifting threshold
        lifted_object = (z_lift > self.lifting_bonus_threshold) | self._lifted_object
        just_lifted = lifted_object & (~self._lifted_object)
        lift_bonus_rew = self.lifting_bonus * just_lifted.float()

        # Stop lifting reward once lifted
        lifting_rew = lifting_rew * (~lifted_object)
        lifting_rew = lifting_rew * self.lifting_rew_scale

        # Update lifted state
        self._lifted_object = lifted_object

        # === Phase 2: Place reward (after lifting) ===
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        # Initialize closest distance on the step when object is first lifted
        just_entered_phase2 = lifted_object & (self._closest_obj_to_goal_dist < 0)
        self._closest_obj_to_goal_dist[just_entered_phase2] = obj_to_goal_dist[just_entered_phase2]

        # Delta-based: reward for getting closer to goal
        place_deltas = self._closest_obj_to_goal_dist - obj_to_goal_dist
        # Only update closest distance AFTER lift (preserve Phase 2 improvement room)
        lifted_mask = lifted_object
        self._closest_obj_to_goal_dist[lifted_mask] = torch.minimum(
            self._closest_obj_to_goal_dist[lifted_mask], obj_to_goal_dist[lifted_mask]
        )
        place_deltas = torch.clamp(place_deltas, 0, 100)
        place_rew = place_deltas * lifted_object.float()
        place_rew = place_rew * self.place_rew_scale

        # === Success bonus (distributed over success_steps) ===
        near_goal = obj_to_goal_dist <= self.goal_thresh
        is_robot_static = self.agent.is_static(0.2)
        near_goal_and_static = near_goal & is_robot_static
        # Consecutive near-goal tracking
        self._near_goal_steps = (self._near_goal_steps + near_goal_and_static.long()) * near_goal_and_static.long()
        bonus_rew = near_goal_and_static.float() * (self.reach_goal_bonus / self.success_steps)

        # === Action penalties ===
        qvel = self.agent.robot.get_qvel()
        arm_penalty = -torch.sum(torch.abs(qvel[..., :7]), dim=-1) * self.arm_actions_penalty_scale
        hand_penalty = -torch.sum(torch.abs(qvel[..., 7:23]), dim=-1) * self.hand_actions_penalty_scale

        # === Total reward ===
        reward = (
            fingertip_delta_rew
            + lifting_rew
            + lift_bonus_rew
            + place_rew
            + bonus_rew
            + arm_penalty
            + hand_penalty
        )
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        # SimToolReal-style delta rewards don't have a fixed max, so no normalization.
        # Return raw dense reward directly.
        return self.compute_dense_reward(obs=obs, action=action, info=info)
