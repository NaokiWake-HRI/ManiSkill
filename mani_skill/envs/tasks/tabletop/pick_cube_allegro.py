"""PickCube task with Panda arm + Allegro dexterous hand."""
from typing import Any, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import PandaAllegro, PandaAllegroTouch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


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
    cube_half_size = 0.02
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
        # Stage 1: Reach the cube
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)

        # Stage 2: Contact (smooth, opposing-grasp aware)
        contact_r = info["contact_r"]

        # Stage 3: Lift gated by contact (RL_project key insight)
        # No lift reward without proper finger contact on the cube
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

        reward = reaching_reward + contact_r + lift_reward + place_reward + static_reward
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="panda_allegro_touch", **kwargs)
