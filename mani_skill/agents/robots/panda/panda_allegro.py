from copy import deepcopy

import numpy as np
import sapien
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import vectorize_pose


@register_agent()
class PandaAllegro(BaseAgent):
    uid = "panda_allegro"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_allegro.urdf"
    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link={
            "allegro_link_3.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "allegro_link_7.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "allegro_link_11.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "allegro_link_15.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    # Panda arm (7 joints)
                    # Matches RL_project config: [0, 15, 0, -120, 180, 135, 90] deg
                    0.0,                    # joint1:   0 deg
                    np.deg2rad(15),         # joint2:  15 deg
                    0.0,                    # joint3:   0 deg
                    np.deg2rad(-120),       # joint4: -120 deg
                    np.pi,                  # joint5:  180 deg
                    np.deg2rad(135),        # joint6:  135 deg
                    np.deg2rad(90),         # joint7:   90 deg
                    # Allegro hand (16 joints) - open hand pose
                    0.0, 0.0, 0.0, 0.0,    # index finger
                    0.0, 0.0, 0.0, 0.0,    # middle finger
                    0.0, 0.0, 0.0, 0.0,    # ring finger
                    0.83, 0.0, 0.0, 0.0,   # thumb (slight pre-rotation)
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    hand_joint_names = [
        "allegro_joint_0.0",
        "allegro_joint_1.0",
        "allegro_joint_2.0",
        "allegro_joint_3.0",
        "allegro_joint_4.0",
        "allegro_joint_5.0",
        "allegro_joint_6.0",
        "allegro_joint_7.0",
        "allegro_joint_8.0",
        "allegro_joint_9.0",
        "allegro_joint_10.0",
        "allegro_joint_11.0",
        "allegro_joint_12.0",
        "allegro_joint_13.0",
        "allegro_joint_14.0",
        "allegro_joint_15.0",
    ]

    ee_link_name = "allegro_tcp"

    # Tip link names (thumb, index, middle, ring)
    tip_link_names = [
        "allegro_link_15.0_tip",
        "allegro_link_3.0_tip",
        "allegro_link_7.0_tip",
        "allegro_link_11.0_tip",
    ]
    palm_link_name = "allegro_palm"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    hand_stiffness = 4e2
    hand_damping = 1e1
    hand_force_limit = 5e1

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-2.0,
            pos_upper=2.0,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,
            self.arm_force_limit,
        )

        # -------------------------------------------------------------------------- #
        # Hand (Allegro)
        # -------------------------------------------------------------------------- #
        hand_pd_joint_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            lower=None,
            upper=None,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            normalize_action=False,
        )
        hand_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            use_delta=True,
        )
        # Coupled fingers: index controls middle + ring; thumb independent.
        # This reduces hand action DOF from 16 -> 8 (index 4 + thumb 4).
        hand_mimic = {
            # Middle -> Index
            "allegro_joint_4.0": {"joint": "allegro_joint_0.0"},
            "allegro_joint_5.0": {"joint": "allegro_joint_1.0"},
            "allegro_joint_6.0": {"joint": "allegro_joint_2.0"},
            "allegro_joint_7.0": {"joint": "allegro_joint_3.0"},
            # Ring -> Index
            "allegro_joint_8.0": {"joint": "allegro_joint_0.0"},
            "allegro_joint_9.0": {"joint": "allegro_joint_1.0"},
            "allegro_joint_10.0": {"joint": "allegro_joint_2.0"},
            "allegro_joint_11.0": {"joint": "allegro_joint_3.0"},
            # Thumb self-mimic (keeps thumb independently controllable)
            "allegro_joint_12.0": {"joint": "allegro_joint_12.0"},
            "allegro_joint_13.0": {"joint": "allegro_joint_13.0"},
            "allegro_joint_14.0": {"joint": "allegro_joint_14.0"},
            "allegro_joint_15.0": {"joint": "allegro_joint_15.0"},
        }
        hand_pd_joint_delta_pos_coupled = PDJointPosMimicControllerConfig(
            self.hand_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            use_delta=True,
            mimic=hand_mimic,
        )
        hand_pd_joint_target_delta_pos = deepcopy(hand_pd_joint_delta_pos)
        hand_pd_joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, hand=hand_pd_joint_delta_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, hand=hand_pd_joint_pos),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos, hand=hand_pd_joint_pos
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, hand=hand_pd_joint_pos
            ),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, hand=hand_pd_joint_pos),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                hand=hand_pd_joint_target_delta_pos,
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, hand=hand_pd_joint_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, hand=hand_pd_joint_pos
            ),
            pd_joint_vel=dict(arm=arm_pd_joint_vel, hand=hand_pd_joint_pos),
            pd_joint_delta_pos_coupled=dict(
                arm=arm_pd_joint_delta_pos, hand=hand_pd_joint_delta_pos_coupled
            ),
        )

        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.tip_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.tip_link_names
        )
        self.palm_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.palm_link_name
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if any two fingertips are applying force on the object from
        opposing directions, indicating a grasp."""
        tip_forces = []
        tip_magnitudes = []
        for tip_link in self.tip_links:
            contact_forces = self.scene.get_pairwise_contact_forces(
                tip_link, object
            )
            force_mag = torch.linalg.norm(contact_forces, axis=1)
            tip_forces.append(contact_forces)
            tip_magnitudes.append(force_mag)

        # Check if at least 2 fingertips have contact
        has_contact = torch.stack(
            [mag >= min_force for mag in tip_magnitudes], dim=-1
        )
        num_contacts = has_contact.sum(dim=-1)
        return num_contacts >= 2

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :7]  # only check arm joints
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    @property
    def tip_poses(self):
        tip_poses = [
            vectorize_pose(link.pose, device=self.device)
            for link in self.tip_links
        ]
        return torch.stack(tip_poses, dim=-2)

    @property
    def palm_pose(self):
        return vectorize_pose(self.palm_link.pose, device=self.device)

    def get_proprioception(self):
        obs = super().get_proprioception()
        obs.update(
            {
                "palm_pose": self.palm_pose,
                "tip_poses": self.tip_poses.reshape(
                    -1, len(self.tip_links) * 7
                ),
            }
        )
        return obs
