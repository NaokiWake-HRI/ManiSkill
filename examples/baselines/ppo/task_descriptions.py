"""Task descriptions and state access documentation for LLM/VLM outer loops.

Extracted from ppo_outer_loop.py / ppo_outer_loop_full.py to avoid duplication.
"""


def get_llm_task_descs(env_id: str) -> dict:
    """Return per-task LLM descriptions.

    ``env_id`` is interpolated into the PickCube entry to preserve the
    original f-string behaviour.
    """
    return {
        "PushCube": (
            "The robot arm must push a cube to the goal position (PushCube).\n\n"
            "日本語補足: ロボットアームがキューブを目標位置まで押すタスク。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "PickCube": (
            f"The robot must pick up a cube and place it at the goal ({env_id}).\n\n"
            "日本語補足: ロボットがキューブを掴んで目標位置に運ぶタスク。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "OpenCabinetDoor": (
            "The robot must open a cabinet door (OpenCabinetDoor).\n\n"
            "日本語補足: キャビネットのドアを開けるタスク。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "OpenCabinetDrawer": (
            "The robot must open a cabinet drawer (OpenCabinetDrawer).\n\n"
            "日本語補足: キャビネットの引き出しを開けるタスク。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "PegInsertionSide": (
            "The robot must grasp a peg and insert it into a hole from the side "
            "(PegInsertionSide).\n\n"
            "日本語補足: ペグを掴んで横方向から穴に挿入するタスク。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "PushT": (
            "The robot must push a T-shaped block to match the goal position and "
            "rotation (PushT).\n\n"
            "日本語補足: T字ブロックを目標位置・回転に合わせるタスク。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "AnymalC": (
            "A quadruped robot (AnymalC) must walk to a goal position while "
            "maintaining balance.\n\n"
            "日本語補足: 四脚ロボットが目標位置まで歩行するタスク。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "UnitreeG1PlaceAppleInBowl": (
            "A humanoid robot (UnitreeG1) must pick up an apple and place it into "
            "a bowl (UnitreeG1PlaceAppleInBowl).\n\n"
            "日本語補足: ヒューマノイドロボットがリンゴを掴んでボウルに入れるタスク。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "PickCubePandaAllegro": (
            "A Panda arm with Allegro dexterous hand must pick up a box and place it "
            "at the goal position (PickCubePandaAllegro-v2).\n\n"
            "日本語補足: Pandaアーム+Allegro多指ハンドでボックスを掴んで目標位置に運ぶタスク。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "RotateValve": (
            "A three-fingered DClaw hand must rotate a valve by 90 degrees "
            "(RotateValveLevel0).\n\n"
            "日本語補足: 3本指のDClawハンドでバルブを90度回転させるタスク。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "UnitreeG1TransportBox": (
            "A humanoid robot (UnitreeG1 upper body) must pick up a box from one table "
            "and place it on a second table (UnitreeG1TransportBox).\n\n"
            "日本語補足: ヒューマノイドが箱をテーブル1からテーブル2に運ぶタスク。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
    }


STATE_ACCESS_DOCS = {
    "PushCube": """
Available state attributes (base = env.unwrapped):
- base.obj.pose.p: Cube position, torch.Tensor (batch_size, 3)
- base.agent.tcp.pose.p: End effector position, torch.Tensor (batch_size, 3)
- base.goal_region.pose.p: Goal position, torch.Tensor (batch_size, 3)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool

Success condition (from environment):
- Cube XY distance to goal < 0.1m (goal_radius=0.1) AND cube Z < 0.025m (on table).
- Key constants: cube_half_size=0.02m, goal_radius=0.1m.

Reward design guidelines:
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
- ALWAYS use full 3D Euclidean distances (not just XY) for reach/approach components.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
    "PickCube": """
Available state attributes (base = env.unwrapped):
- base.cube.pose.p: Object position, torch.Tensor (batch_size, 3)
- base.agent.tcp_pose.p: End effector position, torch.Tensor (batch_size, 3)
- base.goal_site.pose.p: Goal position, torch.Tensor (batch_size, 3)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool
- info["is_grasped"]: torch.Tensor (batch_size,) bool
- info["is_obj_placed"]: torch.Tensor (batch_size,) bool

Success condition (from environment):
- Cube within 0.025m (3D Euclidean) of goal AND robot is static (joint velocities <= 0.2 rad/s).
- Key constants: cube_half_size=0.02m, goal_thresh=0.025m.

Reward design guidelines:
- Total reward MUST be in [0, 5] range. On success, override reward to exactly 5.
- ALWAYS use full 3D Euclidean distances for reach/approach components.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
    "OpenCabinetDoor": """
Available state attributes (base = env.unwrapped):
- base.agent.tcp.pose.p: End effector position, torch.Tensor (batch_size, 3)
- base.handle_link.pose.p: Door handle link position, torch.Tensor (batch_size, 3)
- base.handle_link.joint.qpos: Door joint position (angle), torch.Tensor (batch_size, 1)
- base.target_qpos: Target joint position (75% of max range), torch.Tensor (batch_size, 1)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool
- info["handle_link_pos"]: torch.Tensor (batch_size, 3) handle center position (computed in evaluate())
- info["open_enough"]: torch.Tensor (batch_size,) bool (door opened beyond threshold)

Success condition (from environment):
- Door joint opened >= 75% of max range (min_open_frac=0.75) AND door is static (angular velocity <= 1 rad/s, linear velocity <= 0.1 m/s).
- Use base.target_qpos to compute opening progress: amount_to_open_left = (target_qpos - joint_qpos) / target_qpos.

Reward design guidelines:
- Total reward MUST be in [0, 5] range. On success, override reward to exactly 5.
- ALWAYS use full 3D Euclidean distances for reaching the handle.
- Reward door opening progress as a continuous value using (1 - amount_to_open_left).
- Note: The handle is attached to the door. As the door opens, the handle position changes accordingly.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
    "OpenCabinetDrawer": """
Available state attributes (base = env.unwrapped):
- base.agent.tcp.pose.p: End effector position, torch.Tensor (batch_size, 3)
- base.handle_link.pose.p: Drawer handle link position, torch.Tensor (batch_size, 3)
- base.handle_link.joint.qpos: Drawer joint position (distance), torch.Tensor (batch_size, 1)
- base.target_qpos: Target joint position (75% of max range), torch.Tensor (batch_size, 1)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool
- info["handle_link_pos"]: torch.Tensor (batch_size, 3) handle center position (computed in evaluate())
- info["open_enough"]: torch.Tensor (batch_size,) bool (drawer opened beyond threshold)

Success condition (from environment):
- Drawer joint opened >= 75% of max range (min_open_frac=0.75) AND drawer is static (angular velocity <= 1 rad/s, linear velocity <= 0.1 m/s).
- Use base.target_qpos to compute opening progress: amount_to_open_left = (target_qpos - joint_qpos) / target_qpos.

Reward design guidelines:
- Total reward MUST be in [0, 5] range. On success, override reward to exactly 5.
- ALWAYS use full 3D Euclidean distances for reaching the handle.
- Reward drawer opening progress as a continuous value using (1 - amount_to_open_left).
- Note: The handle is attached to the drawer. As the drawer opens, the handle position changes accordingly.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
    "PegInsertionSide": """
Available state attributes (base = env.unwrapped):
- base.peg.pose: Peg Pose object (has .p for position (batch_size, 3) and .inv() for inverse transform)
- base.agent.tcp.pose.p: End effector position, torch.Tensor (batch_size, 3)
- base.peg_head_pose: Peg head (tip) Pose object, property (batch_size,). Computed from peg pose + offset.
- base.box_hole_pose: Hole Pose object (has .p for position and .inv() for inverse transform)
- base.goal_pose: Goal Pose object — the target pose for the peg center to achieve insertion. Property (batch_size,).
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)
- base.agent.is_grasping(base.peg, max_angle=20): Grasp check, returns torch.Tensor (batch_size,) bool

Pose objects support coordinate transforms:
- pose_a.inv() * pose_b: transform pose_b into pose_a's local frame, returns a new Pose
- result.p: position in the local frame, torch.Tensor (batch_size, 3)
- result.p[:, 0]: X component, result.p[:, 1:]: YZ components

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool (peg fully inserted)
- info["peg_head_pos_at_hole"]: torch.Tensor (batch_size, 3) peg head position in hole's local frame

Success condition (from environment):
- Peg head inserted >= 15mm into hole (X-axis in hole frame) AND peg head Y,Z within hole radius.
- Key constants: peg radius 0.015-0.025m (randomized), hole clearance 0.003m, peg half-length 0.085-0.125m.

Reward design guidelines:
- Total reward MUST be in [0, 10] range. On success, override reward to exactly 10.
- ALWAYS use full 3D Euclidean distances for reach and alignment components.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
    "PushT": """
Available state attributes (base = env.unwrapped):
- base.tee.pose.p: T-block position, torch.Tensor (batch_size, 3)
- base.tee.pose.q: T-block rotation (quaternion), torch.Tensor (batch_size, 4)
- base.agent.tcp.pose.p: End effector position, torch.Tensor (batch_size, 3)
- base.goal_tee.pose.p: Goal T-block position, torch.Tensor (batch_size, 3)
- base.goal_z_rot: Goal z-axis rotation (Euler angle), torch.Tensor (batch_size,)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool

Success condition (from environment):
- T-block overlaps >= 90% of goal T region area (2D intersection). Requires BOTH position AND rotation alignment.
- intersection_thresh = 0.90.

Reward design guidelines:
- Total reward MUST be in [0, 3] range. On success, override reward to exactly 3.
- ALWAYS use full 3D Euclidean distances for reach/approach components.
- Reward both position proximity and rotation alignment toward goal.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
    "AnymalC": """
Available state attributes (base = env.unwrapped):
- base.agent.robot.pose.p: Robot position, torch.Tensor (batch_size, 3)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)
- base.goal.pose.p: Goal sphere position, torch.Tensor (batch_size, 3)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool (reached goal without falling)
- info["fail"]: torch.Tensor (batch_size,) bool (robot fell)
- info["robot_to_goal_dist"]: torch.Tensor (batch_size,) float (xy distance to goal)
- info["reached_goal"]: torch.Tensor (batch_size,) bool (dist < 0.35m)
- info["is_fallen"]: torch.Tensor (batch_size,) bool

Success condition (from environment):
- Robot XY distance to goal < 0.35m AND robot has NOT fallen.
- Goal is typically 2.0-3.0m away from the start position.

Reward design guidelines:
- Total reward MUST be in [0, 3] range. No explicit success bonus. On failure (fall), override reward to 0.
- Penalize falling (info["is_fallen"]).
- Reward progress toward goal (reducing XY distance).

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
    "UnitreeG1PlaceAppleInBowl": """
Available state attributes (base = env.unwrapped):
- base.apple.pose.p: Apple position, torch.Tensor (batch_size, 3)
- base.agent.right_tcp.pose.p: Right hand end effector position, torch.Tensor (batch_size, 3)
- base.bowl.pose.p: Bowl position, torch.Tensor (batch_size, 3)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)
- base.agent.right_hand_dist_to_open_grasp(): Distance to open grasp, torch.Tensor (batch_size,)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool (apple in bowl AND hand outside)
- info["is_grasped"]: torch.Tensor (batch_size,) bool
- info["hand_outside_bowl"]: torch.Tensor (batch_size,) bool (hand z > bowl z + 0.125m)

Success condition (from environment):
- Apple within 0.05m (3D Euclidean) of bowl position AND right hand Z > bowl Z + 0.125m (hand retracted above bowl).

Reward design guidelines:
- Total reward MUST be in [0, 10] range. On success, override reward to exactly 10.
- ALWAYS use full 3D Euclidean distances for reach/approach components.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
    "PickCubePandaAllegro": """
Available state attributes (base = env.unwrapped):
- base.cube.pose.p: Box position, torch.Tensor (batch_size, 3)
- base.agent.tcp_pose.p: End effector (TCP) position, torch.Tensor (batch_size, 3)
- base.goal_site.pose.p: Goal position, torch.Tensor (batch_size, 3)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, 23) (7 arm + 16 hand)
- base.agent.tip_links: List of 4 fingertip links [thumb, index, middle, ring]
- base.agent.tip_links[i].pose.p: Fingertip position, torch.Tensor (batch_size, 3)
- base.agent.palm_link.pose.p: Palm position, torch.Tensor (batch_size, 3)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool
- info["is_grasped"]: torch.Tensor (batch_size,) bool (>=2 fingertips in contact with box)
- info["is_obj_placed"]: torch.Tensor (batch_size,) bool (box within 0.025m of goal)
- info["is_robot_static"]: torch.Tensor (batch_size,) bool

Action space: 8D (6 arm EE delta + 2 hand scalars via CoupledAllegroActionWrapper).
  action[0:6]: arm end-effector delta pose (position + rotation)
  action[6]: finger group scalar [-1=open, 1=closed] (controls index/middle/ring together)
  action[7]: thumb scalar [-1=open, 1=closed]

Success condition (from environment):
- Box within 0.025m (3D Euclidean) of goal AND robot arm is static (joint velocities <= 0.2 rad/s).
- Key constants: cube_half_size=[0.02, 0.04, 0.02], goal_thresh=0.025m.

Reward design guidelines:
- Total reward MUST be in [0, 6] range. On success, override reward to exactly 6.
- ALWAYS use full 3D Euclidean distances for reach/approach components.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
    "RotateValve": """
Available state attributes (base = env.unwrapped):
- base.agent.tip_poses: 3 fingertip poses, torch.Tensor (batch_size, 3, 7) [x,y,z,qw,qx,qy,qz]
- base.valve.qpos: Valve joint position, torch.Tensor (batch_size, 1)
- base.valve.qvel: Valve joint velocity, torch.Tensor (batch_size, 1)
- base.valve_link.pose.p: Valve center position, torch.Tensor (batch_size, 3)
- base.rest_qpos: Initial valve joint position, torch.Tensor (batch_size, 1)
- base.rotate_direction: Target rotation direction (+1 or -1), torch.Tensor (batch_size,)
- base.capsule_lens: Valve tip radius from center, torch.Tensor (batch_size,)
- base.capsule_offset: Offset constant (0.01), float
- base.success_threshold: Required rotation angle (pi/2 for Level0), float

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool
- info["valve_rotation"]: torch.Tensor (batch_size,) float (cumulative rotation from start)

Success condition (from environment):
- valve_rotation * rotate_direction > success_threshold (pi/2 = 90 degrees for Level0).
- Max episode steps: 80 for Level0.

Reward design guidelines:
- Total reward MUST be in [0, 6] range. No explicit success bonus (reward is continuous).

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
    "UnitreeG1TransportBox": """
Available state attributes (base = env.unwrapped):
- base.agent.robot.qpos: Robot joint positions, torch.Tensor (batch_size, n_joints)
  - qpos[:, 0]: torso yaw (rotation around vertical axis)
  - qpos[:, 3]: right shoulder pitch
  - qpos[:, 4]: left shoulder pitch
- base.agent.right_tcp.pose.p: Right hand TCP position, torch.Tensor (batch_size, 3)
- base.agent.left_tcp.pose.p: Left hand TCP position, torch.Tensor (batch_size, 3)
- base.box.pose.p: Box position, torch.Tensor (batch_size, 3)
- base.box_right_grasp_point.p: Target grasp point for right hand, torch.Tensor (batch_size, 3)
- base.box_left_grasp_point.p: Target grasp point for left hand, torch.Tensor (batch_size, 3)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool (box on table 2 AND not grasped)
- info["facing_table_with_box"]: torch.Tensor (batch_size,) bool (torso facing table 1)
- info["box_grasped"]: torch.Tensor (batch_size,) bool (both hands contact box)
- info["box_at_correct_table_xy"]: torch.Tensor (batch_size,) bool (box XY over table 2)
- info["left_hand_hit_box"]: torch.Tensor (batch_size,) bool
- info["right_hand_hit_box"]: torch.Tensor (batch_size,) bool

Success condition (from environment):
- Box resting on table 2 (Z in [0.750, 0.751], XY within table bounds) AND box NOT grasped.
- Max episode steps: 100.

Reward design guidelines:
- Total reward MUST be in [0, 5] range. On success, override reward to exactly 5.
- The robot must pick up a box from one table and place it on another table.
- Torso yaw ~-1.4 rad faces table 1, ~+1.4 rad faces table 2.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
}
