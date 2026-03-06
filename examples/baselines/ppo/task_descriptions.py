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
            "The robot arm must push a cube to the goal position (PushCube).\n"
            "Reward components: w_reach (TCP approach to push position behind cube), "
            "w_push (cube movement toward goal, gated by reach), "
            "w_z_keep (keep cube on table surface), w_success (success bonus).\n\n"
            "日本語補足: ロボットアームがキューブを目標位置まで押すタスク。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "PickCube": (
            f"The robot must pick up a cube and place it at the goal ({env_id}).\n"
            "Reward components: w_reach (TCP approach to cube), w_grasp (grasp success), "
            "w_place (cube toward goal, gated by grasp), "
            "w_static (robot static with object placed), w_success (success bonus).\n\n"
            "日本語補足: ロボットがキューブを掴んで目標位置に運ぶタスク。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "OpenCabinetDoor": (
            "The robot must open a cabinet door (OpenCabinetDoor).\n"
            "Reward components: w_reach (TCP to handle), w_open (door opening progress), "
            "w_static (maintain open state), w_success (success bonus).\n\n"
            "日本語補足: キャビネットのドアを開けるタスク。回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "OpenCabinetDrawer": (
            "The robot must open a cabinet drawer (OpenCabinetDrawer).\n"
            "Reward components: w_reach (TCP to handle), w_open (drawer opening progress), "
            "w_static (maintain open state), w_success (success bonus).\n\n"
            "日本語補足: キャビネットの引き出しを開けるタスク。回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "PegInsertionSide": (
            "The robot must grasp a peg and insert it into a hole from the side (PegInsertionSide).\n"
            "This is a multi-stage task: reach the peg, grasp it, align with the hole, then insert.\n"
            "Reward components: w_reach (gripper approach to peg tail), "
            "w_grasp (binary grasp success), "
            "w_pre_insertion (peg-hole yz alignment, gated by grasp), "
            "w_insertion (peg head into hole, gated by grasp AND pre-insertion alignment), "
            "w_success (success bonus).\n\n"
            "日本語補足: ペグを掴んで横方向から穴に挿入するタスク。"
            "reach→grasp→alignment→insertionの段階的な報酬構造。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "PushT": (
            "The robot must push a T-shaped block to match the goal position and rotation (PushT).\n"
            "This is a 2D pushing task requiring both position and rotation alignment.\n"
            "Reward components: w_rotation (cos similarity of tee vs goal z-rotation, squared), "
            "w_position (tee-to-goal xy distance, tanh-shaped and squared), "
            "w_tcp_guide (encourage TCP to stay near the tee block), "
            "w_success (success bonus).\n\n"
            "日本語補足: T字ブロックを目標位置・回転に合わせるタスク。"
            "位置と回転の両方を合わせる必要がある。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "AnymalC": (
            "A quadruped robot (AnymalC) must walk to a goal position (AnymalC-Reach).\n"
            "The robot must maintain balance while locomoting toward the target.\n"
            "Reward components: w_reach (robot-to-goal distance, tanh-shaped), "
            "w_vel_z_penalty (penalize vertical velocity oscillation), "
            "w_ang_vel_penalty (penalize angular velocity in xy), "
            "w_contact_penalty (penalize undesired knee/body contacts with ground), "
            "w_qpos_penalty (penalize deviation from default standing pose).\n"
            "Note: reward has a base of +1.0 per step; fails (falls) give 0.\n\n"
            "日本語補足: 四脚ロボットが目標位置まで歩行するタスク。"
            "バランスを保ちながら移動する必要がある。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "UnitreeG1PlaceAppleInBowl": (
            "A humanoid robot (UnitreeG1) must pick up an apple and place it into a bowl "
            "(UnitreeG1PlaceAppleInBowl).\n"
            "Multi-stage: reach apple, grasp it, carry to above the bowl, then release.\n"
            "Reward components: w_reach (TCP-to-apple distance), "
            "w_grasp (binary grasp success), "
            "w_place (apple-to-bowl distance with +0.15m z-offset, gated by grasp), "
            "w_above_bowl (binary bonus when apple is within 0.025m of above-bowl target), "
            "w_release (encourage opening hand, gated by above_bowl), "
            "w_success (success bonus).\n"
            "Note: the bowl target has a +0.15m z-offset to encourage bringing the apple "
            "above the bowl before releasing.\n\n"
            "日本語補足: ヒューマノイドロボットがリンゴを掴んでボウルに入れるタスク。"
            "reach→grasp→carry→above_bowl→releaseの段階的な報酬構造。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "PickCubePandaAllegro": (
            "A Panda arm with Allegro dexterous hand must pick up a box and place it at "
            "the goal position (PickCubePandaAllegro-v2).\n"
            "The hand uses coupled control: 2D action (finger group + thumb open/close) "
            "mapped to 16D joint positions. Arm uses 6D EE delta pose.\n"
            "Reward components: w_reach (TCP approach to box), w_grasp (>=2 fingertips in contact), "
            "w_place (box toward goal, gated by grasp), "
            "w_static (arm static with object placed), w_success (success bonus).\n\n"
            "日本語補足: Pandaアーム+Allegro多指ハンドでボックスを掴んで目標位置に運ぶタスク。"
            "指は2自由度（親指・対抗指の開閉）で制御。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "RotateValve": (
            "A three-fingered DClaw hand must rotate a valve by 90 degrees "
            "(RotateValveLevel0).\n"
            "The valve has 3 capsule-shaped heads. The DClaw has 3 fingertips that must "
            "maintain contact with the valve tips while rotating.\n"
            "Reward components: w_contact (fingertip-to-valve distance error), "
            "w_velocity (directed angular velocity of valve rotation), "
            "w_progress (cumulative rotation toward goal).\n"
            "Note: No explicit success bonus — reward is continuous.\n\n"
            "日本語補足: 3本指のDClawハンドでバルブを90度回転させるタスク。"
            "指先をバルブ先端に接触させながら回転させる必要がある。"
            "回答の末尾に日本語での簡潔な要約も追加してください。"
        ),
        "UnitreeG1TransportBox": (
            "A humanoid robot (UnitreeG1 upper body) must grasp a box from one table "
            "and transport it to a second table (UnitreeG1TransportBox).\n"
            "This is a 4-stage task: (1) face the box, (2) grasp with both hands, "
            "(3) turn and carry to the other table, (4) release the box.\n"
            "Reward components: w_face_box (torso orientation toward table 1), "
            "w_grasp (arms down + TCPs near box grasp points, gated by facing), "
            "w_transport (turn toward table 2, gated by grasping), "
            "w_release (raise arms to release, gated by box at target), "
            "w_success (success bonus).\n"
            "Success requires box resting on table 2 AND not being grasped.\n\n"
            "日本語補足: ヒューマノイドが箱をテーブル1からテーブル2に運ぶタスク。"
            "向き変え→両手把持→運搬→リリースの4段階。"
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
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
- ALWAYS use full 3D Euclidean distances for reach/approach components.
- Use info["is_grasped"] to gate place rewards (only reward placing when grasped).

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
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool
- info["handle_link_pos"]: torch.Tensor (batch_size, 3) handle center position (computed in evaluate())
- info["open_enough"]: torch.Tensor (batch_size,) bool (door opened beyond threshold)

Success condition (from environment):
- Door joint opened >= 75% of max range (min_open_frac=0.75) AND door is static (angular velocity <= 1 rad/s, linear velocity <= 0.1 m/s).

Reward design guidelines:
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
- ALWAYS use full 3D Euclidean distances for reaching the handle.
- Reward door opening progress (joint position increase) after reaching the handle.

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
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool
- info["handle_link_pos"]: torch.Tensor (batch_size, 3) handle center position (computed in evaluate())
- info["open_enough"]: torch.Tensor (batch_size,) bool (drawer opened beyond threshold)

Success condition (from environment):
- Drawer joint opened >= 75% of max range (min_open_frac=0.75) AND drawer is static (angular velocity <= 1 rad/s, linear velocity <= 0.1 m/s).

Reward design guidelines:
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
- ALWAYS use full 3D Euclidean distances for reaching the handle.
- Reward drawer opening progress (joint position increase) after reaching the handle.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
    "PegInsertionSide": """
Available state attributes (base = env.unwrapped):
- base.peg.pose.p: Peg position, torch.Tensor (batch_size, 3)
- base.agent.tcp.pose.p: End effector position, torch.Tensor (batch_size, 3)
- base.box_hole_pose.p: Hole position, torch.Tensor (batch_size, 3)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool (peg fully inserted)
- info["peg_head_pos_at_hole"]: torch.Tensor (batch_size, 3) computed peg head position

Success condition (from environment):
- Peg head inserted >= 15mm into hole (X-axis in hole frame) AND peg head Y,Z within hole radius.
- Key constants: peg radius 0.015-0.025m (randomized), hole clearance 0.003m, peg half-length 0.085-0.125m.

Reward design guidelines:
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
- ALWAYS use full 3D Euclidean distances for reach and alignment components.
- Multi-stage: reach peg -> grasp -> align with hole -> insert.

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
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
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
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
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
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
- ALWAYS use full 3D Euclidean distances for reach/approach components.
- Use info["is_grasped"] to gate place rewards.
- After placing, reward hand retraction (moving hand above bowl).

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
- Total reward MUST be in [0, 5] range. On success, override reward to exactly 5.
- ALWAYS use full 3D Euclidean distances for reach/approach components.
- Use info["is_grasped"] to gate place rewards (only reward placing when grasped).
- Only use arm joint velocities (first 7) for static penalty, not hand joints.

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
- Fingertip contact: compute distance between 3 fingertip XY positions and valve center,
  compare to desired distance (capsule_lens - capsule_offset).
- Velocity: reward valve angular velocity in the correct direction.
- Progress: reward cumulative rotation toward goal.

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
- 4-stage task: face box -> grasp -> transport -> release.
- Each stage gates on the previous stage's success (use info dict booleans).
- Torso yaw ~-1.4 rad faces table 1, ~+1.4 rad faces table 2.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
}
