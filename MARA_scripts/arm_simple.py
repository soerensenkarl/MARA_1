# arm_simple.py
import time
from bosdyn.api import arm_command_pb2, geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    GRAV_ALIGNED_BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    get_a_tform_b,
)
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
)
from bosdyn.client.robot_state import RobotStateClient


def run(robot, seconds: float = 2.0):
    """
    Simple two-pose arm move with open/close gripper, then STOW.
    PRECONDITIONS (handled in sequence.py):
      - robot is authenticated & time-synced
      - lease keepalive is active
      - robot is powered on AND standing
    """
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    # --- Pose 1: forward & a bit up, gripper open ---
    x, y, z = 0.75, 0.0, 0.25
    flat_pose = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(x=x, y=y, z=z),
        rotation=geometry_pb2.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    )

    robot_state = robot_state_client.get_robot_state()
    odom_T_flat = get_a_tform_b(
        robot_state.kinematic_state.transforms_snapshot,
        ODOM_FRAME_NAME,
        GRAV_ALIGNED_BODY_FRAME_NAME,
    )
    odom_T_hand = odom_T_flat * math_helpers.SE3Pose.from_proto(flat_pose)

    cmd1 = RobotCommandBuilder.build_synchro_command(
        RobotCommandBuilder.claw_gripper_open_fraction_command(1.0),
        RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z,
            odom_T_hand.rot.w, odom_T_hand.rot.x, odom_T_hand.rot.y, odom_T_hand.rot.z,
            ODOM_FRAME_NAME, seconds,
        ),
    )
    cmd_id = command_client.robot_command(cmd1)
    _block_until_cartesian_done(robot, command_client, cmd_id)

    # --- Pose 2: lower & rotate a bit, gripper close ---
    flat_pose.position.z = 0.0
    flat_pose.rotation.w, flat_pose.rotation.x, flat_pose.rotation.y, flat_pose.rotation.z = 0.707, 0.707, 0.0, 0.0
    odom_T_hand2 = odom_T_flat * math_helpers.SE3Pose.from_proto(flat_pose)

    cmd2 = RobotCommandBuilder.build_synchro_command(
        RobotCommandBuilder.claw_gripper_open_fraction_command(0.0),
        RobotCommandBuilder.arm_pose_command(
            odom_T_hand2.x, odom_T_hand2.y, odom_T_hand2.z,
            odom_T_hand2.rot.w, odom_T_hand2.rot.x, odom_T_hand2.rot.y, odom_T_hand2.rot.z,
            ODOM_FRAME_NAME, seconds,
        ),
    )
    cmd_id = command_client.robot_command(cmd2)
    block_until_arm_arrives(command_client, cmd_id)

    # --- STOW the arm (brief wait to retract) ---
    stow_cmd = RobotCommandBuilder.arm_stow_command()
    command_client.robot_command(stow_cmd)
    time.sleep(1.0)

    robot.logger.info("arm_simple.run() complete (stowed).")


def _block_until_cartesian_done(robot, command_client, cmd_id):
    while True:
        fb = command_client.robot_command_feedback(cmd_id)
        cart = fb.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback
        if cart.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
            break
        time.sleep(0.1)
