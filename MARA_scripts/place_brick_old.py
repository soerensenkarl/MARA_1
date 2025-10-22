# Copyright (c) 2023 Boston Dynamics, Inc.
# All rights reserved.
#
# SDK License: 20191101-BDSDK-SL
#
# place_brick.py — Simple chained arm movements relative to an AprilTag fiducial.
#
# Designed to be called from sequence.py as:
#     import place_brick
#     place_brick.run(robot)
#
# The script automatically finds fiducial tag_id=4 (change below if needed),
# and executes a predefined sequence of movements relative to that tag:
#   1) Move 10 cm above the tag.
#   2) Move 10 cm to the left.
#   3) Move 10 cm to the right.
#
# You can edit the "STEPS" list to define your own motion chain.

import time
from typing import List, Tuple, Optional

from bosdyn.api import arm_command_pb2, world_object_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient

SE3 = math_helpers.SE3Pose

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

TAG_ID = 4          # ID number of the fiducial (printed tag)
SECONDS_PER_STEP = 1.5  # time per move
CUMULATIVE = False      # if True, steps build on each other
SEND_IN_ODOM = False    # command frame (False = vision)

# Define a simple chain of relative offsets (meters)
STEPS: List[Tuple[float, float, float]] = [
    (0.00, 0.00, 0.50),  # move 10 cm up
    (0.00, 0.00, 0.50), # move 10 cm left
    (0.00, 0.00, 0.20),  # move 10 cm right
]

# ---------------------------------------------------------------------------


def _find_fiducial(robot, tag_id: Optional[int], wait_sec: float = 3.0):
    """Find an AprilTag by tag_id and return (frame_name, vision_T_fiducial)."""
    wo_client = robot.ensure_client(WorldObjectClient.default_service_name)
    t0 = time.time()
    chosen = None

    while True:
        resp = wo_client.list_world_objects(object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG])
        objs = resp.world_objects

        chosen = next((o for o in objs if o.apriltag_properties.tag_id == tag_id), None)
        if chosen or (time.time() - t0) > wait_sec:
            break
        time.sleep(0.1)

    if not chosen:
        raise RuntimeError(f"No AprilTag with ID {tag_id} detected by World Object service.")

    fid_frame = chosen.apriltag_properties.frame_name_fiducial
    vision_T_fid = get_a_tform_b(chosen.transforms_snapshot, VISION_FRAME_NAME, fid_frame)
    if vision_T_fid is None:
        raise RuntimeError(f"Could not compute transform to fiducial frame '{fid_frame}'.")
    return fid_frame, vision_T_fid


def _send_pose(command_client, pose: SE3, frame_name: str, seconds: float):
    """Send a single arm pose command and block until complete."""
    arm_cmd = RobotCommandBuilder.arm_pose_command(
        pose.x, pose.y, pose.z,
        pose.rot.w, pose.rot.x, pose.rot.y, pose.rot.z,
        frame_name, seconds
    )
    cmd = RobotCommandBuilder.build_synchro_command(arm_cmd)
    cmd_id = command_client.robot_command(cmd)
    block_until_arm_arrives(command_client, cmd_id)


def run(robot):
    """Execute the chained movements relative to the fiducial."""
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    goal_rot = math_helpers.Quat(w=1, x=0, y=0, z=0)
    acc = SE3(0, 0, 0, goal_rot)

    for i, (dx, dy, dz) in enumerate(STEPS, start=1):
        fid_name, vision_T_fid = _find_fiducial(robot, tag_id=TAG_ID)
        offset = SE3(
            acc.x + dx if CUMULATIVE else dx,
            acc.y + dy if CUMULATIVE else dy,
            acc.z + dz if CUMULATIVE else dz,
            goal_rot,
        )
        if CUMULATIVE:
            acc = offset

        if SEND_IN_ODOM:
            rs = robot_state_client.get_robot_state()
            tf = rs.kinematic_state.transforms_snapshot
            odom_T_vision = get_a_tform_b(tf, ODOM_FRAME_NAME, VISION_FRAME_NAME)
            if odom_T_vision is None:
                raise RuntimeError("Could not compute odom_T_vision.")
            target_pose = odom_T_vision * (vision_T_fid * offset)
            frame_used = ODOM_FRAME_NAME
        else:
            target_pose = vision_T_fid * offset
            frame_used = VISION_FRAME_NAME

        robot.logger.info(
            f"[{i}/{len(STEPS)}] Move to {fid_name} + "
            f"({offset.x:.3f}, {offset.y:.3f}, {offset.z:.3f}) m in {frame_used}"
        )
        _send_pose(command_client, target_pose, frame_used, SECONDS_PER_STEP)

    robot.logger.info("place_brick.run() chain complete.")


# Optional back-compat helper (not used but kept for clarity)
def _block_until_cartesian_done(robot, command_client, cmd_id):
    while True:
        fb = command_client.robot_command_feedback(cmd_id)
        cart = fb.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback
        if cart.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
            break
        time.sleep(0.1)
