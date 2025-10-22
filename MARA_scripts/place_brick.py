# Copyright (c) 2023 Boston Dynamics, Inc.
# All rights reserved.
#
# SDK License: 20191101-BDSDK-SL
#
# place_brick.py — Simple chained arm movements relative to an AprilTag fiducial,
# with optional body-follow and hip-height assist so Spot can move its base to help the arm.
#
# Designed to be called from sequence.py as:
#     import place_brick
#     place_brick.run(robot)

import time
from typing import List, Tuple, Optional
import math


from bosdyn.api import arm_command_pb2, world_object_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
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

TAG_ID = 4                 # AprilTag ID to follow
SECONDS_PER_STEP = 1.5     # time per move
CUMULATIVE = False         # if True, steps build on each other
SEND_IN_ODOM = False       # command frame (False = vision)

# Mobility assist / body follow
USE_BODY_FOLLOW = True         # if True, command base to follow the hand
ENABLE_HIP_HEIGHT_ASSIST = True  # if True, allow Spot to lower/adjust base height/yaw to help arm

# ---- User configuration ----
DEFAULT_PITCH_DEG = 90  # “looking down” if a step doesn't specify pitch

# Each step can be (dx, dy, dz) or (dx, dy, dz, pitch_deg)
STEPS = [
    (0.00,  0.00, 0.10),            # uses DEFAULT_PITCH_DEG
    (0.00,  0.20, 0.10),     # custom pitch
    (0.00, -0.20, 0.10),            # uses DEFAULT_PITCH_DEG
]

# ---------------------------------------------------------------------------

def _quat_from_pitch_deg(pitch_deg: float) -> math_helpers.Quat:
    """Create a quaternion rotated 'pitch_deg' about the Y axis."""
    # Boston Dynamics helpers include from_pitch(radians)
    return math_helpers.Quat.from_pitch(math.radians(pitch_deg))

def _mobility_params():
    """Mobility params that assist manipulation (hip height and optional yaw)."""
    body_assist = spot_command_pb2.BodyControlParams.BodyAssistForManipulation(
        enable_hip_height_assist=ENABLE_HIP_HEIGHT_ASSIST,
        enable_body_yaw_assist=False,
    )
    return spot_command_pb2.MobilityParams(
        body_control=spot_command_pb2.BodyControlParams(
            body_assist_for_manipulation=body_assist
        )
    )


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


def _send_pose(robot, command_client, pose: SE3, frame_name: str, seconds: float):
    """Send a single arm pose, optionally with body-follow + mobility assist, and block until done."""
    # Arm pose.
    arm_cmd = RobotCommandBuilder.arm_pose_command(
        pose.x, pose.y, pose.z,
        pose.rot.w, pose.rot.x, pose.rot.y, pose.rot.z,
        frame_name, seconds
    )

    # Stand with mobility params so Spot can lower/adjust to help the arm.
    stand_cmd = RobotCommandBuilder.synchro_stand_command(params=_mobility_params())

    # Optionally tell the base to follow the hand.
    if USE_BODY_FOLLOW:
        follow_cmd = RobotCommandBuilder.follow_arm_command()
        # Just pass commands positionally—no build_on_command kwarg.
        command = RobotCommandBuilder.build_synchro_command(follow_cmd, arm_cmd, stand_cmd)
    else:
        command = RobotCommandBuilder.build_synchro_command(stand_cmd, arm_cmd)

    cmd_id = command_client.robot_command(command)
    block_until_arm_arrives(command_client, cmd_id)



def run(robot):
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    acc = SE3(0, 0, 0, math_helpers.Quat())  # start with identity; rotation handled per-step

    for i, step in enumerate(STEPS, start=1):
        # Parse (dx, dy, dz[, pitch_deg])
        if len(step) == 3:
            dx, dy, dz = step
            pitch_deg = DEFAULT_PITCH_DEG
        elif len(step) == 4:
            dx, dy, dz, pitch_deg = step
        else:
            raise ValueError("Each step must be (dx, dy, dz) or (dx, dy, dz, pitch_deg).")

        # Per-step rotation: “looking down” by default or the provided pitch
        step_rot = _quat_from_pitch_deg(pitch_deg)

        fid_name, vision_T_fid = _find_fiducial(robot, tag_id=TAG_ID)

        offset = SE3(
            (acc.x + dx) if CUMULATIVE else dx,
            (acc.y + dy) if CUMULATIVE else dy,
            (acc.z + dz) if CUMULATIVE else dz,
            step_rot,
        )
        if CUMULATIVE:
            # Accumulate position only; keep rotation controlled per-step (not cumulative).
            acc = SE3(offset.x, offset.y, offset.z, acc.rot)

        # Compute target pose in chosen root frame
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
            f"({offset.x:.3f}, {offset.y:.3f}, {offset.z:.3f}) m in {frame_used} "
            f"pitch={pitch_deg:.1f}° "
            f"[body_follow={'ON' if USE_BODY_FOLLOW else 'OFF'}, hip_assist={'ON' if ENABLE_HIP_HEIGHT_ASSIST else 'OFF'}]"
        )

        _send_pose(robot, command_client, target_pose, frame_used, SECONDS_PER_STEP)

    robot.logger.info("place_brick.run() chain complete.")


# Optional back-compat helper (unchanged)
def _block_until_cartesian_done(robot, command_client, cmd_id):
    while True:
        fb = command_client.robot_command_feedback(cmd_id)
        cart = fb.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback
        if cart.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
            break
        time.sleep(0.1)
