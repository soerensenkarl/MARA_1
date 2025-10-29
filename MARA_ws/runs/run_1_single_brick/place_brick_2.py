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
SECONDS_PER_STEP = 2.0     # time per move
CUMULATIVE = False         # if True, steps build on each other
SEND_IN_ODOM = False       # command frame (False = vision)

# Mobility assist / body follow
USE_BODY_FOLLOW = True         # if True, command base to follow the hand
ENABLE_HIP_HEIGHT_ASSIST = True  # if True, allow Spot to lower/adjust base height/yaw to help arm

# ---- User configuration ----
DEFAULT_PITCH_DEG = 90  # “looking down” if a step doesn't specify pitch

# Each step can be (dx, dy, dz) or (dx, dy, dz, pitch_deg)
STEPS = [
################# First layer #################
    (-0.30, 0.00, 0.70),
    (-0.30, 0.00, 0.30),
    (-0.30, 0.00, 0.05),
    (-0.30, 0.00, 0.05, 'open'),
    (-0.32, 0.00, 0.05),
    (-0.30, 0.00, 0.70),
]
# ---------------------------------------------------------------------------

def _parse_grip(grip_val) -> Optional[float]:
    """Return open fraction in [0,1], or None to keep current."""
    if grip_val is None:
        return None
    if isinstance(grip_val, bool):
        return 1.0 if grip_val else 0.0
    if isinstance(grip_val, (int, float)):
        return max(0.0, min(1.0, float(grip_val)))
    if isinstance(grip_val, str):
        g = grip_val.strip().lower()
        if g in ('open', 'o'):
            return 1.0
        if g in ('close', 'closed', 'c'):
            return 0.0
    raise ValueError("Grip must be one of: 'open'/'close'/True/False/float[0..1].")

def _split_step(step):
    """Accept (dx,dy,dz), (dx,dy,dz,pitch), (dx,dy,dz,'open'), or (dx,dy,dz,pitch,grip)."""
    if len(step) == 3:
        dx, dy, dz = step
        pitch_deg = DEFAULT_PITCH_DEG
        grip = None
    elif len(step) == 4:
        dx, dy, dz, fourth = step
        # If 4th is numeric -> pitch; else treat as gripper
        if isinstance(fourth, (int, float)) and not isinstance(fourth, bool):
            pitch_deg = float(fourth)
            grip = None
        else:
            pitch_deg = DEFAULT_PITCH_DEG
            grip = _parse_grip(fourth)
    elif len(step) == 5:
        dx, dy, dz, pitch_deg, grip_val = step
        pitch_deg = float(pitch_deg)
        grip = _parse_grip(grip_val)
    else:
        raise ValueError("Each step must be (dx,dy,dz), (dx,dy,dz,pitch), "
                         "(dx,dy,dz,grip), or (dx,dy,dz,pitch,grip).")
    return dx, dy, dz, pitch_deg, grip



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


def _send_pose(robot, command_client, pose: SE3, frame_name: str, seconds: float,
               grip_open_frac: Optional[float]):
    """Send arm pose (+ optional gripper), with body-follow + mobility assist, and wait."""
    arm_cmd = RobotCommandBuilder.arm_pose_command(
        pose.x, pose.y, pose.z,
        pose.rot.w, pose.rot.x, pose.rot.y, pose.rot.z,
        frame_name, seconds
    )

    stand_cmd = RobotCommandBuilder.synchro_stand_command(params=_mobility_params())

    subcmds = []
    if USE_BODY_FOLLOW:
        subcmds.append(RobotCommandBuilder.follow_arm_command())

    # Order isn’t strict; include stand so hip-height assist can do its thing.
    subcmds.extend([arm_cmd, stand_cmd])

    # Optional gripper command (only if specified this step)
    if grip_open_frac is not None:
        grip_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(grip_open_frac)
        subcmds.append(grip_cmd)

    command = RobotCommandBuilder.build_synchro_command(*subcmds)
    cmd_id = command_client.robot_command(command)
    block_until_arm_arrives(command_client, cmd_id)


def run(robot):
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    acc = SE3(0, 0, 0, math_helpers.Quat())  # rotation controlled per-step

    for i, step in enumerate(STEPS, start=1):
        dx, dy, dz, pitch_deg, grip = _split_step(step)
        step_rot = _quat_from_pitch_deg(pitch_deg)

        fid_name, vision_T_fid = _find_fiducial(robot, tag_id=TAG_ID)

        offset = SE3(
            (acc.x + dx) if CUMULATIVE else dx,
            (acc.y + dy) if CUMULATIVE else dy,
            (acc.z + dz) if CUMULATIVE else dz,
            step_rot,
        )
        if CUMULATIVE:
            acc = SE3(offset.x, offset.y, offset.z, acc.rot)  # keep rotation separate

        # Compute target pose
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

        grip_str = "keep" if grip is None else (f"{grip:.2f}" if isinstance(grip, float) else str(grip))
        robot.logger.info(
            f"[{i}/{len(STEPS)}] Move to {fid_name} + "
            f"({offset.x:.3f}, {offset.y:.3f}, {offset.z:.3f}) m in {frame_used} "
            f"pitch={pitch_deg:.1f}°, grip={grip_str}, "
            f"[body_follow={'ON' if USE_BODY_FOLLOW else 'OFF'}, hip_assist={'ON' if ENABLE_HIP_HEIGHT_ASSIST else 'OFF'}]"
        )

        _send_pose(robot, command_client, target_pose, frame_used, SECONDS_PER_STEP, grip)

    robot.logger.info("place_brick.run() chain complete.")


# Optional back-compat helper (unchanged)
def _block_until_cartesian_done(robot, command_client, cmd_id):
    while True:
        fb = command_client.robot_command_feedback(cmd_id)
        cart = fb.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback
        if cart.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
            break
        time.sleep(0.1)
