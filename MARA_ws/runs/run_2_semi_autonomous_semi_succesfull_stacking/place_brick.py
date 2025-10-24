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
#     place_brick.run(robot, target=(x, y, z))   # uses lay_brick(x, y, z)
# or, for back-compat (uses the hardcoded STEPS below):
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

# Each step can be (dx, dy, dz) or (dx, dy, dz, pitch_deg) or include a gripper token.
# (Back-compat STEPS removed — run(...) now requires an explicit target.)

def move(x: float, y: float, z: float, *, pitch: Optional[float] = None,
         time: Optional[float] = None, grip=None) -> dict:
    """
    User-friendly step:
    Returns a dict the parser understands.
    """
    return {
        'x': float(x), 'y': float(y), 'z': float(z),
        'pitch': None if pitch is None else float(pitch),
        'seconds': None if time is None else float(time),
        'grip': grip
    }


def lay_brick(x: float, y: float, z: float):
    """
    Return a list of steps. You can mix formats:
      - move(x,y,z, pitch=..., time=..., grip=...)  <-- recommended
      - (dx,dy,dz)
      - (dx,dy,dz,pitch)
      - (dx,dy,dz,'open'|'close'|bool|float[0..1])
      - (dx,dy,dz,pitch,grip)
      - (dx,dy,dz,pitch,grip,seconds)
      - (dx,dy,dz,grip,seconds)
      - (dx,dy,dz,pitch,seconds)
    """
    return [
        # Take brick
        # move(0, -1, 0.70, grip = 0.65, pitch = 0.00, time=2),
        # move(0, -1, 0.70, grip = 0.65, pitch = 0.00, time=2),
        # move(0, -1, 0.70, grip = 0.4, pitch = 0.00, time=0),
        # move(0, -1, 0.70, grip = 0.4, pitch = 0.00, time=1.5),
        # move(x, y, 0.50, grip= 0.4),

        # place brick
        move(x, y, z + 0.30, time=1),
        move(x, y, z + 0.055),
        move(x, y, z + 0.055, grip='open', time=0.0),                       # down (default time)
        move(x - 0.02, y, z + 0.06, time = 0.5),
        move(x, y, 0.3, time=0.5),
    ]



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
    """
    Parses either:
      - dict from move(...): {'x','y','z','pitch','seconds','grip'}
      - tuples:
          (dx,dy,dz)
          (dx,dy,dz,pitch)
          (dx,dy,dz,grip)
          (dx,dy,dz,pitch,grip)
          (dx,dy,dz,pitch,grip,seconds)
          (dx,dy,dz,grip,seconds)
          (dx,dy,dz,pitch,seconds)
    Returns: (dx, dy, dz, pitch_deg, grip, seconds_opt)
    """
    # Dict form (preferred)
    if isinstance(step, dict):
        for k in ('x', 'y', 'z'):
            if k not in step:
                raise ValueError("move(...) dict must include x, y, z.")
        dx, dy, dz = float(step['x']), float(step['y']), float(step['z'])
        pitch_deg = float(step['pitch']) if step.get('pitch') is not None else DEFAULT_PITCH_DEG
        seconds = float(step['seconds']) if step.get('seconds') is not None else None
        grip = _parse_grip(step.get('grip')) if ('grip' in step and step['grip'] is not None) else None
        return dx, dy, dz, pitch_deg, grip, seconds

    # Tuple forms
    n = len(step)
    if n < 3:
        raise ValueError("Step must have at least (dx,dy,dz).")

    dx, dy, dz = step[0], step[1], step[2]
    pitch_deg = DEFAULT_PITCH_DEG
    grip = None
    seconds = None

    if n == 3:
        pass

    elif n == 4:
        fourth = step[3]
        if isinstance(fourth, (int, float)) and not isinstance(fourth, bool):
            # Interpret as pitch (back-compat)
            pitch_deg = float(fourth)
        else:
            # Interpret as grip
            grip = _parse_grip(fourth)

    elif n == 5:
        fourth, fifth = step[3], step[4]
        if isinstance(fourth, (int, float)) and not isinstance(fourth, bool):
            # (dx,dy,dz,pitch, ? )
            pitch_deg = float(fourth)
            if isinstance(fifth, (int, float)) and not isinstance(fifth, bool):
                # seconds
                seconds = float(fifth)
            else:
                # grip
                grip = _parse_grip(fifth)
        else:
            # (dx,dy,dz,grip,seconds?)
            grip = _parse_grip(fourth)
            if isinstance(fifth, (int, float)) and not isinstance(fifth, bool):
                seconds = float(fifth)
            else:
                raise ValueError("5th element should be seconds (number) when 4th is grip.")

    elif n == 6:
        # (dx,dy,dz,pitch,grip,seconds)
        pitch_deg = float(step[3])
        grip = _parse_grip(step[4])
        seconds = float(step[5])

    else:
        raise ValueError("Step must be length 3..6 or a dict from move(...).")

    return dx, dy, dz, pitch_deg, grip, seconds



def _quat_from_pitch_deg(pitch_deg: float) -> math_helpers.Quat:
    """Create a quaternion rotated 'pitch_deg' about the Y axis."""
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


def run(robot, target: Tuple[float, float, float]):
    """
    Execute the steps to place a brick.

    - target=(x,y,z) is required; deprecated STEPS usage removed.
    """
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    # Build steps for this run (STEPS deprecated; target is required).
    steps_to_run = lay_brick(*target)

    acc = SE3(0, 0, 0, math_helpers.Quat())  # rotation controlled per-step

    for i, step in enumerate(steps_to_run, start=1):
        dx, dy, dz, pitch_deg, grip, step_seconds = _split_step(step)
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

        secs_used = step_seconds if step_seconds is not None else SECONDS_PER_STEP
        grip_str = "keep" if grip is None else (f"{grip:.2f}" if isinstance(grip, float) else str(grip))

        robot.logger.info(
            f"[{i}/{len(steps_to_run)}] Move to {fid_name} + "
            f"({offset.x:.3f}, {offset.y:.3f}, {offset.z:.3f}) m in {frame_used} "
            f"pitch={pitch_deg:.1f}°, grip={grip_str}, seconds={secs_used:.2f}, "
            f"[body_follow={'ON' if USE_BODY_FOLLOW else 'OFF'}, hip_assist={'ON' if ENABLE_HIP_HEIGHT_ASSIST else 'OFF'}]"
        )

        # ✅ Send the command for THIS step, using per-step time.
        _send_pose(robot, command_client, target_pose, frame_used, secs_used, grip)
            # --- After placing the brick ---
    robot.logger.info("Stowing arm after placing brick.")
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    stow_cmd = RobotCommandBuilder.arm_stow_command()
    stow_command_id = command_client.robot_command(stow_cmd)
    block_until_arm_arrives(command_client, stow_command_id, timeout_sec=3.0)
    robot.logger.info("Arm successfully stowed.")


    robot.logger.info("place_brick.run() chain complete.")



# Optional back-compat helper (unchanged)
def _block_until_cartesian_done(robot, command_client, cmd_id):
    while True:
        fb = command_client.robot_command_feedback(cmd_id)
        cart = fb.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback
        if cart.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
            break
        time.sleep(0.1)
