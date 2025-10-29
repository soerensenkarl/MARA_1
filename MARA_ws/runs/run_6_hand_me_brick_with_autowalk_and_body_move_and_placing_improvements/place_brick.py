# Copyright (c) 2023 Boston Dynamics, Inc.
# All rights reserved.
#
# place_brick.py — Place a brick at (x,y,z) relative to an AprilTag fiducial.
# - Walks Spot's base to x-0.8 (in fiducial frame), heading aligned to fiducial +X.
# - Then executes a short arm motion sequence to place the brick.
#
# Usage (from sequence.py):
#     import place_brick
#     place_brick.run(robot, target=(x, y, z))

import time
import math
from typing import Tuple, Optional

from bosdyn.api import world_object_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
    block_for_trajectory_cmd,
)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient

SE3 = math_helpers.SE3Pose
Quat = math_helpers.Quat

# ----------------------------
# User configuration
# ----------------------------
TAG_ID = 4                      # AprilTag to follow
SECONDS_PER_STEP = 2.0          # Default arm move time
SEND_IN_ODOM = False            # Command frame (False=vision, True=odom)

USE_BODY_FOLLOW = True          # Base follows the hand while moving the arm
ENABLE_HIP_HEIGHT_ASSIST = True # Allow base height/yaw assist during manipulation

DEFAULT_PITCH_DEG = 90.0        # End-effector pitch if not specified
PREPLACE_X_OFFSET = -0.8        # Base walks to x-0.8 before placing
BASE_WALK_SPEED = 0.4           # m/s, used to estimate trajectory end time

# ----------------------------
# Small helpers
# ----------------------------
def move(x: float, y: float, z: float, *, pitch: Optional[float] = None,
         time: Optional[float] = None, grip: Optional[str] = None) -> dict:
    """Convenience for step dicts."""
    return {'x': float(x), 'y': float(y), 'z': float(z),
            'pitch': pitch, 'seconds': time, 'grip': grip}

def lay_brick(x: float, y: float, z: float):
    """Simple place sequence."""
    return [
        move(x,     y, z + 0.30, time=1.0),
        move(x,     y, z + 0.05),
        {"sleep": 0.2},  # wait before open
        "open",
        move(x - .02, y, z + 0.05, time=0.5),
        move(x - .02, y, 0.30,     time=0.5),
    ]

def _parse_grip(grip: Optional[str]) -> Optional[float]:
    """'open' -> 1.0, 'close' -> 0.0, None -> None."""
    if grip is None:
        return None
    g = str(grip).strip().lower()
    if g in ('open', 'o'):
        return 1.0
    if g in ('close', 'closed', 'c'):
        return 0.0
    raise ValueError("Grip must be 'open', 'close', or None.")

def _send_gripper_only(command_client, *, open_frac: float, seconds: float = 0.25):
    """Open/close the gripper without moving the arm."""
    subcmds = [
        RobotCommandBuilder.synchro_stand_command(params=_mobility_params()),
        RobotCommandBuilder.claw_gripper_open_fraction_command(open_frac),
    ]
    cmd = RobotCommandBuilder.build_synchro_command(*subcmds)
    command_client.robot_command(cmd)
    # No dedicated "block until gripper done" helper; a short dwell is sufficient.
    time.sleep(max(0.0, seconds))


def _split_step(step: dict):
    """Normalize a step dict -> (x,y,z,pitch_deg,grip_frac,seconds)."""
    for k in ('x', 'y', 'z'):
        if k not in step:
            raise ValueError("move(...) must include x, y, z.")
    pitch_deg = float(step['pitch']) if step.get('pitch') is not None else DEFAULT_PITCH_DEG
    seconds   = float(step['seconds']) if step.get('seconds') is not None else None
    grip_frac = _parse_grip(step.get('grip'))
    return float(step['x']), float(step['y']), float(step['z']), pitch_deg, grip_frac, seconds

def _quat_from_pitch_deg(pitch_deg: float) -> Quat:
    return Quat.from_pitch(math.radians(pitch_deg))

def _mobility_params():
    body_assist = spot_command_pb2.BodyControlParams.BodyAssistForManipulation(
        enable_hip_height_assist=ENABLE_HIP_HEIGHT_ASSIST,
        enable_body_yaw_assist=False,
    )
    return spot_command_pb2.MobilityParams(
        body_control=spot_command_pb2.BodyControlParams(
            body_assist_for_manipulation=body_assist
        )
    )

def _find_fiducial(robot, tag_id: int, wait_sec: float = 3.0):
    """Return (fiducial_frame_name, vision_T_fid)."""
    wo = robot.ensure_client(WorldObjectClient.default_service_name)
    t0 = time.time()
    chosen = None
    while True:
        resp = wo.list_world_objects(object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG])
        chosen = next((o for o in resp.world_objects if o.apriltag_properties.tag_id == tag_id), None)
        if chosen or (time.time() - t0) > wait_sec:
            break
        time.sleep(0.1)
    if not chosen:
        raise RuntimeError(f"No AprilTag with ID {tag_id} detected.")
    fid_frame = chosen.apriltag_properties.frame_name_fiducial
    vision_T_fid = get_a_tform_b(chosen.transforms_snapshot, VISION_FRAME_NAME, fid_frame)
    if vision_T_fid is None:
        raise RuntimeError(f"Could not compute transform to fiducial frame '{fid_frame}'.")
    return fid_frame, vision_T_fid

def _send_pose(command_client, pose: SE3, frame_name: str, seconds: float,
               grip_open_frac: Optional[float]):
    """Send arm pose (with optional gripper and base-follow) and wait for arrival."""
    arm_cmd = RobotCommandBuilder.arm_pose_command(
        pose.x, pose.y, pose.z,
        pose.rot.w, pose.rot.x, pose.rot.y, pose.rot.z,
        frame_name, seconds
    )
    stand_cmd = RobotCommandBuilder.synchro_stand_command(params=_mobility_params())

    subcmds = [arm_cmd, stand_cmd]
    if USE_BODY_FOLLOW:
        subcmds.insert(0, RobotCommandBuilder.follow_arm_command())
    if grip_open_frac is not None:
        subcmds.append(RobotCommandBuilder.claw_gripper_open_fraction_command(grip_open_frac))

    cmd = RobotCommandBuilder.build_synchro_command(*subcmds)
    cmd_id = command_client.robot_command(cmd)
    block_until_arm_arrives(command_client, cmd_id)

def _preplace_move_base(command_client, used_T_fid: SE3, x: float, y: float, frame_name: str):
    """Walk base to (x + PREPLACE_X_OFFSET, y), heading aligned with fiducial +X."""
    goal_fid = SE3(x + PREPLACE_X_OFFSET, y, 0.0, Quat())
    goal_in_used = used_T_fid * goal_fid
    gx, gy = goal_in_used.x, goal_in_used.y

    # Heading: direction of fiducial +X expressed in the command frame.
    p0 = used_T_fid * SE3(0.0, 0.0, 0.0, Quat())
    p1 = used_T_fid * SE3(1.0, 0.0, 0.0, Quat())
    heading = math.atan2(p1.y - p0.y, p1.x - p0.x)

    print(f"[pre-place] Walking base to ({gx:.3f}, {gy:.3f}) m in {frame_name}, "
          f"heading={math.degrees(heading):.1f}° (aligned to fiducial +X)")

    traj_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=gx, goal_y=gy, goal_heading=heading, frame_name=frame_name, params=_mobility_params()
    )

    # Conservative end-time: distance from fiducial origin plus margin.
    dist = math.hypot(gx - p0.x, gy - p0.y)
    end_time_secs = time.time() + max(5.0, dist / max(0.1, BASE_WALK_SPEED) + 3.0)

    cmd_id = command_client.robot_command(traj_cmd, end_time_secs=end_time_secs)
    block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=30.0)

# ----------------------------
# Entry point
# ----------------------------
def run(robot, target: Tuple[float, float, float]):
    """Place a brick at (x,y,z) relative to TAG_ID fiducial."""
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    x, y, z = target

    # Fiducial pose in chosen command frame.
    _, vision_T_fid = _find_fiducial(robot, tag_id=TAG_ID)
    if SEND_IN_ODOM:
        rs = robot_state_client.get_robot_state()
        tf = rs.kinematic_state.transforms_snapshot
        odom_T_vision = get_a_tform_b(tf, ODOM_FRAME_NAME, VISION_FRAME_NAME)
        if odom_T_vision is None:
            raise RuntimeError("Could not compute odom_T_vision.")
        frame_used = ODOM_FRAME_NAME
        used_T_fid = odom_T_vision * vision_T_fid
    else:
        frame_used = VISION_FRAME_NAME
        used_T_fid = vision_T_fid

    # Move base before placing (aligns body X with fiducial +X).
    _preplace_move_base(command_client, used_T_fid, x, y, frame_used)

    # Arm steps.
    steps = lay_brick(x, y, z)
    for i, step in enumerate(steps, start=1):
        # --- Handle gripper-only steps (no pose) ---
        if isinstance(step, str) and step.lower() in ('open', 'close'):
            open_frac = 1.0 if step.lower() == 'open' else 0.0
            print(f"Gripper-only step: {step.lower()}")
            _send_gripper_only(command_client, open_frac=open_frac, seconds=0.25)
            continue

        if isinstance(step, dict) and ('open' in step or 'close' in step) and not any(k in step for k in ('x', 'y', 'z')):
            # Dict form like {"open": True, "seconds": 0.3} or {"close": True}
            if step.get('open', False):
                open_frac = 1.0
                label = 'open'
            elif step.get('close', False):
                open_frac = 0.0
                label = 'close'
            else:
                raise ValueError("Gripper-only dict must include 'open' or 'close' = True.")

            dwell = float(step.get('seconds', 0.25))
            print(f"Gripper-only step: {label} (dwell {dwell:.2f}s)")
            _send_gripper_only(command_client, open_frac=open_frac, seconds=dwell)
            continue

        # --- Handle explicit sleep steps early ---
        if isinstance(step, dict) and 'sleep' in step:
            dur = float(step['sleep'])
            print(f"Pausing {dur:.2f} s before next step...")
            time.sleep(max(0.0, dur))
            continue

        # --- Refresh fiducial for each move (robust to drift) ---
        fid_name, vision_T_fid = _find_fiducial(robot, tag_id=TAG_ID)

        dx, dy, dz, pitch_deg, grip_frac, step_seconds = _split_step(step)
        rot = _quat_from_pitch_deg(pitch_deg)
        offset = SE3(dx, dy, dz, rot)

        # --- Compute target pose in chosen frame ---
        if SEND_IN_ODOM:
            rs = robot_state_client.get_robot_state()
            tf = rs.kinematic_state.transforms_snapshot
            odom_T_vision = get_a_tform_b(tf, ODOM_FRAME_NAME, VISION_FRAME_NAME)
            if odom_T_vision is None:
                raise RuntimeError("Could not compute odom_T_vision.")
            target_pose = odom_T_vision * (vision_T_fid * offset)
            frame = ODOM_FRAME_NAME
        else:
            target_pose = vision_T_fid * offset
            frame = VISION_FRAME_NAME

        # --- Build and send arm command ---
        secs_used = step_seconds if step_seconds is not None else SECONDS_PER_STEP
        grip_label = "keep" if grip_frac is None else ("open" if grip_frac >= 0.5 else "close")

        print(
            f"[{i}/{len(steps)}] Move to {fid_name} + ({dx:.3f}, {dy:.3f}, {dz:.3f}) m "
            f"in {frame} pitch={pitch_deg:.1f}°, grip={grip_label}, seconds={secs_used:.2f}, "
            f"[body_follow={'ON' if USE_BODY_FOLLOW else 'OFF'}, "
            f"hip_assist={'ON' if ENABLE_HIP_HEIGHT_ASSIST else 'OFF'}]"
        )

        _send_pose(command_client, target_pose, frame, secs_used, grip_frac)



    print("Stowing arm after placing brick.")
    stow_cmd = RobotCommandBuilder.arm_stow_command()
    stow_id = command_client.robot_command(stow_cmd)
    block_until_arm_arrives(command_client, stow_id, timeout_sec=3.0)
    print("Arm successfully stowed. place_brick.run() done.")
