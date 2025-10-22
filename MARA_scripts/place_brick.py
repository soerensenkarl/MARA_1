# Copyright (c) 2023 Boston Dynamics, Inc.
# All rights reserved.
#
# SDK License: 20191101-BDSDK-SL

"""
place_brick.py — Move Spot's hand to poses RELATIVE to an AprilTag fiducial,
with simple support for chaining multiple relative offsets.

Typical (single move, same as before):
    place_brick.run(robot, tag_id=4, target_x=0.20, target_z=0.20)

Chained moves (sequence of offsets in the fiducial frame, in meters):
    steps = [
        (0.00, 0.00, 0.10),  # 10 cm above
        (-0.10, 0.00, 0.00), # 10 cm left
        (0.10, 0.00, 0.00),  # 10 cm right
    ]
    place_brick.run(robot, tag_id=4, steps=steps, seconds=1.5)

Notes:
- Each step is relative to the fiducial frame origin (not cumulative). If you want
  cumulative behavior, set cumulative=True.
- Uses WorldObjectClient to discover the tag and uses the fiducial's own
  transforms_snapshot (pattern from fiducial_follow.py) to build vision_T_fiducial.
"""

import time
from typing import Iterable, List, Optional, Tuple, Union

from bosdyn.api import arm_command_pb2, world_object_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    VISION_FRAME_NAME,
    ODOM_FRAME_NAME,
    get_a_tform_b,
)
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient


SE3 = math_helpers.SE3Pose


def _find_fiducial(robot, tag_id: Optional[int], fiducial_frame: Optional[str],
                   wait_sec: float = 3.0) -> Tuple[str, SE3]:
    """Return (fiducial_frame_name, vision_T_fiducial) using the fiducial's own snapshot."""
    wo_client = robot.ensure_client(WorldObjectClient.default_service_name)
    t0 = time.time()
    chosen = None

    while True:
        resp = wo_client.list_world_objects(object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG])
        objs = resp.world_objects

        if tag_id is not None:
            chosen = next((o for o in objs if o.apriltag_properties.tag_id == tag_id), None)
        elif fiducial_frame is not None:
            chosen = next((o for o in objs if o.apriltag_properties.frame_name_fiducial == fiducial_frame), None)
        else:
            chosen = objs[0] if objs else None

        if chosen is not None:
            break

        if time.time() - t0 > wait_sec:
            raise RuntimeError("No AprilTag detected by the World Object service within timeout. "
                               "Ensure the tag is visible and well lit.")
        time.sleep(0.1)

    fid_frame = chosen.apriltag_properties.frame_name_fiducial
    vision_T_fid = get_a_tform_b(chosen.transforms_snapshot, VISION_FRAME_NAME, fid_frame)
    if vision_T_fid is None:
        raise RuntimeError(f"Could not compute VISION_T_{fid_frame}.")
    return fid_frame, vision_T_fid


def _normalize_steps(
    steps: Optional[Iterable[Union[Tuple[float, float, float], dict]]]
) -> List[Tuple[float, float, float, Optional[float]]]:
    """
    Convert user-provided steps into a list of (x, y, z, seconds) in meters.
    Accepts:
      - (x, y, z)
      - {'x':..., 'y':..., 'z':..., 'seconds': optional}
    """
    if steps is None:
        return []
    norm = []
    for s in steps:
        if isinstance(s, dict):
            x = float(s.get('x', 0.0))
            y = float(s.get('y', 0.0))
            z = float(s.get('z', 0.0))
            secs = s.get('seconds', None)
            norm.append((x, y, z, secs))
        else:
            x, y, z = s  # type: ignore
            norm.append((float(x), float(y), float(z), None))
    return norm


def _send_pose(command_client, pose: SE3, frame_name: str, seconds: float):
    arm_cmd = RobotCommandBuilder.arm_pose_command(
        pose.x, pose.y, pose.z,
        pose.rot.w, pose.rot.x, pose.rot.y, pose.rot.z,
        frame_name, seconds
    )
    cmd = RobotCommandBuilder.build_synchro_command(arm_cmd)
    cmd_id = command_client.robot_command(cmd)
    block_until_arm_arrives(command_client, cmd_id)


def run(
    robot,
    seconds: float = 2.0,
    # Choose a fiducial by tag id (preferred) or frame name:
    tag_id: Optional[int] = 4,
    fiducial_frame: Optional[str] = None,
    # One-shot target (kept for backward compatibility):
    target_x: float = 0.20,
    target_y: float = 0.00,
    target_z: float = 0.20,
    # New: list of steps to chain. If provided, overrides the single target.
    steps: Optional[Iterable[Union[Tuple[float, float, float], dict]]] = None,
    # Behavior:
    cumulative: bool = False,       # If True, steps accumulate offsets
    refresh_each_step: bool = True, # Reacquire vision_T_fid for every step
    do_stow: bool = False,
    send_in_odom: bool = False,
):
    """
    Move the hand to one or more poses defined in the fiducial frame.
    Preconditions: robot is authenticated, time-synced, leased, powered on, standing.
    """
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    # Resolve steps
    steps_norm = _normalize_steps(steps)
    use_steps = len(steps_norm) > 0

    # Base orientation (identity; adjust if you want wrist pitch/yaw)
    goal_rot = math_helpers.Quat(w=1.0, x=0.0, y=0.0, z=0.0)

    # Helper to get latest vision_T_fid and (optionally) odom_T_vision
    def get_frames():
        fid_name, vision_T_fid = _find_fiducial(robot, tag_id=tag_id, fiducial_frame=fiducial_frame)
        odom_T_vision = None
        if send_in_odom:
            rs = robot_state_client.get_robot_state()
            tf = rs.kinematic_state.transforms_snapshot
            odom_T_vision_local = get_a_tform_b(tf, ODOM_FRAME_NAME, VISION_FRAME_NAME)
            if odom_T_vision_local is None:
                raise RuntimeError("Could not compute odom_T_vision.")
            odom_T_vision = odom_T_vision_local
        return fid_name, vision_T_fid, odom_T_vision

    # Accumulator for cumulative steps
    acc = SE3(0, 0, 0, goal_rot)

    # If doing a single move (back-compat path)
    if not use_steps:
        steps_norm = [(target_x, target_y, target_z, None)]

    # Execute each step
    for idx, (sx, sy, sz, secs_override) in enumerate(steps_norm, start=1):
        if refresh_each_step or idx == 1:
            fid_name, vision_T_fid, odom_T_vision = get_frames()

        step_offset = SE3(sx, sy, sz, goal_rot)
        if cumulative:
            acc = SE3(acc.x + sx, acc.y + sy, acc.z + sz, goal_rot)
            offset = acc
        else:
            offset = step_offset

        seconds_this = float(secs_override) if secs_override is not None else seconds

        if send_in_odom:
            odom_T_goal = odom_T_vision * (vision_T_fid * offset)  # type: ignore
            robot.logger.info(f"[{idx}/{len(steps_norm)}] Move to {fid_name} + "
                              f"({offset.x:.3f},{offset.y:.3f},{offset.z:.3f}) m in ODOM.")
            _send_pose(command_client, odom_T_goal, ODOM_FRAME_NAME, seconds_this)
        else:
            vision_T_goal = vision_T_fid * offset
            robot.logger.info(f"[{idx}/{len(steps_norm)}] Move to {fid_name} + "
                              f"({offset.x:.3f},{offset.y:.3f},{offset.z:.3f}) m in VISION.")
            _send_pose(command_client, vision_T_goal, VISION_FRAME_NAME, seconds_this)

    if do_stow:
        stow_cmd = RobotCommandBuilder.arm_stow_command()
        command_client.robot_command(stow_cmd)
        time.sleep(1.0)

    robot.logger.info("place_brick.run() complete.")
