# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""
Autowalk replay as a callable function for use from sequence.py.

Minimal changes:
- Removed CLI, SDK init, power-on, and stand commands (sequence.py handles those).
- Exposes run(...) which uploads/localizes (optional) and plays an Autowalk mission
  using the existing authenticated, time-synced, powered-on, standing robot and an
  active LeaseKeepAlive managed by sequence.py.
"""



import os
import time
from pathlib import Path


import bosdyn.api.mission
import bosdyn.client
import bosdyn.client.lease
import bosdyn.geometry
import bosdyn.mission.client
import bosdyn.util
from bosdyn.api import robot_state_pb2
from bosdyn.api.autowalk import walks_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.api.mission import mission_pb2, nodes_pb2
from bosdyn.client.autowalk import AutowalkResponseError
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient



SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WALK_DIR = SCRIPT_DIR / "walk.walk"             # folder containing graph, snapshots, missions/
# Optionally let env override; otherwise use the default above
WALK_DIR = Path(os.environ.get("WALK_DIR", str(DEFAULT_WALK_DIR)))
# If not set, we'll auto-detect the .walk file inside WALK_DIR/missions in play()
WALK_FILE = os.environ.get("WALK_FILE", None)



# Default runtime options
WALK_DEFAULTS = {
    "upload_timeout": 300.0,
    "mission_timeout": 3.0,
    "noloc": False,
    "disable_alternate_route_finding": False,
    "disable_directed_exploration": False,
    "strict_mode": False,
    "duration": 0.0,   # 0.0 = run once
    "static_mode": False,
}



# -------------------------
# Public entry point
# -------------------------
def run(
    robot,
    walk_directory: str,
    walk_filename: str,
    *,
    upload_timeout: float = 300.0,
    mission_timeout: float = 3.0,
    noloc: bool = False,
    disable_alternate_route_finding: bool = False,
    disable_directed_exploration: bool = False,
    strict_mode: bool = False,
    duration: float = 0.0,
    static_mode: bool = False,
) -> bool:
    """Replay an Autowalk mission using an already-authenticated, powered-on, standing robot.

    Preconditions handled by sequence.py:
      - robot is authenticated & time-synced
      - a LeaseKeepAlive is active
      - robot is powered on and standing
    """
    path_following_mode = map_pb2.Edge.Annotations.PATH_MODE_UNKNOWN
    if strict_mode:
        disable_alternate_route_finding = True
        disable_directed_exploration = True
        path_following_mode = map_pb2.Edge.Annotations.PATH_MODE_STRICT
        print('[ STRICT MODE ENABLED: Alternate waypoints and directed exploration disabled ]')

    walk_directory = Path(walk_directory)
    mission_file = walk_directory / "missions" / walk_filename
    if not mission_file.is_file():
        robot.logger.fatal(f"Unable to find mission file: {mission_file}")
        return False

    # Use existing lease context from sequence.py — just get the client.
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

    # Init clients + upload graph/mission (no power/stand here)
    robot_state_client, command_client, mission_client, graph_nav_client = init_clients(
        robot, mission_file, walk_directory, True,  # do_map_load=True for Autowalk
        disable_alternate_route_finding, upload_timeout)

    # Optional localization (skip if noloc=True)
    if not noloc:
        try:
            _ = graph_nav_client.download_graph(timeout=upload_timeout)
            robot.logger.info('Localizing robot...')
            localization = nav_pb2.Localization()
            graph_nav_client.set_localization(
                initial_guess_localization=localization,
                ko_tform_body=None, max_distance=None, max_yaw=None,
                fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST)
        except Exception as e:
            robot.logger.warning(f'Localization attempt failed: {e}')

    # Run once or repeat for duration (static_mode = do nothing)
    if static_mode:
        robot.logger.info('Static mode requested; not starting mission.')
        return True

    if duration == 0.0:
        return run_mission(robot, mission_client, lease_client, True,  # fail_on_question
                           mission_timeout, disable_directed_exploration, path_following_mode)
    else:
        return repeat_mission(robot, mission_client, lease_client, duration, True,
                              mission_timeout, disable_directed_exploration, path_following_mode)


# -------------------------
# Helpers (unchanged logic)
# -------------------------
def init_clients(robot, mission_file, walk_directory, do_map_load, disable_alternate_route_finding,
                 upload_timeout):
    """Initialize clients"""

    graph_nav_client = None

    # Create autowalk and mission client
    robot.logger.info('Creating mission client...')
    mission_client = robot.ensure_client(bosdyn.mission.client.MissionClient.default_service_name)
    robot.logger.info('Creating autowalk client...')
    autowalk_client = robot.ensure_client(
        bosdyn.client.autowalk.AutowalkClient.default_service_name)

    if do_map_load:
        if not os.path.isdir(walk_directory):
            robot.logger.fatal(f'Unable to find walk directory: {walk_directory}.')
            return None, None, None, None

        # Create graph-nav client
        robot.logger.info('Creating graph-nav client...')
        graph_nav_client = robot.ensure_client(
            bosdyn.client.graph_nav.GraphNavClient.default_service_name)

        # Clear map state and localization
        robot.logger.info('Clearing graph-nav state...')
        graph_nav_client.clear_graph()

        # Upload map to robot
        upload_graph_and_snapshots(robot, graph_nav_client, walk_directory,
                                   disable_alternate_route_finding, upload_timeout)

        # Try autowalk upload; fallback to mission upload if needed
        try:
            upload_autowalk(robot, autowalk_client, mission_file, upload_timeout)
        except Exception:
            robot.logger.warning(
                f'Failed to parse/load autowalk from {mission_file}. Attempting to parse as node proto.'
            )
            upload_mission(robot, mission_client, mission_file, upload_timeout)
    else:
        # Upload mission to robot
        upload_mission(robot, mission_client, mission_file, upload_timeout)

    # Create command + state clients
    robot.logger.info('Creating command client...')
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    robot.logger.info('Creating robot state client...')
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    return robot_state_client, command_client, mission_client, graph_nav_client


def upload_graph_and_snapshots(robot, client, path, disable_alternate_route_finding,
                               upload_timeout):
    """Upload the graph and snapshots to the robot"""

    # Load the graph from disk.
    graph_filename = os.path.join(path, 'graph')
    robot.logger.info(f'Loading graph from {graph_filename}')

    with open(graph_filename, 'rb') as graph_file:
        data = graph_file.read()
        current_graph = map_pb2.Graph()
        current_graph.ParseFromString(data)
        robot.logger.info(
            f'Loaded graph has {len(current_graph.waypoints)} waypoints and {len(current_graph.edges)} edges'
        )

    if disable_alternate_route_finding:
        for edge in current_graph.edges:
            edge.annotations.disable_alternate_route_finding = True

    # Load the waypoint snapshots from disk.
    current_waypoint_snapshots = dict()
    for waypoint in current_graph.waypoints:
        if len(waypoint.snapshot_id) == 0:
            continue
        snapshot_filename = os.path.join(path, 'waypoint_snapshots', waypoint.snapshot_id)
        robot.logger.info(f'Loading waypoint snapshot from {snapshot_filename}')
        with open(snapshot_filename, 'rb') as snapshot_file:
            waypoint_snapshot = map_pb2.WaypointSnapshot()
            waypoint_snapshot.ParseFromString(snapshot_file.read())
            current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot

    # Load the edge snapshots from disk.
    current_edge_snapshots = dict()
    for edge in current_graph.edges:
        if len(edge.snapshot_id) == 0:
            continue
        snapshot_filename = os.path.join(path, 'edge_snapshots', edge.snapshot_id)
        robot.logger.info(f'Loading edge snapshot from {snapshot_filename}')
        with open(snapshot_filename, 'rb') as snapshot_file:
            edge_snapshot = map_pb2.EdgeSnapshot()
            edge_snapshot.ParseFromString(snapshot_file.read())
            current_edge_snapshots[edge_snapshot.id] = edge_snapshot

    # Upload the graph to the robot.
    robot.logger.info('Uploading the graph and snapshots to the robot...')
    true_if_empty = not len(current_graph.anchoring.anchors)
    response = client.upload_graph(graph=current_graph, generate_new_anchoring=true_if_empty,
                                   timeout=upload_timeout)
    robot.logger.info('Uploaded graph.')

    # Upload the snapshots to the robot.
    for snapshot_id in response.unknown_waypoint_snapshot_ids:
        waypoint_snapshot = current_waypoint_snapshots[snapshot_id]
        client.upload_waypoint_snapshot(waypoint_snapshot=waypoint_snapshot, timeout=upload_timeout)
        robot.logger.info(f'Uploaded {waypoint_snapshot.id}')

    for snapshot_id in response.unknown_edge_snapshot_ids:
        edge_snapshot = current_edge_snapshots[snapshot_id]
        client.upload_edge_snapshot(edge_snapshot=edge_snapshot, timeout=upload_timeout)
        robot.logger.info(f'Uploaded {edge_snapshot.id}')


def upload_autowalk(robot, autowalk_client, filename, upload_timeout):
    """Upload the autowalk mission to the robot"""

    # Load the autowalk from disk
    robot.logger.info(f'Loading autowalk from {filename}')

    autowalk_proto = walks_pb2.Walk()
    with open(filename, 'rb') as walk_file:
        data = walk_file.read()
        autowalk_proto.ParseFromString(data)

    # Upload the mission to the robot and report the load_autowalk_response
    robot.logger.info('Uploading the autowalk to the robot...')
    try:
        autowalk_client.load_autowalk(autowalk_proto, timeout=upload_timeout)
    except AutowalkResponseError as resp_err:
        load_autowalk_response = resp_err.response
        print(f'failed_nodes:\n{load_autowalk_response.failed_nodes}')
        print(f'failed_elements: {load_autowalk_response.failed_elements}')
        raise resp_err


def upload_mission(robot, client, filename, upload_timeout):
    """Upload the mission to the robot"""

    # Load the mission from disk
    robot.logger.info(f'Loading mission from {filename}')

    mission_proto = nodes_pb2.Node()
    with open(filename, 'rb') as mission_file:
        data = mission_file.read()
        mission_proto.ParseFromString(data)

    # Upload the mission to the robot
    robot.logger.info('Uploading the mission to the robot...')
    client.load_mission(mission_proto, timeout=upload_timeout)
    robot.logger.info('Uploaded mission to robot.')


def run_mission(robot, mission_client, lease_client, fail_on_question, mission_timeout,
                disable_directed_exploration, path_following_mode):
    """Run mission once"""

    robot.logger.info('Running mission')

    mission_state = mission_client.get_state()

    while mission_state.status in (mission_pb2.State.STATUS_NONE, mission_pb2.State.STATUS_RUNNING):
        # Optionally fail if any questions are triggered (common in Autowalks).
        if mission_state.questions and fail_on_question:
            robot.logger.info(
                f'Mission failed by triggering operator question: {mission_state.questions}')
            return False

        body_lease = lease_client.lease_wallet.advance()
        local_pause_time = time.time() + mission_timeout

        play_settings = mission_pb2.PlaySettings(
            disable_directed_exploration=disable_directed_exploration,
            path_following_mode=path_following_mode)

        mission_client.play_mission(local_pause_time, [body_lease], play_settings)
        time.sleep(1)

        mission_state = mission_client.get_state()

    robot.logger.info(f'Mission status = {mission_state.Status.Name(mission_state.status)}')

    return mission_state.status in (mission_pb2.State.STATUS_SUCCESS,
                                    mission_pb2.State.STATUS_PAUSED)


def restart_mission(robot, mission_client, lease_client, mission_timeout):
    """Restart current mission"""

    robot.logger.info('Restarting mission')

    body_lease = lease_client.lease_wallet.advance()
    local_pause_time = time.time() + mission_timeout

    status = mission_client.restart_mission(local_pause_time, [body_lease])
    time.sleep(1)

    return status == mission_pb2.State.STATUS_SUCCESS


def repeat_mission(robot, mission_client, lease_client, total_time, fail_on_question,
                   mission_timeout, disable_directed_exploration, path_following_mode):
    """Repeat mission for period of time"""

    robot.logger.info(f'Repeating mission for {total_time} seconds.')

    # Run first mission
    start_time = time.time()
    mission_success = run_mission(robot, mission_client, lease_client, fail_on_question,
                                  mission_timeout, disable_directed_exploration,
                                  path_following_mode)
    elapsed_time = time.time() - start_time
    robot.logger.info(f'Elapsed time = {elapsed_time} (out of {total_time})')

    if not mission_success:
        robot.logger.info('Mission failed.')
        return False

    # Repeat mission until total time has expired
    while elapsed_time < total_time:
        restart_mission(robot, mission_client, lease_client, mission_timeout=3)
        mission_success = run_mission(robot, mission_client, lease_client, fail_on_question,
                                      mission_timeout, disable_directed_exploration,
                                      path_following_mode)

        elapsed_time = time.time() - start_time
        robot.logger.info(f'Elapsed time = {elapsed_time} (out of {total_time})')

        if not mission_success:
            robot.logger.info('Mission failed.')
            break

    return mission_success


def play(robot, walk_directory: str | Path = None, walk_filename: str = None, **kwargs) -> bool:
    """Simplified one-call interface using defaults or overrides."""
    wd = Path(walk_directory) if walk_directory else WALK_DIR
    wf = walk_filename or WALK_FILE

    # Auto-detect the .walk file if none specified
    if wf is None:
        missions_dir = wd / "missions"
        walk_files = sorted(p.name for p in missions_dir.glob("*.walk"))
        if len(walk_files) == 1:
            wf = walk_files[0]
        else:
            raise FileNotFoundError(
                f"Expected exactly one .walk in {missions_dir}, found: {walk_files or 'none'}"
            )

    opts = {**WALK_DEFAULTS, **kwargs}



    return run(
        robot,
        walk_directory=wd,
        walk_filename=wf,
        upload_timeout=opts["upload_timeout"],
        mission_timeout=opts["mission_timeout"],
        noloc=opts["noloc"],
        disable_alternate_route_finding=opts["disable_alternate_route_finding"],
        disable_directed_exploration=opts["disable_directed_exploration"],
        strict_mode=opts["strict_mode"],
        duration=opts["duration"],
        static_mode=opts["static_mode"],
    )

# Convenience: play a walk when the folder name == .walk filename
def play_named(robot, name: str, **kwargs) -> bool:
    """
    Example: name="to_wall.walk" expects:
      - <this folder>/to_wall.walk/         (contains graph/, waypoint_snapshots/, edge_snapshots/, missions/)
      - <this folder>/to_wall.walk/missions/to_wall.walk
    """
    from pathlib import Path
    walk_dir = (SCRIPT_DIR / name)  # folder named the same as the .walk file
    walk_file = name                # the .walk file inside missions/
    return play(robot, walk_directory=walk_dir, walk_filename=walk_file, **kwargs)

