# sequence.py
import time
from pathlib import Path
import json

import bosdyn.client
import bosdyn.client.util
import bosdyn.mission.client
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
import place_brick as place_brick
import walk
import pick_brick as pick

# ---- Set once here (or swap to env vars) ----
HOST = "192.168.80.3"
CRED_PATH = Path(r"C:\Users\soere\OneDrive\Desktop\spot_creds.txt")  # change this if needed


def load_credentials(path: Path):
    """Read username/password from a two-line text file."""
    with path.open("r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Credential file {path} must contain at least two lines: USER and PASS")
    return lines[0], lines[1]


# --- load brick positions (now supports yaw_deg) from wall.json ---
WALL_FILE = Path(__file__).with_name("wall.json")
with WALL_FILE.open("r", encoding="utf-8") as f:
    brick_rows = json.load(f)  # expects [x, y, z] or [x, y, z, yaw_deg]

if not isinstance(brick_rows, list) or not all(isinstance(p, (list, tuple)) and (3 <= len(p) <= 4) for p in brick_rows):
    raise ValueError(f"wall.json must be a list of [x, y, z] or [x, y, z, yaw_deg] items.")

# Normalize to (x,y,z,yaw_deg)
brick_targets = []
for p in brick_rows:
    if len(p) == 3:
        x, y, z = map(float, p)
        yaw = 0.0
    else:
        x, y, z, yaw = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    brick_targets.append((x, y, z, yaw))
# ------------------------------------------------------------------

WALK_TO_WALL = "to_wall.walk"
WALK_TO_SOURCE = "to_source.walk"


def main():
    USER, PASS = load_credentials(CRED_PATH)

    sdk = bosdyn.client.create_standard_sdk(
        "MARA_Sequence",
        [bosdyn.mission.client.MissionClient]
    )

    robot = sdk.create_robot(HOST)
    robot.authenticate(USER, PASS)
    robot.time_sync.wait_for_sync()

    assert not robot.is_estopped(), "Robot is E-Stopped. Configure external E-Stop first."
    assert robot.has_arm(), "This sequence expects a Spot with an arm."

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Power on & stand once for the entire sequence
        print("Powering on…")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Power on failed."

        cmd_client = robot.ensure_client(RobotCommandClient.default_service_name)
        print("Standing…")
        blocking_stand(cmd_client, timeout_sec=20)

        # ---- Loop through each target: AI pick -> walk -> place -> walk back ----
        N = len(brick_targets)
        print(f"Running {N} iterations (from wall.json).")

        for i in range(N):
            (x, y, z, yaw_deg) = brick_targets[i]
            print(f"Starting sequence {i+1}/{N}")

            # 1) Pick brick at source (AI-based from hand camera, no UI)
            print(f"Picking brick {i+1}/{N} at the source (AI detection via hand camera)…")
            # Explicit, but still auto-nearest (no UI)
            pick_ok = pick.run(
                robot,
                image_source="hand_color_image",  # must be a DEPTH_U16 source
                click_ui=False,                   # auto-pick nearest
                force_top_down_grasp=True
            )
            assert pick_ok, "Pick (hand camera) failed."

            # 2) Autowalk to the wall
            print("Autowalk: to_wall.walk …")
            walk_ok = walk.play_named(robot, WALK_TO_WALL)
            assert walk_ok, "Autowalk to_wall.walk failed."

            # 3) Place brick at target with yaw
            print(f"Placing brick {i+1}/{N} at (x={x:.3f}, y={y:.3f}, z={z:.3f}), yaw={yaw_deg:.1f}°")
            place_brick.run(robot, target=(x, y, z), yaw_deg=yaw_deg)

            # 4) Autowalk back to the source
            print("Autowalk: to_source.walk …")
            walk_ok = walk.play_named(robot, WALK_TO_SOURCE)
            assert walk_ok, "Autowalk to_source.walk failed."

        # ---- Finish: safe sit + power off ----
        print("All bricks complete. Powering off (safe sit)…")
        robot.power_off(cut_immediately=False, timeout_sec=20)

    print("Lease returned. Done.")


if __name__ == "__main__":
    main()
