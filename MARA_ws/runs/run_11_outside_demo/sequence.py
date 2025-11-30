# sequence.py — now logs timestamps for each step (pick, walk, place)
import time
from datetime import datetime
from pathlib import Path
import csv
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
CRED_PATH = Path(r"C:\Users\soere\OneDrive\Desktop\spot_creds.txt")  # change if needed

# ---- Logging setup ----
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CSV_PATH = RESULTS_DIR / f"sequence_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

CSV_HEADERS = [
    "iteration",
    "step",
    "event",
    "timestamp_iso",
    "timestamp_unix",
]

def log_event(csv_path: Path, iteration: int, step: str, event: str):
    """Append one row to CSV log."""
    now = time.time()
    iso = datetime.fromtimestamp(now).isoformat(timespec="seconds")
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([iteration, step, event, iso, f"{now:.3f}"])


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

brick_targets = []
for p in brick_rows:
    if len(p) == 3:
        x, y, z = map(float, p)
        yaw = 0.0
    else:
        x, y, z, yaw = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    brick_targets.append((x, y, z, yaw))

# --- load source pick positions ---
SOURCE_FILE = Path(__file__).with_name("source.json")
with SOURCE_FILE.open("r", encoding="utf-8") as f:
    source_rows = json.load(f)

source_targets = []
for p in source_rows:
    if len(p) == 3:
        x, y, z = map(float, p)
        yaw = 0.0
    else:
        x, y, z, yaw = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    source_targets.append((x, y, z, yaw))

WALK_TO_WALL = "to_wall.walk"
WALK_TO_SOURCE = "to_source.walk"


def main():
    USER, PASS = load_credentials(CRED_PATH)

    # create new CSV log with headers
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
    print(f"Logging timestamps to: {CSV_PATH}")

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
        print("Powering on…")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Power on failed."

        cmd_client = robot.ensure_client(RobotCommandClient.default_service_name)
        print("Standing…")
        blocking_stand(cmd_client, timeout_sec=20)

        N = min(len(brick_targets), len(source_targets))
        print(f"Running {N} iterations.")

        for i in range(N):
            (sx, sy, sz, syaw_deg) = source_targets[i]
            (x, y, z, yaw_deg) = brick_targets[i]
            print(f"Sequence {i+1}/{N}")

            # 1) Pick brick
            log_event(CSV_PATH, i+1, "pick", "start")
            pick.run(robot, target=(sx, sy, sz), yaw_deg=syaw_deg)
            log_event(CSV_PATH, i+1, "pick", "end")

            # 2) Walk to wall
            log_event(CSV_PATH, i+1, "walk_to_wall", "start")
            walk_ok = walk.play_named(robot, WALK_TO_WALL)
            log_event(CSV_PATH, i+1, "walk_to_wall", "end")
            assert walk_ok, "Autowalk to_wall.walk failed."

            # 3) Place brick
            log_event(CSV_PATH, i+1, "place", "start")
            place_brick.run(robot, target=(x, y, z), yaw_deg=yaw_deg)
            log_event(CSV_PATH, i+1, "place", "end")

            # 4) Walk back to source
            log_event(CSV_PATH, i+1, "walk_to_source", "start")
            walk_ok = walk.play_named(robot, WALK_TO_SOURCE)
            log_event(CSV_PATH, i+1, "walk_to_source", "end")
            assert walk_ok, "Autowalk to_source.walk failed."

        print("All bricks complete. Powering off…")
        robot.power_off(cut_immediately=False, timeout_sec=20)

    print("Lease returned. Done.")


if __name__ == "__main__":
    main()
