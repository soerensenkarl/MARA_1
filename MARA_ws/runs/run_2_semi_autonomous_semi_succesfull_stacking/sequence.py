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


# Load brick coordinates from json file (same folder as this script)
# --- load brick positions from wall.json (must sit next to sequence.py) ---
WALL_FILE = Path(__file__).with_name("wall.json")
with WALL_FILE.open("r", encoding="utf-8") as f:
    brick_positions = json.load(f)  # expects a list of [x, y, z] triples
if not isinstance(brick_positions, list) or not all(isinstance(p, (list, tuple)) and len(p) == 3 for p in brick_positions):
    raise ValueError(f"wall.json must be a list of [x, y, z] items. Got: {type(brick_positions)}")
# -------------------------------------------------------------------------


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
        robot.logger.info("Powering on…")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Power on failed."

        cmd_client = robot.ensure_client(RobotCommandClient.default_service_name)
        robot.logger.info("Standing…")
        blocking_stand(cmd_client, timeout_sec=20)

        # ---- Loop through all bricks ----
        for i, brick in enumerate(brick_positions):
            robot.logger.info(f"Starting sequence for brick {i+1}/{len(brick_positions)}")

            # 1. Pick brick
            pick_ok = pick.run(
                robot,
                image_source="hand_color_image",
                force_top_down_grasp=True,
                click_ui=True,
                pixel_xy=None,
            )
            assert pick_ok, "Pick (hand camera) failed."

            # 2. Autowalk to the wall
            robot.logger.info("Autowalk: to_wall.walk …")
            walk_ok = walk.play_named(robot, WALK_TO_WALL)
            assert walk_ok, "Autowalk to_wall.walk failed."

            # 3. Place brick at target from wall.json
            x, y, z = map(float, brick_positions[i])  # each item is [x, y, z]
            robot.logger.info(f"Placing brick {i+1}/{len(brick_positions)} at target (x={x:.3f}, y={y:.3f}, z={z:.3f})")
            place_brick.run(robot, target=(x, y, z))  

            # 4. Autowalk back to the source
            robot.logger.info("Autowalk: to_source.walk …")
            walk_ok = walk.play_named(robot, WALK_TO_SOURCE)
            assert walk_ok, "Autowalk to_source.walk failed."


        # ---- Finish: safe sit + power off ----
        robot.logger.info("All bricks complete. Powering off (safe sit)…")
        robot.power_off(cut_immediately=False, timeout_sec=20)

    robot.logger.info("Lease returned. Done.")


if __name__ == "__main__":
    main()
