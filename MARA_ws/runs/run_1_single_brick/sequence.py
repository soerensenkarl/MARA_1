# sequence.py
import time
from pathlib import Path

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

# Path to credentials file
CRED_PATH = Path(r"C:\Users\soere\OneDrive\Desktop\spot_creds.txt")  # change this if needed

def load_credentials(path: Path):
    """Read username/password from a two-line text file."""
    with path.open("r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Credential file {path} must contain at least two lines: USER and PASS")
    return lines[0], lines[1]


def main():
    USER, PASS = load_credentials(CRED_PATH)
    
    sdk = bosdyn.client.create_standard_sdk(
    "MARA_Sequence",
    [bosdyn.mission.client.MissionClient]  # ← register Mission service like in Mission Replay example
    )

    robot = sdk.create_robot(HOST)
    robot.authenticate(USER, PASS)
    robot.time_sync.wait_for_sync()

    assert not robot.is_estopped(), "Robot is E-Stopped. Configure external E-Stop first."
    assert robot.has_arm(), "This sequence expects a Spot with an arm."

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Power on & STAND ONCE for the entire sequence
        robot.logger.info("Powering on…")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Power on failed."

        cmd_client = robot.ensure_client(RobotCommandClient.default_service_name)
        robot.logger.info("Standing…")
        blocking_stand(cmd_client, timeout_sec=20)


        # ---- Call actions (they assume standing & ready) ----

        #1.  Pick brick with hand camera
        pick_ok = pick.run(
            robot,
            image_source="frontleft_fisheye_image",
            force_top_down_grasp=True,
            click_ui=True,            # show window, click to pick
            pixel_xy=None,            # or set a pixel programmatically
        )
        assert pick_ok, "Pick (hand camera) failed."

        # 2. Autowalk to build area
        robot.logger.info("Starting Autowalk…")
        walk_ok = walk.play(robot)
        assert walk_ok, "Autowalk failed."

        # 3. Place brick\
        place_brick.run(robot)

        # ---- Finish: safe sit + power off ----
        robot.logger.info("Sequence complete. Powering off (safe sit)…")
        robot.power_off(cut_immediately=False, timeout_sec=20)

    robot.logger.info("Lease returned. Done.")


if __name__ == "__main__":
    main()
