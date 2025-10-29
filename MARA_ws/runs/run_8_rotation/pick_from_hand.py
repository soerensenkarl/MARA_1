# pick_brick.py
# Simple sequence:
# 1) Unstow arm
# 2) Open gripper (e.g. 0.75)
# 3) Hold for 2 seconds
# 4) Close gripper (e.g. 0.40)
# 5) Stow arm
#
# Callable from sequence.py: pick_brick.run(robot, ...)

import time
from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
    block_until_arm_arrives,
)


def run(
    robot,
    *,
    open_frac: float = 0.70,
    close_frac: float = 0.40,
    hold_s: float = 1.0,
    **kwargs,  # ignore extra args passed by sequence.py
) -> bool:
    """Unstow → open → hold → close → stow sequence."""
    try:
        cmd_client = robot.ensure_client(RobotCommandClient.default_service_name)

        print("[pick_brick] Unstowing arm...")
        cmd_id = cmd_client.robot_command(RobotCommandBuilder.arm_ready_command())
        block_until_arm_arrives(cmd_client, cmd_id, timeout_sec=2.0)

        print(f"[pick_brick] Opening gripper to {open_frac:.2f}...")
        cmd_id = cmd_client.robot_command(
            RobotCommandBuilder.claw_gripper_open_fraction_command(open_frac)
        )
        block_until_arm_arrives(cmd_client, cmd_id, timeout_sec=2.0)

        print(f"[pick_brick] Holding for {hold_s:.1f} seconds...")
        time.sleep(hold_s)

        print(f"[pick_brick] Closing gripper to {close_frac:.2f}...")
        cmd_id = cmd_client.robot_command(
            RobotCommandBuilder.claw_gripper_open_fraction_command(close_frac)
        )
        block_until_arm_arrives(cmd_client, cmd_id, timeout_sec=1.0)

        print("[pick_brick] Stowing arm...")
        cmd_id = cmd_client.robot_command(RobotCommandBuilder.arm_stow_command())
        block_until_arm_arrives(cmd_client, cmd_id, timeout_sec=2.0)

        print("[pick_brick] Done.")
        return True

    except Exception as e:
        print(f"[pick_brick] Error: {e}")
        return False


if __name__ == "__main__":
    print(
        "Run this via sequence.py or import and call run(robot). "
        "Example: run(robot, open_frac=0.8, close_frac=0.4)"
    )
