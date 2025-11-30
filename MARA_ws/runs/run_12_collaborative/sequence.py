# sequence.py
# Collaborative two-Spot sequence:
# - Both robots pre-stage in parallel (PICK + WALK).
# - Only PLACE is serialized using a small local token file (strict A, B, A, B... order).
# - Run the *same* script in two terminals. Supply --hostname and --creds for each robot,
#   and set --robot-label=A / --robot-label=B (or use ROBOT_LABEL env var).

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Union, Dict

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand

# Your local modules
import pick_brick as pick
import place_brick as place_brick
import walk


# -----------------------------
# Turn coordinator (local files)
# -----------------------------
class TurnCoordinator:
    """
    Serializes only the final PLACE step using local files (same machine, two terminals):
    - .lock for mutual exclusion via atomic create.
    - next_index.txt holds the integer index of the next target to PLACE.
    """
    def __init__(self, shared_dir: str, total_targets: int):
        self.shared = Path(shared_dir)
        self.shared.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.shared / ".lock"
        self.idx_path  = self.shared / "next_index.txt"
        self.total = int(total_targets)
        if not self.idx_path.exists():
            self._atomic_write(self.idx_path, b"0")

    def acquire_lock(self, retry_delay=0.05):
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return
            except FileExistsError:
                time.sleep(retry_delay)

    def release_lock(self):
        try:
            self.lock_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[turn] WARN: failed to release lock: {e}")

    def _atomic_write(self, path: Path, data: bytes):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _read_index(self) -> int:
        try:
            with open(self.idx_path, "rb") as f:
                raw = f.read().strip() or b"0"
            return int(raw)
        except Exception as e:
            print(f"[turn] WARN: read index failed ({e}); assuming 0")
            return 0

    def _write_index(self, value: int):
        self._atomic_write(self.idx_path, str(int(value)).encode("utf-8"))

    def next_index(self) -> int:
        return self._read_index()

    def commit_next(self, new_index: int):
        self._write_index(new_index)


# -----------------------------
# Target loading helper
# -----------------------------
def _coerce_target(entry: Union[Dict, List, Tuple]) -> Dict[str, float]:
    """
    Coerce a wall.json entry to a dict {'x':..., 'y':..., 'z':..., ...}
    Accepts either dicts with x,y,z keys or lists/tuples [x,y,z,(opt...)].
    Any extra keys are preserved (e.g., yaw, id).
    """
    if isinstance(entry, dict):
        if not all(k in entry for k in ("x", "y", "z")):
            raise ValueError(f"Target dict missing x/y/z keys: {entry}")
        return dict(entry)
    elif isinstance(entry, (list, tuple)):
        if len(entry) < 3:
            raise ValueError(f"Target list/tuple must have at least 3 numbers: {entry}")
        return {"x": float(entry[0]), "y": float(entry[1]), "z": float(entry[2])}
    else:
        raise ValueError(f"Unsupported target entry type: {type(entry)}")

def load_wall_targets(path: Union[str, Path]) -> List[Dict[str, float]]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "targets" in data:
        raw_list = data["targets"]
    else:
        raw_list = data
    targets = [_coerce_target(e) for e in raw_list]
    print(f"[sequence] Loaded {len(targets)} targets from {p}")
    return targets


# -----------------------------
# Core run: pre-stage in parallel, serialize place
# -----------------------------
def run(robot, *, robot_label: str = None, shared_turn_dir: str = None, wall_path: Union[str, Path] = "wall.json"):
    # ----- Identify this terminal / robot -----
    if robot_label is None:
        robot_label = os.getenv("ROBOT_LABEL", "A")
    robot_label = (robot_label or "A").strip().upper()
    my_parity = 0 if robot_label == "A" else 1
    print(f"[sequence] Robot label: {robot_label} (A=even, B=odd)")

    # ----- Local shared folder for token files (same computer, two terminals) -----
    if shared_turn_dir is None:
        shared_turn_dir = os.getenv("COLLAB_TURN_DIR", str(Path.cwd() / "turns"))
    print(f"[sequence] Shared turn dir: {shared_turn_dir}")

    # ----- Load targets -----
    wall_targets = load_wall_targets(wall_path)

    # ----- Lease + stand -----
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        print("[sequence] Lease acquired.")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        print("[sequence] Powering on and standing...")
        blocking_stand(command_client, timeout_sec=10)
        print("[sequence] Standing. Starting collaborative loop.")

        coordinator = TurnCoordinator(shared_turn_dir, total_targets=len(wall_targets))

        while True:
            # ===== Determine the specific target index THIS robot should prepare next =====
            coordinator.acquire_lock()
            global_next = coordinator.next_index()

            # Everyone done?
            if global_next >= len(wall_targets):
                coordinator.release_lock()
                print(f"[sequence] All targets completed (next_index={global_next}). Exiting.")
                break

            # Choose the next index for THIS robot (by parity) at or after global_next
            desired_idx = global_next if (global_next % 2) == my_parity else (global_next + 1)

            # Past the list?
            if desired_idx >= len(wall_targets):
                coordinator.release_lock()
                print(f"[sequence] No remaining indices for {robot_label} (desired={desired_idx}). Exiting.")
                break

            target = wall_targets[desired_idx]
            print(f"[sequence] [{robot_label}] Preparing target #{desired_idx} while waiting turn. (global_next={global_next})")
            coordinator.release_lock()

            # ---------- PREP STAGE (parallel allowed): PICK + WALK ----------
            pick_ok = pick.run(
                robot,
                image_source="hand_color_image",  # AI auto-nearest hand grasp
                click_ui=False,
                force_top_down_grasp=True
            )
            if not pick_ok:
                print(f"[sequence] [{robot_label}] Pick failed for target #{desired_idx}; retrying soon.")
                time.sleep(0.8)
                continue

            # Walk to the desired target approach/pose (your walk.run handles details)
            walk.run(robot, target=target)

            # ---------- SERIALIZED PLACE (token controlled) ----------
            print(f"[sequence] [{robot_label}] Staged at target #{desired_idx}. Waiting for turn to place...")
            while True:
                coordinator.acquire_lock()
                current_turn = coordinator.next_index()

                if current_turn == desired_idx:
                    # It's our turn to place.
                    coordinator.release_lock()
                    print(f"[sequence] [{robot_label}] >>> PLACE TURN for #{desired_idx}")

                    place_ok = place_brick.run(robot, target=target)
                    if not place_ok:
                        print(f"[sequence] [{robot_label}] Place failed at #{desired_idx}; NOT advancing token. Retrying place shortly.")
                        time.sleep(0.8)
                        # Try to place again (still our turn).
                        continue

                    # Placement succeeded — advance the global turn atomically.
                    coordinator.acquire_lock()
                    still_turn = coordinator.next_index()
                    if still_turn == desired_idx:
                        coordinator.commit_next(desired_idx + 1)
                        print(f"[sequence] [{robot_label}] Committed next_index -> {desired_idx + 1}")
                    else:
                        print(f"[sequence] [{robot_label}] WARN: token moved externally (current={still_turn}, expected={desired_idx}).")
                    coordinator.release_lock()

                    # Done with this target; proceed to prepare the next one.
                    time.sleep(0.2)
                    break

                else:
                    # Not our turn yet — release and wait briefly.
                    coordinator.release_lock()
                    time.sleep(0.15)
                    continue

        print(f"[sequence] {robot_label} finished its assigned (serialized) placements.")
        print("[sequence] Returning lease and ending.")


# -----------------------------
# Command-line entry
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collaborative two-Spot sequence (pre-stage parallel, serialized place).")
    bosdyn.client.util.add_common_arguments(parser)
    parser.add_argument("--robot-label", choices=["A", "B"], default=os.getenv("ROBOT_LABEL", "A"),
                        help="Label for this terminal/robot. A handles even indices; B handles odd.")
    parser.add_argument("--shared-turn-dir", default=os.getenv("COLLAB_TURN_DIR", str(Path.cwd() / "turns")),
                        help="Local folder (same machine) to store turn token files.")
    parser.add_argument("--wall", default="wall.json", help="Path to wall.json target list.")
    parser.add_argument("--creds", required=True, help="Path to credentials file (first line = username, second line = password).")
    args = parser.parse_args()

    # ---- Read credentials ----
    creds_path = Path(args.creds)
    if not creds_path.exists():
        print(f"[sequence] ERROR: credentials file not found at {creds_path}")
        return False
    with open(creds_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) < 2:
        print(f"[sequence] ERROR: credentials file must have username and password on separate lines.")
        return False
    username, password = lines[0], lines[1]

    # ---- Create SDK + connect ----
    sdk = bosdyn.client.create_standard_sdk("mara-collab-sequence")
    robot = sdk.create_robot(args.hostname)
    robot.authenticate(username, password)
    robot.time_sync.wait_for_sync()
    print(f"[sequence] Authenticated as '{username}' to {args.hostname}")

    # ---- Run collaborative sequence ----
    try:
        run(robot,
            robot_label=args.robot_label,
            shared_turn_dir=args.shared_turn_dir,
            wall_path=args.wall)
        return True
    except KeyboardInterrupt:
        print("\n[sequence] Interrupted by user.")
        return False
    except Exception as e:
        print(f"[sequence] ERROR: {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
