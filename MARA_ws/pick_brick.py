# pick_brick.py — auto-pick nearest depth pixel from the arm depth camera, then optional lift/back/stow
#
# Minimal change for auto-retry:
# - Added `retries: int = 1` to run(...)
# - Track `fallback_triggered` and, if True and retries > 0, recursively re-run run(...) with retries-1.
#
# Other notes:
# - Keeps your existing behavior, including duplicate image acquire lines, sleeps, and fallback step-back.

import time
from typing import Optional, Tuple

import cv2
import numpy as np
import math
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_odom_tform_body

import bosdyn.client
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2, robot_state_pb2
from bosdyn.client import frame_helpers, math_helpers
from bosdyn.client.estop import EstopClient
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
    block_until_arm_arrives,
)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, BODY_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_odom_tform_body
from bosdyn.client.frame_helpers import get_se2_a_tform_b


# ---- UI globals (unchanged; used only if click_ui=True) ----
g_image_click = None
g_image_display = None


def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        color = (30, 30, 30)
        thickness = 2
        image_title = "Click to grasp"
        h, w = clone.shape[0], clone.shape[1]
        cv2.line(clone, (0, y), (w, y), color, thickness)
        cv2.line(clone, (x, 0), (x, h), color, thickness)
        cv2.imshow(image_title, clone)


def _verify_not_estopped(robot):
    """Check that Spot is not currently estopped."""
    estop_client = robot.ensure_client(EstopClient.default_service_name)
    status = estop_client.get_status()
    if status.stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        level_name = estop_pb2.EstopStopLevel.Name(status.stop_level)
        endpoints = [e.endpoint_name for e in status.endpoints]
        raise RuntimeError(
            f"Robot is estopped: stop_level={level_name}. "
            f"Active endpoints: {', '.join(endpoints) or 'none'}.\n"
            "Make sure your E-Stop is registered and in HOLD/RUNNING, then try again."
        )


def _add_grasp_constraint(opts, grasp, robot_state_client):
    use_vector = opts.get("force_top_down_grasp") or opts.get("force_horizontal_grasp")
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector:
        if opts.get("force_top_down_grasp"):
            # align gripper +X with -Z of vision (top-down)
            axis_on_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
            axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=-1)
        else:  # horizontal grasp: align +Y with +Z
            axis_on_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)
            axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=1)

        c = grasp.grasp_params.allowable_orientation.add()
        c.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper)
        c.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(axis_to_align)
        c.vector_alignment_with_tolerance.threshold_radians = 0.05

    elif opts.get("force_45_angle_grasp"):
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = frame_helpers.get_vision_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        )
        body_Q_grasp = math_helpers.Quat.from_pitch(1.7)  # ~45°
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp
        c = grasp.grasp_params.allowable_orientation.add()
        c.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())
        c.rotation_with_tolerance.threshold_radians = 0.10

    elif opts.get("force_squeeze_grasp"):
        c = grasp.grasp_params.allowable_orientation.add()
        c.squeeze_grasp.SetInParent()


def _get_image(robot, image_source):
    image_client = robot.ensure_client(ImageClient.default_service_name)
    responses = image_client.get_image_from_sources([image_source])
    if len(responses) != 1:
        raise RuntimeError(f"Expected 1 image, got {len(responses)}")
    image = responses[0]

    # Decode
    dtype = (
        np.uint16
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
        else np.uint8
    )
    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(image.shot.image.rows, image.shot.image.cols)
    else:
        img = cv2.imdecode(img, -1)
    return image, img


def _nearest_depth_pixel(depth_img: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Return (x, y) for the centroid of the nearest area.
    Depth is expected in millimeters (DEPTH_U16). Ignores a 2-pixel border and
    clamps absurd values > 10 m.
    """
    if depth_img is None or depth_img.size == 0:
        return None

    crop = depth_img[2:-2, 2:-2].astype(np.uint32)
    mask = (crop > 0) & (crop < 10000)
    if not np.any(mask):
        return None

    # Find the minimum valid depth
    min_depth = np.min(crop[mask])

    # Create a mask for the nearest "area" within 10 mm of the minimum
    area_mask = (crop <= min_depth + 10) & mask

    # Find the centroid (mean x, y) of this area
    y_coords, x_coords = np.where(area_mask)
    if y_coords.size == 0:
        return None

    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)

    # Return the centroid, adjusted for the 2-pixel border
    return int(center_x + 2), int(center_y + 2)


def _get_debug_image_path() -> str:
    """Return the absolute path for debug images, creating it if needed."""
    import os
    ws_path = os.path.dirname(os.path.abspath(__file__))
    debug_path = os.path.join(ws_path, "debug_images")
    os.makedirs(debug_path, exist_ok=True)
    return debug_path


def _save_debug_image(image, img: np.ndarray, chosen_xy: Tuple[int, int]) -> None:
    """Save raw depth (.npy) and a colorized PNG, or a color JPG, with the chosen pixel marked."""
    import os
    from datetime import datetime

    try:
        save_dir = _get_debug_image_path()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        is_depth = img.dtype == np.uint16
        file_type = "depth" if is_depth else "color"
        base = os.path.join(save_dir, f"spot_{file_type}_{ts}")

        if is_depth:
            # Save raw U16 for exact debugging
            np.save(base + ".npy", img)

            # Make a safe visualization
            d = img.astype(np.float32)
            nonzero = d[d > 0]
            if nonzero.size > 0:
                d_min = float(np.percentile(nonzero, 1.0))
                d_max = float(np.percentile(nonzero, 99.0))
            else:
                d_min, d_max = 0.0, 1000.0
            d_clamped = np.clip(d, d_min, d_max)
            vis = cv2.convertScaleAbs((d_clamped - d_min) / max(d_max - d_min, 1e-6), alpha=255.0)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            out_path = base + ".png"
        else:
            vis = img.copy()
            out_path = base + ".jpg"

        # Draw chosen pixel
        if chosen_xy is not None:
            x, y = int(chosen_xy[0]), int(chosen_xy[1])
            cv2.drawMarker(vis, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

        cv2.imwrite(out_path, vis)
    except Exception as e:
        print(f"[pick_brick] Warning: could not save debug image: {e}")


def _fallback_step_back(
    robot,
    command_client,
    robot_state_client,
    *,
    distance_m: float = 0.2,
    seconds: float = 2.0,
):
    """Stow arm, then step straight backwards by distance_m relative to BODY, executed in ODOM."""
    # 1) Stow the arm
    robot.logger.info("Fallback: stowing arm…")
    stow_cmd = RobotCommandBuilder.arm_stow_command()
    stow_id = command_client.robot_command(stow_cmd)
    block_until_arm_arrives(command_client, stow_id)

    # 2) Build a BODY-frame offset and express the goal in ODOM
    try:
        transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        # Negative X in BODY frame = step back
        body_tform_goal = math_helpers.SE2Pose(x=-abs(distance_m), y=0.0, angle=0.0)

        # Convert to ODOM frame
        out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        out_tform_goal = out_tform_body * body_tform_goal

        # 3) Send the SE2 trajectory point with an end_time.
        mobility_params = RobotCommandBuilder.mobility_params()
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x,
            goal_y=out_tform_goal.y,
            goal_heading=out_tform_goal.angle,
            frame_name=ODOM_FRAME_NAME,
            params=mobility_params,
        )
        cmd_id = command_client.robot_command(
            command=robot_cmd,
            end_time_secs=time.time() + max(seconds, 1.0),
        )

        # 4) Wait on trajectory feedback
        from bosdyn.client.robot_command import block_for_trajectory_cmd
        robot.logger.info(f"Fallback: stepping back {distance_m:.2f} m…")
        block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=seconds + 3.0)
        robot.logger.info("Fallback: step-back complete.")
    except Exception as e:
        print(f"Fallback step-back failed: {e}")


def run(
    robot,
    *,
    image_source: str = "hand_depth_in_hand_color_frame",
    force_top_down_grasp: bool = True,
    force_horizontal_grasp: bool = False,
    force_45_angle_grasp: bool = False,
    force_squeeze_grasp: bool = False,
    click_ui: bool = False,  # default to automatic nearest-depth selection
    pixel_xy: Optional[Tuple[int, int]] = None,
    feedback_poll_s: float = 0.25,
    stow_after_grasp: bool = True,
    retries: int = 1,  # <-- MINIMAL CHANGE: number of auto-retries when fallback triggers
) -> bool:
    _verify_not_estopped(robot)

    # Ensure clients
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    # Track if we triggered a fallback in this attempt
    fallback_triggered = False  # <-- MINIMAL CHANGE

    # Put arm in a sane pose for imaging
    robot.logger.info("Unstowing arm (arm_ready) before taking image…")
    unstow_cmd = RobotCommandBuilder.arm_ready_command()
    unstow_id = command_client.robot_command(unstow_cmd)
    block_until_arm_arrives(command_client, unstow_id, timeout_sec=3.0)
    robot.logger.info("Arm is ready for imaging.")

    # Open gripper fully before imaging
    robot.logger.info("Opening gripper fully before imaging…")
    open_cmd = RobotCommandBuilder.claw_gripper_open_command()
    open_id = command_client.robot_command(open_cmd)
    block_until_arm_arrives(command_client, open_id, timeout_sec=1.0)

    # Move gripper up 30cm and pitch ~45° down relative to BODY frame for a good view
    robot.logger.info("Moving gripper up 30cm and pitching down for imaging…")
    robot_state = robot_state_client.get_robot_state()
    transforms = robot_state.kinematic_state.transforms_snapshot
    body_T_hand = get_a_tform_b(transforms, BODY_FRAME_NAME, HAND_FRAME_NAME)

    pitch_down_q = math_helpers.Quat.from_pitch(1.7)
    new_rot = body_T_hand.rotation * pitch_down_q

    pitch_cmd = RobotCommandBuilder.arm_pose_command(
        body_T_hand.x,
        body_T_hand.y,
        body_T_hand.z + 0.20,
        new_rot.w,
        new_rot.x,
        new_rot.y,
        new_rot.z,
        BODY_FRAME_NAME,
        1.5,
    )
    pitch_id = command_client.robot_command(pitch_cmd)
    block_until_arm_arrives(command_client, pitch_id)

    # Acquire image once (fix duplicate)
    image, img = _get_image(robot, image_source)

    # Acquire image once (fix duplicate)
    image, img = _get_image(robot, image_source)

    # Always save the image that was used for picking, even if picking is canceled.
    _save_debug_image(image, img, None)

    # Decide target pixel
    target_px = pixel_xy

    # Optional click UI (debug)
    if click_ui and target_px is None:
        robot.logger.info("Click on an object to grasp (press Q to abort)…")
        global g_image_click, g_image_display
        g_image_click = None
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            v = img.astype(np.float32)
            v[v == 0] = np.nan
            mn = np.nanmin(v) if np.isfinite(v).any() else 0.0
            mx = np.nanmax(v) if np.isfinite(v).any() else 1000.0
            norm = (np.nan_to_num(v, nan=mn) - mn) / max(mx - mn, 1e-6)
            g_image_display = (norm * 255.0).astype(np.uint8)
        else:
            g_image_display = img
        title = "Click to grasp"
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, cv_mouse_callback)
        cv2.imshow(title, g_image_display)
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                robot.logger.info("User canceled grasp.")
                cv2.destroyAllWindows()
                return False
        target_px = (int(g_image_click[0]), int(g_image_click[1]))
        cv2.destroyAllWindows()

    # Auto-pick nearest valid depth pixel (default path)
    if target_px is None:
        if image.shot.image.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            raise ValueError(
                "Automatic nearest-pixel mode requires a depth image (PIXEL_FORMAT_DEPTH_U16). "
                f"Got pixel_format={image.shot.image.pixel_format} from source '{image_source}'."
            )
        auto_px = _nearest_depth_pixel(img)
        if auto_px is None:
            _save_debug_image(image, img, None)
            raise RuntimeError("Could not find a valid depth pixel (image may be empty or invalid).")
        target_px = auto_px
        robot.logger.info(f"Auto-selected nearest depth pixel: {target_px} from '{image_source}'")

    # Save debug image with the chosen pixel marked
    _save_debug_image(image, img, target_px)

    # Build grasp request at the target pixel
    robot.logger.info(f"Picking at image pixel {target_px} from source '{image_source}'")
    pick_vec = geometry_pb2.Vec2(x=target_px[0], y=target_px[1])

    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=image.shot.transforms_snapshot,
        frame_name_image_sensor=image.shot.frame_name_image_sensor,
        camera_model=image.source.pinhole,
    )

    # Closer to palm for bricks. 0.0=palm, 1.0=fingertip.
    grasp.grasp_params.grasp_palm_to_fingertip = 0.2

    _add_grasp_constraint(
        {
            "force_top_down_grasp": force_top_down_grasp,
            "force_horizontal_grasp": force_horizontal_grasp,
            "force_45_angle_grasp": force_45_angle_grasp,
            "force_squeeze_grasp": force_squeeze_grasp,
        },
        grasp,
        robot_state_client,
    )

    req = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
    cmd = manipulation_api_client.manipulation_api_command(manipulation_api_request=req)

    # Feedback loop
    while True:
        fb_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd.manipulation_cmd_id
        )
        fb = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=fb_req
        )
        state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(fb.current_state)
        print(f"Current state: {state_name}")
        if fb.current_state in (
            manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
            manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
        ):
            break
        time.sleep(feedback_poll_s)

    success = fb.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED
    robot.logger.info("Finished grasp: %s", "SUCCESS" if success else "FAILED")

    # -- Check gripper state and open only if it's fully closed AND not holding an item --
    state = robot_state_client.get_robot_state()
    ms = state.manipulator_state

    open_pct = getattr(ms, "gripper_open_percentage", None)
    if open_pct is None:
        open_pct = getattr(ms, "gripper_open_percent", None)

    holding = bool(getattr(ms, "is_gripper_holding_item", False))

    robot.logger.info(
        "Gripper status — open%%: %s, holding: %s",
        f"{open_pct:.1f}" if open_pct is not None else "n/a",
        holding,
    )

    print(open_pct)

    # Define "fully closed" conservatively as <= 10% open.
    should_open = (open_pct is not None and open_pct <= 10.0)

    if should_open:
        robot.logger.info("Gripper appears fully closed — opening gripper and triggering fallback().")
        open_cmd = RobotCommandBuilder.claw_gripper_open_command()
        open_id = command_client.robot_command(open_cmd)
        block_until_arm_arrives(command_client, open_id, timeout_sec=1.5)

        _fallback_step_back(robot, command_client, robot_state_client, distance_m=0.3, seconds=2.0)

        robot.logger.info("Waiting 3 s after fallback before continuing…")
        fallback_triggered = True  # <-- MINIMAL CHANGE
    else:
        robot.logger.info("Gripper not fully closed (or reading unavailable) — continuing.")

    if success and stow_after_grasp:
        # Allow stow while holding (set carry state override)
        override = manipulation_api_pb2.ApiGraspOverrideRequest(
            api_grasp_override=manipulation_api_pb2.ApiGraspOverride(
                override_request=manipulation_api_pb2.ApiGraspOverride.OVERRIDE_HOLDING
            ),
            carry_state_override=manipulation_api_pb2.ApiGraspedCarryStateOverride(
                override_request=robot_state_pb2.ManipulatorState.CARRY_STATE_CARRIABLE_AND_STOWABLE
            ),
        )
        try:
            manipulation_api_client.grasp_override_command(override)
        except Exception:
            pass

        robot.logger.info("Stowing arm…")
        stow_cmd = RobotCommandBuilder.arm_stow_command()
        stow_id = command_client.robot_command(stow_cmd)
        block_until_arm_arrives(command_client, stow_id)

        # Repeat the post-action gripper check after a successful pick + stow.
        state_after_stow = robot_state_client.get_robot_state()
        ms2 = state_after_stow.manipulator_state

        open_pct2 = getattr(ms2, "gripper_open_percentage", None)
        if open_pct2 is None:
            open_pct2 = getattr(ms2, "gripper_open_percent", None)

        holding2 = bool(getattr(ms2, "is_gripper_holding_item", False))

        robot.logger.info(
            "Post-stow gripper status — open%%: %s, holding: %s",
            f"{open_pct2:.1f}" if open_pct2 is not None else "n/a",
            holding2,
        )

        should_open2 = (open_pct2 is not None and open_pct2 <= 10.0)

        if should_open2:
            robot.logger.info("Post-stow: gripper still fully closed — opening and triggering fallback().")
            open_cmd2 = RobotCommandBuilder.claw_gripper_open_command()
            open_id2 = command_client.robot_command(open_cmd2)
            block_until_arm_arrives(command_client, open_id2, timeout_sec=1.5)

            _fallback_step_back(robot, command_client, robot_state_client, distance_m=0.3, seconds=2.0)

            robot.logger.info("Waiting 3 s after fallback before continuing…")
            time.sleep(3.0)
            fallback_triggered = True  # <-- MINIMAL CHANGE
        else:
            robot.logger.info("Post-stow gripper looks fine — continuing.")



    # ---- MINIMAL CHANGE: auto-retry if fallback happened ----
    if fallback_triggered and retries > 0:
        robot.logger.info(f"Fallback triggered — re-running pick_brick.run (retries left: {retries})")
        return run(
            robot,
            image_source=image_source,
            force_top_down_grasp=force_top_down_grasp,
            force_horizontal_grasp=force_horizontal_grasp,
            force_45_angle_grasp=force_45_angle_grasp,
            force_squeeze_grasp=force_squeeze_grasp,
            click_ui=click_ui,
            pixel_xy=pixel_xy,
            feedback_poll_s=feedback_poll_s,
            stow_after_grasp=stow_after_grasp,
            retries=retries - 1,
        )

    return success
