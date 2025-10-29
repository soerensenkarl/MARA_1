
import time
from typing import Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from openvino import Core  # AI detection backend (from ai_grasp)
import random

import bosdyn.client
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2, robot_state_pb2
from bosdyn.client import frame_helpers, math_helpers
from bosdyn.client.estop import EstopClient
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
    block_until_arm_arrives,
)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import (
    VISION_FRAME_NAME,
    BODY_FRAME_NAME,
    HAND_FRAME_NAME,
    GRAV_ALIGNED_BODY_FRAME_NAME,
    get_a_tform_b,
)
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_se2_a_tform_b


# ---- UI globals (kept for compatibility, though AI path doesn't use them) ----
g_image_click = None
g_image_display = None

# ---- OpenVINO model load (from ai_grasp.py) ----
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path.cwd()/"detection_images"
MODEL_XML = HERE / "model.xml"
MODEL_BIN = HERE / "model.bin"

if not MODEL_XML.exists() or not MODEL_BIN.exists():
    raise FileNotFoundError(
        f"Model files not found:\n  {MODEL_XML}\n  {MODEL_BIN}\n"
        "Place model.xml and model.bin next to this script."
    )

print(f"[INFO] Loading OpenVINO model from {HERE}")
_ie = Core()
_model = _ie.read_model(model=str(MODEL_XML), weights=str(MODEL_BIN))
_compiled_model = _ie.compile_model(model=_model, device_name="CPU")
_input_layer = _compiled_model.input(0)
_output_layer = _compiled_model.output(0)
print("[OK] OpenVINO model compiled for CPU.")

def _ai_detect_pick_pixel(robot, image_source: str):
    """Capture a color image, run OpenVINO detection, return (image_proto, img_bgr, (x,y))."""
    image_client = robot.ensure_client(ImageClient.default_service_name)

    print(f"[INFO] Getting an image from: {image_source}")
    requests = [build_image_request(image_source, pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8)]
    image_responses = image_client.get_image(requests)
    if len(image_responses) != 1:
        raise RuntimeError(f"Invalid number of images: {len(image_responses)}")
    image = image_responses[0]

    # Decode to BGR (OpenCV)
    img = np.frombuffer(image.shot.image.data, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode image from Spot.")

    # Save raw capture
    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_path = OUTPUT_DIR / f"det_raw_{ts}.jpg"
    cv2.imwrite(str(raw_path), img)
    print(f"[INFO] Saved raw capture to: {raw_path}")

    # Inference (match ai_grasp’s preprocessing sizes)
    input_h, input_w = 800, 992
    resized = cv2.resize(img, (input_w, input_h))
    input_image = resized[np.newaxis, ...].astype(np.float32)

    results = _compiled_model([input_image])[_output_layer]
    if isinstance(results, dict):
        results = list(results.values())[0]
    arr = np.array(results)

    while arr.ndim > 2:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim == 1:
        reshaped = False
        for k in (7, 6, 5):
            if arr.size % k == 0:
                arr = arr.reshape(-1, k)
                reshaped = True
                break
        if not reshaped:
            raise RuntimeError(f"Unexpected flat detection vector of length {arr.size}")
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise RuntimeError(f"Unexpected detection tensor shape: {arr.shape}")

    # Postprocess: pick first detection >= 0.6 conf; compute center pixel in ORIGINAL image coords
    x_pick, y_pick = None, None
    for det in arr:
        x1, y1, x2, y2, conf = float(det[0]), float(det[1]), float(det[2]), float(det[3]), float(det[4])
        if conf < 0.6:
            continue
        x_scale = img.shape[1] / float(input_w)
        y_scale = img.shape[0] / float(input_h)
        x1i, x2i = int(x1 * x_scale), int(x2 * x_scale)
        y1i, y2i = int(y1 * y_scale), int(y2 * y_scale)
        cx = (x1i + x2i) // 2
        cy = (y1i + y2i) // 2

        cv2.rectangle(img, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(img, f"{conf:.2f}", (x1i, max(0, y1i - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        x_pick, y_pick = cx, cy
        break

    overlay_path = OUTPUT_DIR / f"det_overlay_{ts}.jpg"
    if x_pick is None:
        cv2.putText(img, "NO DETECTION >= 0.6", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(str(overlay_path), img)
    print(f"[INFO] Saved detection overlay to: {overlay_path}")

    if x_pick is None:
        raise RuntimeError("No detection above confidence threshold.")

    print(f"[INFO] Pick pixel selected by AI: ({x_pick}, {y_pick})")
    return image, img, (int(x_pick), int(y_pick))



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


def _move_hand_imaging_pose(
    command_client,
    *,
    yaw_rad: float = 0.0,
    pitch_rad: float = 1.3,
    x: float = 0.6,
    y: float = 0.0,
    z: float = 0.25,
    timeout_sec: float = 8.0,
) -> None:
    """Position the hand for imaging in GRAV_ALIGNED_BODY, with a given yaw jitter."""
    # Compose yaw * pitch so the gripper looks down with a slight yaw offset.
    q_yaw = math_helpers.Quat.from_yaw(yaw_rad)
    q_pitch = math_helpers.Quat.from_pitch(pitch_rad)
    q = q_yaw * q_pitch
    q_proto = q.to_proto()

    print(f"[INFO] Imaging pose -> yaw={yaw_rad:.2f} rad, pitch={pitch_rad:.2f} rad @ (x={x}, y={y}, z={z})")
    arm_pose_cmd = RobotCommandBuilder.arm_pose_command(
        x, y, z,
        q_proto.w, q_proto.x, q_proto.y, q_proto.z,
        GRAV_ALIGNED_BODY_FRAME_NAME,
        0.0,
    )
    pose_id = command_client.robot_command(arm_pose_cmd)
    block_until_arm_arrives(command_client, pose_id, timeout_sec=timeout_sec)


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
        body_Q_grasp = math_helpers.Quat.from_pitch(1.6)  # ~45°
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

def _fallback_step_back(
    robot,
    command_client,
    robot_state_client,
    *,
    distance_m: float = 0.5,   # <-- 0.5 m as requested
    seconds: float = 3.0,
):
    """Stow arm, then step straight backwards by distance_m relative to BODY, executed in ODOM."""
    try:
        print("[FALLBACK] Stowing arm…")
        stow_cmd = RobotCommandBuilder.arm_stow_command()
        stow_id = command_client.robot_command(stow_cmd)
        block_until_arm_arrives(command_client, stow_id)

        print(f"[FALLBACK] Stepping back {distance_m:.2f} m…")
        transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        # Negative X in BODY frame = step back
        body_tform_goal = math_helpers.SE2Pose(x=-abs(distance_m), y=0.0, angle=0.0)

        # Convert BODY->ODOM so we can command a stable goal
        odom_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        odom_tform_goal = odom_tform_body * body_tform_goal

        mobility_params = RobotCommandBuilder.mobility_params()
        traj_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=odom_tform_goal.x,
            goal_y=odom_tform_goal.y,
            goal_heading=odom_tform_goal.angle,
            frame_name=ODOM_FRAME_NAME,
            params=mobility_params,
        )
        cmd_id = command_client.robot_command(
            command=traj_cmd,
            end_time_secs=time.time() + max(seconds, 1.0),
        )

        # Wait for the trajectory to complete
        from bosdyn.client.robot_command import block_for_trajectory_cmd
        block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=seconds + 3.0)
        print("[FALLBACK] Step-back complete.")
    except Exception as e:
        print(f"[FALLBACK] Step-back failed: {e}")


def run(
    robot,
    *,
    image_source: str = "hand_color_image",
    force_top_down_grasp: bool = True,
    force_horizontal_grasp: bool = False,
    force_45_angle_grasp: bool = False,
    force_squeeze_grasp: bool = False,
    click_ui: bool = False,
    pixel_xy: Optional[Tuple[int, int]] = None,
    feedback_poll_s: float = 0.25,
    stow_after_grasp: bool = False,
    retries: int = 5,           # <-- add this
) -> bool:
    # --- safety / clients ---
    def _verify_not_estopped(robot_):
        estop_client = robot_.ensure_client(EstopClient.default_service_name)
        status = estop_client.get_status()
        if status.stop_level != estop_pb2.ESTOP_LEVEL_NONE:
            level_name = estop_pb2.EstopStopLevel.Name(status.stop_level)
            endpoints = [e.endpoint_name for e in status.endpoints]
            raise RuntimeError(
                f"Robot is estopped: stop_level={level_name}. "
                f"Active endpoints: {', '.join(endpoints) or 'none'}.\n"
                "Make sure your E-Stop is registered and in HOLD/RUNNING, then try again."
            )

    _verify_not_estopped(robot)

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    # # --- Arm ready + open gripper ---
    # print("[INFO] Unstowing arm (arm_ready)…")
    # unstow_cmd = RobotCommandBuilder.arm_ready_command()
    # unstow_id = command_client.robot_command(unstow_cmd)
    # block_until_arm_arrives(command_client, unstow_id, timeout_sec=5.0)
    # print("[OK] Arm ready.")

    print("[INFO] Opening gripper fully…")
    open_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
    command_client.robot_command(open_cmd)
    # time.sleep(0.7)

        # --- Position wrist for imaging (neutral yaw first) ---
    _move_hand_imaging_pose(command_client, yaw_rad=0.0, pitch_rad=1.0)
    time.sleep(0.2)

    # --- AI detection on color image, with random-yaw retries if nothing is found ---
    max_attempts = 5  # total attempts incl. the initial neutral yaw pose
    attempt = 1


    while True:
        try:
            print(f"[INFO] AI detection attempt {attempt}/{max_attempts} …")
            image, _img_bgr, target_px = _ai_detect_pick_pixel(robot, image_source)
            break  # success
        except RuntimeError as e:
            msg = str(e)
            if "No detection above confidence threshold" in msg and attempt < max_attempts:
                yaw = random.uniform(-0.7, 0.7)  # ~±40°
                print(f"[WARN] {msg}  -> Jitter yaw by {yaw:.2f} rad and try again.")
                _move_hand_imaging_pose(command_client, yaw_rad=yaw, pitch_rad=1.3)
                time.sleep(0.2)  # brief settle
                attempt += 1
                continue
            raise  # different error or maxed out attempts



    # --- Build grasp request at the AI-selected pixel ---
    print(f"[INFO] Sending grasp request at pixel {target_px} (source: '{image_source}')")
    pick_vec = geometry_pb2.Vec2(x=target_px[0], y=target_px[1])

    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=image.shot.transforms_snapshot,
        frame_name_image_sensor=image.shot.frame_name_image_sensor,
        camera_model=image.source.pinhole,
    )

    # Closer to palm for bricks. 0.0=palm, 1.0=fingertip.
    grasp.grasp_params.grasp_palm_to_fingertip = 0.2

    # --- Constraints (kept from your pick_brick flow) ---
    def _add_grasp_constraint(opts, grasp_, robot_state_client_):
        use_vector = opts.get("force_top_down_grasp") or opts.get("force_horizontal_grasp")
        grasp_.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME
        if use_vector:
            if opts.get("force_top_down_grasp"):
                axis_on_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
                axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=-1)
            else:
                axis_on_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)
                axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=1)
            c = grasp_.grasp_params.allowable_orientation.add()
            c.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper)
            c.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(axis_to_align)
            c.vector_alignment_with_tolerance.threshold_radians = 0.05
        elif opts.get("force_45_angle_grasp"):
            robot_state = robot_state_client_.get_robot_state()
            vision_T_body = frame_helpers.get_vision_tform_body(
                robot_state.kinematic_state.transforms_snapshot
            )
            body_Q_grasp = math_helpers.Quat.from_pitch(1.7)
            vision_Q_grasp = vision_T_body.rotation * body_Q_grasp
            c = grasp_.grasp_params.allowable_orientation.add()
            c.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())
            c.rotation_with_tolerance.threshold_radians = 0.10
        elif opts.get("force_squeeze_grasp"):
            c = grasp_.grasp_params.allowable_orientation.add()
            c.squeeze_grasp.SetInParent()

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

    # --- Feedback loop with "stuck in NO_SOLUTION" watchdog ---
    no_solution_count = 0
    started_ts = time.time()
    fallback_triggered = False  # track if we decide to bail and retry

    while True:
        fb_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd.manipulation_cmd_id
        )
        fb = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=fb_req
        )
        state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(fb.current_state)
        print(f"Current state: {state_name}")

        # Exit on explicit success/failure
        if fb.current_state in (
            manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
            manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
        ):
            break

        # Watchdog: if planner keeps reporting NO_SOLUTION, trigger fallback
        if fb.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION:
            no_solution_count += 1
            # trip if >10 consecutive polls OR >15 s in loop
            if no_solution_count >= 10 or (time.time() - started_ts) > 15.0:
                print("[WARN] Grasp planning stuck (NO_SOLUTION) — triggering fallback.")
                fallback_triggered = True
                break
        else:
            no_solution_count = 0  # reset if state changes

        time.sleep(feedback_poll_s)

    success = (not fallback_triggered) and (
        fb.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED
    )
    print(f"[RESULT] Finished grasp: {'SUCCESS' if success else 'FAILED or ABORTED FOR FALLBACK'}")

    # --- Stow with carry override (unchanged) ---
    if success and stow_after_grasp:
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

          # --- If we detected a stuck planner, back up and retry the entire pickup ---
    if fallback_triggered:
        print("[FALLBACK] Opening gripper (safety) …")
        try:
            open_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
            command_client.robot_command(open_cmd)
        except Exception:
            pass

        _fallback_step_back(robot, command_client, robot_state_client, distance_m=0.5, seconds=3.0)
        time.sleep(0.5)

        if retries > 0:
            print(f"[FALLBACK] Re-running pick (retries left: {retries}) …")
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
        else:
            print("[FALLBACK] No retries left — aborting pick.")
            return False

    # --- Position wrist in GRAV_ALIGNED_BODY and pitch ~45° down (post-pick pose) ---
    x, y, z = 0.6, 0.0, 0.25  # 60 cm forward, 25 cm up relative to body (gravity aligned)
    quat_down_45 = math_helpers.Quat.from_pitch(1.0).to_proto()  # ~45° pitch down
    arm_pose_cmd = RobotCommandBuilder.arm_pose_command(
        x, y, z,
        quat_down_45.w, quat_down_45.x, quat_down_45.y, quat_down_45.z,
        GRAV_ALIGNED_BODY_FRAME_NAME,
        0.0
    )
    pose_id = command_client.robot_command(arm_pose_cmd)
    block_until_arm_arrives(command_client, pose_id, timeout_sec=8.0)

    # --- Post-pose gripper check → fallback if empty (same style as your earlier example) ---
    try:
        rs = robot_state_client.get_robot_state()
        ms = rs.manipulator_state

        # Try both proto field names across releases.
        open_pct = getattr(ms, "gripper_open_percentage", None)
        if open_pct is None:
            open_pct = getattr(ms, "gripper_open_percent", None)
        holding = bool(getattr(ms, "is_gripper_holding_item", False))

        print(f"[CHECK] Post-pose gripper — open%: {('%.1f' % open_pct) if open_pct is not None else 'n/a'}, holding: {holding}")

        # Define “empty” as: fully/mostly closed OR not holding item
        empty_after_pose = ((open_pct is not None and open_pct <= 10.0) or (not holding))
    except Exception as e:
        print(f"[WARN] Could not read gripper state after pose: {e}")
        empty_after_pose = False  # don’t trigger fallback blindly if state is unavailable

    if empty_after_pose:
        print("[FALLBACK] Gripper appears empty after post-pick pose — backing up 0.5 m and retrying.")
        # Safety: ensure gripper is open before backing up / retrying
        try:
            command_client.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(1.0))
        except Exception:
            pass

        _fallback_step_back(robot, command_client, robot_state_client, distance_m=0.5, seconds=3.0)
        time.sleep(0.5)

        if retries > 0:
            print(f"[FALLBACK] Re-running pick (retries left: {retries}) …")
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
        else:
            print("[FALLBACK] No retries left — aborting pick.")
            return False


    # print("[INFO] Stowing arm…")
    # stow_cmd = RobotCommandBuilder.arm_stow_command()
    # stow_id = command_client.robot_command(stow_cmd)
    # block_until_arm_arrives(command_client, stow_id)



    return success

