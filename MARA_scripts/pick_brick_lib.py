# pick_brick.py  — library-style callable from sequence.py

import time
import cv2
import numpy as np

import bosdyn.client
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.estop import EstopClient
from bosdyn.api import estop_pb2
from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
    block_until_arm_arrives,
)

from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
    block_until_arm_arrives,
)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client import frame_helpers
from bosdyn.api import geometry_pb2
import time

# ---- UI globals (reused from the tutorial) ----
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
        image_title = 'Click to grasp'
        h, w = clone.shape[0], clone.shape[1]
        cv2.line(clone, (0, y), (w, y), color, thickness)
        cv2.line(clone, (x, 0), (x, h), color, thickness)
        cv2.imshow(image_title, clone)

def _verify_not_estopped(robot):
    """Check that Spot is not currently estopped."""
    estop_client = robot.ensure_client(EstopClient.default_service_name)
    status = estop_client.get_status()

    # Compare against the enum in estop_pb2, not the status object itself
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
            axis_on_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
            axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=-1)
        else:  # horizontal
            axis_on_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)
            axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=1)

        c = grasp.grasp_params.allowable_orientation.add()
        c.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper)
        c.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(axis_to_align)
        c.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif opts.get("force_45_angle_grasp"):
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp
        c = grasp.grasp_params.allowable_orientation.add()
        c.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())
        c.rotation_with_tolerance.threshold_radians = 0.17

    elif opts.get("force_squeeze_grasp"):
        c = grasp.grasp_params.allowable_orientation.add()
        c.squeeze_grasp.SetInParent()

def _get_image(robot, image_source):
    image_client = robot.ensure_client(ImageClient.default_service_name)
    responses = image_client.get_image_from_sources([image_source])
    if len(responses) != 1:
        raise RuntimeError(f"Expected 1 image, got {len(responses)}")
    image = responses[0]
    dtype = np.uint16 if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16 else np.uint8
    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(image.shot.image.rows, image.shot.image.cols)
    else:
        img = cv2.imdecode(img, -1)
    return image, img

def run(
    robot,
    *,
    image_source: str = "frontleft_fisheye_image",
    force_top_down_grasp: bool = False,
    force_horizontal_grasp: bool = False,
    force_45_angle_grasp: bool = False,
    force_squeeze_grasp: bool = False,
    click_ui: bool = True,
    pixel_xy: tuple[int, int] | None = None,
    feedback_poll_s: float = 0.25,
    stow_after_grasp: bool = True, 
) -> bool:
    """Pick an object by image click or provided pixel on an already-authenticated, standing robot.

    Preconditions (handled by sequence.py):
      - time sync OK, LeaseKeepAlive active, powered on, standing.
    """
    _verify_not_estopped(robot)

    # Ensure clients (NO new SDK/lease/power here)
    robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    # Acquire image & optional click
    image, img = _get_image(robot, image_source)
    target_px = pixel_xy
    if click_ui and target_px is None:
        robot.logger.info("Click on an object to start grasping… (press Q to abort)")
        global g_image_click, g_image_display
        g_image_click = None
        g_image_display = img
        title = "Click to grasp"
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, cv_mouse_callback)
        cv2.imshow(title, g_image_display)
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                robot.logger.info("User canceled grasp.")
                cv2.destroyAllWindows()
                return False
        target_px = (int(g_image_click[0]), int(g_image_click[1]))
        cv2.destroyAllWindows()

    if target_px is None:
        raise ValueError("No target pixel provided and click_ui=False.")

    robot.logger.info(f"Picking at image pixel {target_px} from source '{image_source}'")
    pick_vec = geometry_pb2.Vec2(x=target_px[0], y=target_px[1])

    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=image.shot.transforms_snapshot,
        frame_name_image_sensor=image.shot.frame_name_image_sensor,
        camera_model=image.source.pinhole,
    )

    _add_grasp_constraint(
        {
            "force_top_down_grasp": force_top_down_grasp,
            "force_horizontal_grasp": force_horizontal_grasp,
            "force_45_angle_grasp": force_45_angle_grasp,
            "force_squeeze_grasp": force_squeeze_grasp,
        },
        grasp,
        robot.ensure_client(RobotStateClient.default_service_name),
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


    if success:
        robot.logger.info("Grasp succeeded → up 30 cm → back 20 cm → stow.")
        from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, block_until_arm_arrives
        from bosdyn.client.frame_helpers import BODY_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b

        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

        # --- 1) Move UP +0.30 m in BODY z ---
        robot_state = robot_state_client.get_robot_state()
        transforms = robot_state.kinematic_state.transforms_snapshot
        body_T_hand = get_a_tform_b(transforms, BODY_FRAME_NAME, HAND_FRAME_NAME)

        target_up_x = body_T_hand.x
        target_up_y = body_T_hand.y
        target_up_z = body_T_hand.z + 0.70  # +70 cm up

        arm_up_cmd = RobotCommandBuilder.arm_pose_command(
            target_up_x, target_up_y, target_up_z,
            body_T_hand.rot.w, body_T_hand.rot.x, body_T_hand.rot.y, body_T_hand.rot.z,
            BODY_FRAME_NAME, 2
        )
        cmd_id = command_client.robot_command(RobotCommandBuilder.build_synchro_command(arm_up_cmd))
        block_until_arm_arrives(command_client, cmd_id)

        # --- 2) Move BACK −0.20 m in BODY x (keep the new z) ---
        robot_state = robot_state_client.get_robot_state()  # refresh after the lift
        transforms = robot_state.kinematic_state.transforms_snapshot
        body_T_hand = get_a_tform_b(transforms, BODY_FRAME_NAME, HAND_FRAME_NAME)

        target_back_x = body_T_hand.x - 0.20  # −20 cm back
        target_back_y = body_T_hand.y
        target_back_z = body_T_hand.z         # keep current height

        arm_back_cmd = RobotCommandBuilder.arm_pose_command(
            target_back_x, target_back_y, target_back_z,
            body_T_hand.rot.w, body_T_hand.rot.x, body_T_hand.rot.y, body_T_hand.rot.z,
            BODY_FRAME_NAME, 1
        )
        cmd_id = command_client.robot_command(RobotCommandBuilder.build_synchro_command(arm_back_cmd))
        block_until_arm_arrives(command_client, cmd_id)




    return success
