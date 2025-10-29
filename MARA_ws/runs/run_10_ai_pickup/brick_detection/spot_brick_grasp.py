# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
"""Use OpenVINO detection to pick a brick automatically."""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from openvino import Core  # modern import

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, RobotCommandBuilder, block_until_arm_arrives


g_image_click = None
g_image_display = None

# -------------------------------------------------------------------------
# Load OpenVINO model from same folder as this script
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path.cwd()
MODEL_XML = HERE / "model.xml"
MODEL_BIN = HERE / "model.bin"

if not MODEL_XML.exists() or not MODEL_BIN.exists():
    raise FileNotFoundError(
        f"Model files not found:\n  {MODEL_XML}\n  {MODEL_BIN}\n"
        "Place model.xml and model.bin next to this script."
    )

print(f"[INFO] Loading OpenVINO model from {HERE}")
ie = Core()
model = ie.read_model(model=str(MODEL_XML), weights=str(MODEL_BIN))
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
try:
    print(f"[INFO] Model input partial shape: {input_layer.get_partial_shape()}")
except Exception:
    pass

output_layer = compiled_model.output(0)
print("[OK] OpenVINO model compiled for CPU.")


def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = 'Robot is estopped. Please use an external E-Stop client, such as the' \
                        ' estop SDK example, to configure E-Stop.'
        robot.logger.error(error_message)
        raise Exception(error_message)


def arm_object_grasp(config):
    """A simple example of using the Boston Dynamics API to command Spot's arm."""

    # See hello_spot.py for an explanation of these lines.
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk('ArmObjectGraspClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), 'Robot requires an arm to run this example.'

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    verify_estop(robot)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)

    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info('Powering on robot... This may take a several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # RobotCommandBuilder for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        robot.logger.info('Commanding robot to stand...')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')

        # --- Unstow arm, open gripper, and pitch wrist down 45° before imaging ---
        print("[INFO] Unstowing arm, opening gripper, and pitching wrist down 45° before imaging…")

        # 1. Unstow arm
        cmd_arm_ready = RobotCommandBuilder.arm_ready_command()
        arm_cmd_id = command_client.robot_command(cmd_arm_ready)
        block_until_arm_arrives(command_client, arm_cmd_id, timeout_sec=8)

        # 2. Open gripper
        cmd_open = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
        command_client.robot_command(cmd_open)
        time.sleep(0.7)

        # 3. Move wrist to 45° pitch-down pose (relative to GRAV_ALIGNED_BODY)
        # Frame axes: +x forward, +y left, +z up (gravity-aligned body frame)
        x, y, z = 0.75, 0.0, 0.25  # hand 75 cm forward, 25 cm up
        quat_down_45 = math_helpers.Quat.from_pitch(0.785398).to_proto()  # -45° pitch (down)

        hand_in_body = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=x, y=y, z=z),
            rotation=quat_down_45,
        )

        arm_command = RobotCommandBuilder.arm_pose_command(
            hand_in_body.position.x, hand_in_body.position.y, hand_in_body.position.z,
            hand_in_body.rotation.w, hand_in_body.rotation.x, hand_in_body.rotation.y, hand_in_body.rotation.z,
            GRAV_ALIGNED_BODY_FRAME_NAME, 2.0  # 2s move in Body frame
        )

        cmd_id = command_client.robot_command(arm_command)
        block_until_arm_arrives(command_client, cmd_id, timeout_sec=8)

        print("[OK] Arm ready; gripper open; wrist pitched down 45° (Body frame).")




        # --- Capture image from Spot camera ---
        robot.logger.info('Getting an image from: %s', config.image_source)
        requests = [build_image_request(config.image_source, pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8)]
        image_responses = image_client.get_image(requests)
        if len(image_responses) != 1:
            raise RuntimeError(f"Invalid number of images: {len(image_responses)}")
        image = image_responses[0]

        # Decode
        img = np.frombuffer(image.shot.image.data, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode image from Spot")

        # --- Save raw capture immediately (always) ---
        ts = time.strftime("%Y%m%d_%H%M%S")
        raw_path = OUTPUT_DIR / f"det_raw_{ts}.jpg"
        cv2.imwrite(str(raw_path), img)
        print(f"[INFO] Saved raw capture to: {raw_path}")

        input_h, input_w = 800, 992
        resized = cv2.resize(img, (input_w, input_h))

        # NHWC (1, 800, 992, 3) float32 as in the developer’s example
        input_image = resized[np.newaxis, ...].astype(np.float32)

        results = compiled_model([input_image])[output_layer]

        # --- Normalize output to a 2D array [N, M], M >= 5 ---
        if isinstance(results, dict):
            results = list(results.values())[0]
        arr = np.array(results)

        print(f"[INFO] Model output shape: {arr.shape}, dtype: {arr.dtype}")

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

        # --- Postprocess detections (draw overlays) ---
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
            break  # first confident detection

        # --- Always save an overlay image (with box if any, or "NO DETECTION") ---
        overlay_path = OUTPUT_DIR / f"det_overlay_{ts}.jpg"
        if x_pick is None:
            cv2.putText(img, "NO DETECTION >= 0.6", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(str(overlay_path), img)
        print(f"[INFO] Saved detection overlay to: {overlay_path}")

        # (Optional) preview if GUI available
        try:
            cv2.imshow("Detections", img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
        except Exception:
            pass

        if x_pick is None:
            raise RuntimeError("No detection above confidence threshold.")


        # --- Show + Save image with detections ---
        save_path = HERE / "detection_result.jpg"
        cv2.imshow("Detections", img)
        cv2.imwrite(str(save_path), img)
        print(f"[INFO] Detection image saved to: {save_path}")
        cv2.waitKey(500)
        cv2.destroyAllWindows()


        print(f"[INFO] Pick pixel: ({x_pick}, {y_pick})")

        pick_vec = geometry_pb2.Vec2(x=x_pick, y=y_pick)


        # Build the proto
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole)

        # Optionally add a grasp constraint.  This lets you tell the robot you only want top-down grasps or side-on grasps.
        add_grasp_constraint(config, grasp, robot_state_client)

        # Ask the robot to pick up the object
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        # Send the request
        cmd_response = manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request)

        # Get feedback from the robot
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print(
                f'Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}'
            )

            if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                break

            time.sleep(0.25)

        robot.logger.info('Finished grasp.')
        time.sleep(4.0)

        robot.logger.info('Sitting down and turning off.')

        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), 'Robot power off failed.'
        robot.logger.info('Robot safely powered off.')


def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        # print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to grasp'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)


def add_grasp_constraint(config, grasp, robot_state_client):
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #
    # You can specify more than one if you want and they will be OR'ed together.

    # For these options, we'll use a vector alignment constraint.
    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if config.force_top_down_grasp:
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        if config.force_horizontal_grasp:
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.  That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo)

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif config.force_45_angle_grasp:
        # Demonstration of a RotationWithTolerance constraint.  This constraint allows you to
        # specify a full orientation you want the hand to be in, along with a threshold.
        #
        # You might want this feature when grasping an object with known geometry and you want to
        # make sure you grasp a specific part of it.
        #
        # Here, since we don't have anything in particular we want to grasp,  we'll specify an
        # orientation that will have the hand aligned with robot and rotated down 45 degrees as an
        # example.

        # First, get the robot's position in the world.
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

        # Rotation from the body to our desired grasp.
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

        # Turn into a proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())

        # We'll accept anything within +/- 10 degrees
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif config.force_squeeze_grasp:
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', help='Get image from source',
                        default='frontleft_fisheye_image')
    parser.add_argument('-t', '--force-top-down-grasp',
                        help='Force the robot to use a top-down grasp (vector_alignment demo)',
                        action='store_true')
    parser.add_argument('-f', '--force-horizontal-grasp',
                        help='Force the robot to use a horizontal grasp (vector_alignment demo)',
                        action='store_true')
    parser.add_argument(
        '-r', '--force-45-angle-grasp',
        help='Force the robot to use a 45 degree angled down grasp (rotation_with_tolerance demo)',
        action='store_true')
    parser.add_argument('-s', '--force-squeeze-grasp',
                        help='Force the robot to use a squeeze grasp', action='store_true')
    options = parser.parse_args()

    num = 0
    if options.force_top_down_grasp:
        num += 1
    if options.force_horizontal_grasp:
        num += 1
    if options.force_45_angle_grasp:
        num += 1
    if options.force_squeeze_grasp:
        num += 1

    if num > 1:
        print('Error: cannot force more than one type of grasp.  Choose only one.')
        sys.exit(1)

    try:
        arm_object_grasp(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)
