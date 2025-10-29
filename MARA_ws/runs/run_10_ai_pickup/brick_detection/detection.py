from openvino.runtime import Core
import cv2
import numpy as np
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2

ie = Core()

model_xml = "68f8b78676caed31568f42a4-graph/68f8c88d4f1a2761daf71685-1761140082859/1/model.xml"        # path to your exported model
model_bin = "68f8b78676caed31568f42a4-graph/68f8c88d4f1a2761daf71685-1761140082859/1/model.bin"

model = ie.read_model(model=model_xml, weights=model_bin)
compiled_model = ie.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# ----------------------------------------------
# TODO: image rom gripper cam here (e.g. by build_image_request)
# Resize to model input shape (992x800)
input_height, input_width = 800, 992
resized = cv2.resize(image, (input_width, input_height))

# Convert to NCHW format
input_image = resized[np.newaxis, ...].astype(np.float32)

# ---- Run inference ----
results = compiled_model([input_image])[output_layer]

# ---- Postprocess (draw boxes) ----
for det in results[0]:
    if len(det) < 5:
        continue
    x1, y1, x2, y2, conf = det[:5]
    if conf < 0.75:
        continue

    # Scale coordinates back to original image size
    x_scale = image.shape[1] / input_width
    y_scale = image.shape[0] / input_height
    x1, x2 = int(x1 * x_scale), int(x2 * x_scale)
    y1, y2 = int(y1 * y_scale), int(y2 * y_scale)

    x_pick = int(0.5*(x1+x2))
    y_pick = int(0.5*(y1+y2))

    cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.circle(image, (x_pick,y_pick), 5, (0,0,255), -1)
    cv2.putText(image, f"{conf:.2f}", (x1, max(0, y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create Pick Vector

pick_vec = geometry_pb2.Vec2(x=x_pick, y=y_pick)
#...