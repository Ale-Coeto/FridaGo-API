from ultralytics import YOLO
# from ultralytics.solutions import ObjectCounter
import cv2
# import time
# import collections
# import numpy as np
# from IPython import display
import torch
from pathlib import Path
import openvino as ov
import requests
# from notebook_utils import device_widget
models_dir = Path("./models")
models_dir.mkdir(exist_ok=True)

DET_MODEL_NAME = "yolov8n"

det_model = YOLO(models_dir / f"{DET_MODEL_NAME}.pt")
label_map = det_model.model.names

# Need to make en empty call to initialize the model
res = det_model()
det_model_path = models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
if not det_model_path.exists():
    det_model.export(format="openvino", dynamic=True, half=True)

def run_inference(source, device):
    core = ov.Core()

    det_ov_model = core.read_model(det_model_path)
    ov_config = {}

    if device.value != "CPU":
        det_ov_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    compiled_model = core.compile_model(det_ov_model, device.value, ov_config)

    def infer(*args):
        result = compiled_model(args)
        return torch.from_numpy(result[0])

    # Use openVINO as inference engine
    det_model.predictor.inference = infer
    det_model.predictor.model.pt = False

    frame = cv2.imread(source)
    tracks = det_model.track(frame, persist=True, show=False, classes=[0], verbose=False)   

    # print(tracks)
    boxes = tracks[0].boxes

    for out in tracks:
        for box in out.boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]

            class_id = box.cls[0].item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite("frame.jpg", frame)

def device_widget(default="AUTO", exclude=None, added=None):
    import openvino as ov
    import ipywidgets as widgets

    core = ov.Core()

    supported_devices = core.available_devices + ["AUTO"]
    exclude = exclude or []
    if exclude:
        for ex_device in exclude:
            if ex_device in supported_devices:
                supported_devices.remove(ex_device)

    added = added or []
    if added:
        for add_device in added:
            if add_device not in supported_devices:
                supported_devices.append(add_device)

    device = widgets.Dropdown(
        options=supported_devices,
        value=default,
        description="Device:",
        disabled=False,
    )
    return device


device = device_widget()

device

run_inference("b.jpg", device)