import openvino as ov
import ipywidgets as widgets
from ultralytics import YOLO
import cv2
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.cluster import SpectralClustering
import numpy as np
from dtaidistance import dtw

class CheckoutAnalyzer:

    def __init__(self) -> None:
        self.paths = collections.defaultdict(list)
        models_dir = Path("./models")
        DET_MODEL_NAME = "yolov8n"        
        models_dir.mkdir(exist_ok=True)

        self.det_model = YOLO(models_dir / f"{DET_MODEL_NAME}.pt")
        det_model_path = models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
        label_map = self.det_model.model.names
        self.det_model()

        if not det_model_path.exists():
            self.det_model.export(format="openvino", dynamic=True, half=True)

        def device_widget(default="AUTO", exclude=None, added=None):
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

        core = ov.Core()
        device = device_widget()
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
        self.det_model.predictor.inference = infer
        self.det_model.predictor.model.pt = False

    def count_people(self, frame):
        count = 0
        frame_center_x = int(frame.shape[1] / 2)
        buffer = 80
        left = frame_center_x - buffer
        right = frame_center_x +  buffer
        tracks = self.det_model.track(frame, persist=True, show=False, classes=[0], tracker='bytetrack.yaml', verbose=False)   

        cv2.line(frame, (int(frame_center_x), 0), (int(frame_center_x), frame.shape[0]), (255, 0, 0), 2)
        cv2.line(frame, (left, 0), (left, frame.shape[0]), (255, 0, 0), 2)
        cv2.line(frame, (right, 0), (right, frame.shape[0]), (255, 0, 0), 2)

        for out in tracks:
            for box in out.boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]

                if ((x1 <= frame_center_x and x2 >= frame_center_x) or (x2 >= left  and x2 <= right) or (x1 >= left and x1 <= right)):
                    count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    

                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imwrite("videos/frame.jpg", frame)

        return count