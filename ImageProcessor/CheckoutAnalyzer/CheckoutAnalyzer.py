import openvino as ov
import ipywidgets as widgets
from ultralytics import YOLO
import cv2
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import collections
import numpy as np
from openvino.runtime import Core, Tensor

'''
    Class to analyze the checkout area of a store
    Each cashier will have its own camera
    Methods:
        - Count people in the line
            Uses YOLO and OpenVino
        - Get sentiments of the person in the line
            Uses OpenVino 
            Uses emotions-recognition-retail-0003
            Uses face-detection-adas-0001
'''
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

        # Face detection model
        face_model_path = "models/face-detection-adas-0001/face-detection-adas-0001.xml"
        face_model = core.read_model(face_model_path)
        self.face_compiled_model = core.compile_model(face_model, "CPU")

        # Emotion recognition model
        emotion_model_path = "models/emotions-recognition-retail-0003/emotions-recognition-retail-0003.xml"
        emotion_model = core.read_model(emotion_model_path)
        self.emotion_compiled_model = core.compile_model(emotion_model, "CPU")

        def get_input_size(model):
            input_tensor = model.input(0)  
            input_shape = input_tensor.shape
            return (input_shape[3], input_shape[2])  

        self.face_input_size = get_input_size(face_model)
        self.emotion_input_size = get_input_size(emotion_model)

        if device.value != "CPU":
            det_ov_model.reshape({0: [1, 3, 640, 640]})
        if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
            ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        compiled_model = core.compile_model(det_ov_model, device.value, ov_config)

        def infer(*args):
            result = compiled_model(args)
            return torch.from_numpy(result[0])

        self.det_model.predictor.inference = infer
        self.det_model.predictor.model.pt = False
    
    def preprocess_face(self, image, input_size):
        resized_image = cv2.resize(image, (input_size[0], input_size[1]))
        resized_image = resized_image.astype(np.float32)  
        resized_image = resized_image.transpose((2, 0, 1)) 
        resized_image = resized_image[None, :] 
        return resized_image

    def preprocess_emotion(self, face_image, input_size):
        resized_face = cv2.resize(face_image, (input_size[0], input_size[1]))
        resized_face = resized_face.astype(np.float32)  
        resized_face = resized_face.transpose((2, 0, 1)) 
        resized_face = resized_face[None, :]  
        return resized_face
    
    def postprocess_emotion(self, output):
        emotions = ["neutral", "happiness", "surprise", "anger", "sadness"]
        emotion_scores = output[0] 
        probabilities = np.array(emotion_scores).flatten()
        top_indices = np.argsort(probabilities)[::-1][:2]
        top_emotions = [emotions[index] for index in top_indices]
        return top_emotions
    
    def postprocess_face(self, output, image, threshold=0.5):
        boxes = []
        for detection in output[0][0]:  
            if detection[2] > threshold:  
                xmin = int(detection[3] * image.shape[1])
                ymin = int(detection[4] * image.shape[0])
                xmax = int(detection[5] * image.shape[1])
                ymax = int(detection[6] * image.shape[0])
                boxes.append((xmin, ymin, xmax, ymax))
        return boxes
    
    def preprocess_face(self, image, input_size):
        resized_image = cv2.resize(image, (input_size[0], input_size[1]))
        resized_image = resized_image.astype(np.float32)  
        resized_image = resized_image.transpose((2, 0, 1)) 
        resized_image = resized_image[None, :]  
        return resized_image

    # Extract faces
    def postprocess_face(self, output, image, threshold=0.5):
        boxes = []
        for detection in output[0][0]:  
            if detection[2] > threshold:  
                xmin = int(detection[3] * image.shape[1])
                ymin = int(detection[4] * image.shape[0])
                xmax = int(detection[5] * image.shape[1])
                ymax = int(detection[6] * image.shape[0])
                boxes.append((xmin, ymin, xmax, ymax))
        return boxes
    
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
                cv2.imwrite("videos/lines.jpg", frame)

        return count
    
    def get_sentiments(self, image):
        preprocessed_image = self.preprocess_face(image, self.face_input_size)

        input_tensor = Tensor(preprocessed_image)
        face_infer_request = self.face_compiled_model.create_infer_request()
        face_infer_request.set_input_tensor(input_tensor)
        face_infer_request.infer()
        face_output = face_infer_request.get_output_tensor().data

        detected_faces = self.postprocess_face(face_output, image)
        original_image = image.copy()

        sentiments = []
        largest_face = None

        for (xmin, ymin, xmax, ymax) in detected_faces:
            face_region = original_image[ymin:ymax, xmin:xmax]

            if largest_face is None:
                largest_face = (xmin, ymin, xmax, ymax)
            else:
                if (xmax - xmin) * (ymax - ymin) > (largest_face[2] - largest_face[0]) * (largest_face[3] - largest_face[1]):
                    largest_face = (xmin, ymin, xmax, ymax)
            
        if largest_face is None:
            return sentiments
        xmin, ymin, xmax, ymax = largest_face
        face_region = original_image[ymin:ymax, xmin:xmax]
        preprocessed_face = self.preprocess_emotion(face_region, self.emotion_input_size)
        cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        emotion_tensor = Tensor(preprocessed_face)
        
        emotion_infer_request = self.emotion_compiled_model.create_infer_request()
        emotion_infer_request.set_input_tensor(emotion_tensor)
        emotion_infer_request.infer()
        emotion_output = emotion_infer_request.get_output_tensor().data
        
        emotions = self.postprocess_emotion(emotion_output)
        sentiments.append(emotions)
        emotions = " ".join(emotions)
        cv2.putText(original_image, emotions, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite("videos/sentiment.jpg", original_image)
        return sentiments
