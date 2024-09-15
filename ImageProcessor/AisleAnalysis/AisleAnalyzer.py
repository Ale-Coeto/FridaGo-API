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
from sklearn.metrics import silhouette_score

'''
    Class to analyze differen aisles from a store
    Methods:
        - Person detector 
            Uses YOLO and OpenVino
        - Heat map of the people in the aisle
            Uses heatmap as an overlay
        - Trajectories of the people in the aisle
            Uses Spectral Clustering to group the trajectories
'''

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (255, 255, 255)]

class AisleAnalyzer:

    def __init__(self) -> None:
        self.paths = collections.defaultdict(list)
        models_dir = Path("./models")
        DET_MODEL_NAME = "yolov8n"
        
        models_dir.mkdir(exist_ok=True)
        self.points = []
        self.img = None
        self.w = None
        self.h = None

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


    
    def get_person_detections(self, frame):
        if self.w is None or self.h is None:
            self.h, self.w = frame.shape[:2]  
            self.img = frame.copy()
            print(self.h, self.w)
        tracks = self.det_model.track(frame, persist=True, show=False, classes=[0], tracker='bytetrack.yaml', verbose=False)   

        for out in tracks:
            for box in out.boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                center_feet = (int((x1 + x2) / 2), y2)
                self.points.append(center_feet)
                track_id = box.id.int().item()
                self.paths[track_id].append(center_feet)
                cv2.putText(frame, str(track_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame

    def get_common_paths(self):
        if len(self.paths) == 0:
            return self.img
        
        paths_list = list(self.paths.values())
        for i, path in enumerate(paths_list):
            paths_list[i] = path[::10]
        
        flattened_paths = [np.ravel(path) for path in paths_list]
        
        # Compute the DTW distance between every pair of paths
        n_paths = len(paths_list)
        distance_matrix = np.zeros((n_paths, n_paths))

        for i in range(n_paths):
            for j in range(n_paths):
                if i < j:
                    distance = dtw.distance(flattened_paths[i], flattened_paths[j])
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
        sil_scores = []
        possible_n_clusters = range(2, 11)  

        for n_clusters in possible_n_clusters:
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
            labels = spectral.fit_predict(distance_matrix)
            sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
            sil_scores.append(sil_score)

        optimal_n_clusters = possible_n_clusters[np.argmax(sil_scores)]
        
        # Apply Spectral Clustering
        spectral = SpectralClustering(n_clusters=optimal_n_clusters, affinity='precomputed')
        labels = spectral.fit_predict(distance_matrix)

        alpha = 0.5  
        copy_img = self.img
        overlay = np.full_like(copy_img, (255, 255, 255), dtype=np.uint8)  
        copy_img = cv2.addWeighted(overlay, alpha, copy_img, 1 - alpha, 0)

        # All paths
        for path in paths_list:
            for i in range(len(path) - 1):
                if i % 10 == 0:
                    cv2.line(copy_img, path[i], path[i + 1], (0, 0, 0), 2)

        # Most common paths
        index = 0
        drawn = np.zeros(optimal_n_clusters)
        for i, path in enumerate(paths_list):
            if drawn[labels[i]] == 0:
                for j in range(len(path) - 1):
                    cv2.line(copy_img, path[j], path[j + 1], colors[index], 3)
                drawn[labels[i]] = 1
                index += 1
                if index == len(colors):
                    index = 0

        return copy_img


    def get_heatmap(self):
        if self.img is None:
            return None
        
        image = self.img.copy()
        self.h, self.w = image.shape[:2] 

        if len(self.points) == 0 or self.w is None or self.h is None:
            return image
        
        data = np.array(self.points)

        bin_size_factor = 100
        adjusted_h = int(self.h / bin_size_factor)
        adjusted_w = int(self.w / bin_size_factor)

        heatmap, xedges, yedges = np.histogram2d(
            data[:, 1], data[:, 0],
            bins=[adjusted_h, adjusted_w],  
            range=[[0, self.h], [0, self.w]]  
        )

        heatmap_data = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap((heatmap_data * 255).astype(np.uint8), cv2.COLORMAP_JET)

        if heatmap_colored.shape[0] != image.shape[0] or heatmap_colored.shape[1] != image.shape[1]:
            print("No MATR")
            heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))

        alpha = 0.5  
        overlay = cv2.addWeighted(heatmap_colored, alpha, image, 1 - alpha, 0)

        return overlay