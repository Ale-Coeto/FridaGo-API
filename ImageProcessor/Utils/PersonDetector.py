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

'''
    Class to detect people in a frame using OpenVino and Yolov8
'''
    
class PersonDetector:

    def __init__(self) -> None:
        self.points = []
        self.paths = collections.defaultdict(list)
        
        models_dir = Path("./models")
        models_dir.mkdir(exist_ok=True)
        self.w = None
        self.h = None
        self.img = None


        DET_MODEL_NAME = "yolov8n"

        self.det_model = YOLO(models_dir / f"{DET_MODEL_NAME}.pt")
        label_map = self.det_model.model.names

        # Need to make en empty call to initialize the model
        res = self.det_model()
        det_model_path = models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
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


    
    def get_detections(self, frame):
        if self.w is None or self.h is None:
            self.h, self.w, _ = frame.shape
            self.img = frame
            print(self.h, self.w)
        tracks = self.det_model.track(frame, persist=True, show=False, classes=[0], tracker='bytetrack.yaml', verbose=False)   

        for out in tracks:
            for box in out.boxes:
                x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                # track_id = out.track_id/
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                center_feet = (int((x1 + x2) / 2), y2)
                self.points.append(center)
                track_id = box.id.int().item()
                self.paths[track_id].append(center_feet)
                cv2.putText(frame, str(track_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame

    def get_common_paths(self):
        # print(self.paths)
        paths_list = list(self.paths.values())

        # skip every 10 points of each path
        for i, path in enumerate(paths_list):
            paths_list[i] = path[::10]

        print(paths_list)
        
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

        # Apply Spectral Clustering
        n_clusters = 3  # Define the number of clusters
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        labels = spectral.fit_predict(distance_matrix)

        # Draw a line connecting the points in each path but only for the first path in each cluster 
            # if labels[i] == 0:
            #     for j in range(len(path) - 1):
            #         cv2.line(self.img, path[j], path[j + 1], (0, 255, 0), 2)
        # print results
        for i, path in enumerate(paths_list):
            print(f"Path {i+1} is in cluster {labels[i]}")
            if labels[i] == 0:
                print("OOOOOOOOOPath", i+1, "is in cluster 0")
                for j in range(len(path) - 1):
                    cv2.line(self.img, path[j], path[j + 1], (255, 0, 0), 2)

        for path in paths_list:
            for i in range(len(path) - 1):
                cv2.line(self.img, path[i], path[i + 1], (0, 255, 0), 2)

        cv2.imwrite("common_paths.jpg", self.img)
        # Display the clusters for each path
        # for i, path in enumerate(self.paths):
        #     print(f"Path {i+1} is in cluster {labels[i]}")


    def get_heatmap(self):
        print("Headmat")
        print(len(self.points))
        if len(self.points) == 0 or self.w is None or self.h is None:
            return None
        
        data = np.array(self.points)

        # Define the gri`d size
        bin_size_factor = 50
        grid_size = [self.h, self.w]
        adjusted_h = int(self.h / bin_size_factor)
        adjusted_w = int(self.w / bin_size_factor)

        # Create a 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            data[:, 0], data[:, 1],
            bins=[adjusted_w, adjusted_h],
            range=[[0, max(data[:, 0])], [0, max(data[:, 1])]]
        )

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, origin='lower', cmap='YlGnBu', interpolation='nearest')
        plt.colorbar(label='Frequency')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('Heatmap of Most Visited Zones') 
        plt.gca().invert_yaxis()
        plt.savefig("heatmap.jpg")
        print("done")

        #return heatmap image as numpy array
        return "done"


PersonDetector()