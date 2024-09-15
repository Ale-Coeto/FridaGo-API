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
    Class to detect people in a frame using OpenVino and Yolov8
'''

colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0),   # Olive
    (255,255,255)   # White
]

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
            self.h, self.w = frame.shape[:2]  
            self.img = frame.copy()
            print(self.h, self.w)
        tracks = self.det_model.track(frame, persist=True, show=False, classes=[0], tracker='bytetrack.yaml', verbose=False)   

        for out in tracks:
            for box in out.boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                # track_id = out.track_id/
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                center_feet = (int((x1 + x2) / 2), y2)
                center_feet_flipped = (int((x1 + x2) / 2), self.h - y2)
                # print(center_feet)
                self.points.append(center_feet)
                #draw a point in the cente
                # cv2.circle(self.img, center_feet, 10, (0, 0, 255), -1)
                track_id = box.id.int().item()
                self.paths[track_id].append(center_feet)
                cv2.putText(frame, str(track_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame

    def get_common_paths(self):
        # print(self.paths)
        if len(self.paths) == 0:
            return self.img
        
        paths_list = list(self.paths.values())

        # skip every 10 points of each path
        for i, path in enumerate(paths_list):
            paths_list[i] = path[::10]

        # print(paths_list)
        
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
        possible_n_clusters = range(2, 11)  # Test for 2 to 10 clusters

        for n_clusters in possible_n_clusters:
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
            labels = spectral.fit_predict(distance_matrix)
            sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
            sil_scores.append(sil_score)

        optimal_n_clusters = possible_n_clusters[np.argmax(sil_scores)]
        print("Optimal number of clusters:", optimal_n_clusters)
            
        # Apply Spectral Clustering
        n_clusters = 3  # Define the number of clusters
        spectral = SpectralClustering(n_clusters=optimal_n_clusters, affinity='precomputed')
        labels = spectral.fit_predict(distance_matrix)

        # Draw a line connecting the points in each path but only for the first path in each cluster 
            # if labels[i] == 0:
            #     for j in range(len(path) - 1):
            #         cv2.line(self.img, path[j], path[j + 1], (0, 255, 0), 2)
        # print results
        copy_img = self.img
        overlay = np.full_like(copy_img, (255, 255, 255), dtype=np.uint8)  # White image with the same size as the original

        # Set the transparency level
        alpha = 0.5  # 0 is fully transparent, 1 is fully opaque
        copy_img = cv2.addWeighted(overlay, alpha, copy_img, 1 - alpha, 0)
 
        for path in paths_list:
            for i in range(len(path) - 2):
                if i % 10 == 0:
                    cv2.line(copy_img, path[i], path[i + 1], (0, 0, 0), 2)
                cv2.line(copy_img, path[i], path[i + 1], (0, 0, 0), 2)

        index = 0
        drawn = np.zeros(optimal_n_clusters)
        for i, path in enumerate(paths_list):
            # print(f"Path {i+1} is in cluster {labels[i]}")
            if drawn[labels[i]] == 0:
                for j in range(len(path) - 1):
                    cv2.line(copy_img, path[j], path[j + 1], colors[index], 3)
                drawn[labels[i]] = 1
                index += 1
                if index == len(colors):
                    index = 0



        cv2.imwrite("common_paths.jpg", copy_img)
        # Display the clusters for each path
        # for i, path in enumerate(self.paths):
        #     print(f"Path {i+1} is in cluster {labels[i]}")

        return copy_img


    def get_heatmap(self):
        # print("Headmat")
        # print(len(self.points))
        if self.img is None:
            return None
        
        image = self.img.copy()
        self.h, self.w = image.shape[:2] 

        if len(self.points) == 0 or self.w is None or self.h is None:
            return image
        
        data = np.array(self.points)

        # Define the gri`d size
        bin_size_factor = 100
        # grid_size = [self.h, self.w]
        adjusted_h = int(self.h / bin_size_factor)
        adjusted_w = int(self.w / bin_size_factor)

        # Create a 2D histogram
        # heatmap, xedges, yedges = np.histogram2d(
        #     data[:, 0], data[:, 1],
        #     bins=[adjusted_h, adjusted_w],
        #     range=[[0, self.w], [0, self.h]]
        # )

        heatmap, xedges, yedges = np.histogram2d(
            data[:, 1], data[:, 0],
            bins=[adjusted_h, adjusted_w],  # Set bins equal to image dimensions
            range=[[0, self.h], [0, self.w]]  # Ensure range covers the full image
        )

        print("hsize", heatmap.shape)
        print("imagesize", image.shape)
        heatmap_data = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap((heatmap_data * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Resize heatmap to match the original image size if necessary
        if heatmap_colored.shape[0] != image.shape[0] or heatmap_colored.shape[1] != image.shape[1]:
            print("No MATR")
            heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))

        # Blend heatmap with original image
        alpha = 0.5  # Transparency level of the heatmap
        overlay = cv2.addWeighted(heatmap_colored, alpha, image, 1 - alpha, 0)
        # heatmap = np.zeros((self.h, self.w), dtype=np.float32)

        # radius = 20  # Increase this to make the points larger
        # # Normalize the heatmap
        # heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)

        # # Convert heatmap to color map
        # heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # # Optionally overlay the heatmap onto the original image
        # overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

# Draw a circle for each point on the heatmap
        # for point in self.points:
        #     x, y = int(point[0]), int(point[1])
        #     cv2.circle(heatmap, (x, y), radius, 1, thickness=-1)
        #     # append additional points inside the radius if possible
        #     for i in range(-radius, radius):
        #         for j in range(-radius, radius):
        #             if x + i >= 0 and x + i < self.w and y + j >= 0 and y + j < self.h:
        #                 self.points.append((x + i, y + j))
        # for point in data:
        #     x, y = int(point[0]), int(point[1])
        #     cv2.circle(heatmap, (x, y), radius, 1, thickness=-1)  # Filled circle
        #     # append additional points inside the radius if possible
        #     for i in range(-radius, radius):
        #         for j in range(-radius, radius):
        #             if x + i >= 0 and x + i < self.w and y + j >= 0 and y + j < self.h:
        #                 cv2.circle(heatmap, (x + i, y + j), 1, 1, thickness=-1)


        

        # Save or show the result
        cv2.imwrite('image_with_heatmap_overlay.jpg', overlay)
        return overlay

        # Plot the heatmap
        # plt.figure(figsize=(8, 6))
        # plt.imshow(heatmap.T, origin='lower', cmap='YlGnBu', interpolation='nearest')
        # plt.colorbar(label='Frequency')
        # plt.xlabel('X coordinate')
        # plt.ylabel('Y coordinate')
        # plt.title('Heatmap of Most Visited Zones') 
        # plt.gca().invert_yaxis()
        # plt.savefig("heatmap.jpg")
        # print("done")

        # #return heatmap image as numpy array
        # return "done"


PersonDetector()