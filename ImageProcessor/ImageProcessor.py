import cv2
from ImageProcessor.AisleAnalysis.AisleAnalyzer import AisleAnalyzer

'''
    Class to process images from a camera 
    GetFrame method returns the regular stream
    GetAileView method returns the stream with person detections
'''

class ImageProcessor():
    def __init__(self, camera_index=1) -> None:
        self.cap = cv2.VideoCapture("videos/x.mp4")
        self.aisle_analyzer = AisleAnalyzer()
        success, frame = self.cap.read()
        if not success:
            return None
        
        frame = self.aisle_analyzer.get_person_detections(frame)
        
    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return buffer.tobytes() 
    
    def get_aile_view(self):
        success, frame = self.cap.read()
        if not success:
            return None
        
        frame = self.aisle_analyzer.get_person_detections(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return buffer.tobytes()  
    
    def get_heatmap(self):
        frame = self.aisle_analyzer.get_heatmap()
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return buffer.tobytes()
    
    def get_common_paths(self):
        image = self.aisle_analyzer.get_common_paths()
        ret, buffer = cv2.imencode('.jpg', image)
        if not ret:
            return None
        return buffer.tobytes() 
    
