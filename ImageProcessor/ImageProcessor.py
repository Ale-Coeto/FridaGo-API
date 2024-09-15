import cv2
from ImageProcessor.Utils.PersonDetector import PersonDetector

'''
    Class to process images from a camera 
    GetFrame method returns the regular stream
    GetAileView method returns the stream with person detections
'''

class ImageProcessor():
    def __init__(self, camera_index=1) -> None:
        self.cap = cv2.VideoCapture("videos/x.mp4")
        self.person_detector = PersonDetector()
        
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
        
        frame = self.person_detector.get_detections(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return buffer.tobytes()  
    
    def get_heatmap(self):
        frame = self.person_detector.get_heatmap()
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return buffer.tobytes()
    
    def get_common_paths(self):
        image = self.person_detector.get_common_paths()
        ret, buffer = cv2.imencode('.jpg', image)
        if not ret:
            return None
        return buffer.tobytes() 
    
