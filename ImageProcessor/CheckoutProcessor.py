import cv2
from ImageProcessor.CheckoutAnalyzer.CheckoutAnalyzer import CheckoutAnalyzer

'''
    Class to process images from a camera 
    GetFrame method returns the regular stream
    GetAileView method returns the stream with person detections
'''

class CheckoutProcessor():
    def __init__(self, camera_index=1) -> None:
        self.cap = cv2.VideoCapture(0)
        self.checkout_analyzer = CheckoutAnalyzer()
        
        # while True:
        #     success, frame = self.cap.read()
        #     if not success:
        #         break
        #     frame = self.checkout_analyzer.get_person_detections(frame)
        
    def count_people(self):
        success, frame = self.cap.read()
        if not success:
            return None
        
        count = self.checkout_analyzer.count_people(frame)
        return count
