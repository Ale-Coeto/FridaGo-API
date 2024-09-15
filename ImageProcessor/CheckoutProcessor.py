import cv2
from ImageProcessor.CheckoutAnalyzer.CheckoutAnalyzer import CheckoutAnalyzer

'''
    Class to process images from a camera 
    Count people returns the number of people in the line
    GetSentiments returns the sentiments of the person with largest detection
'''

class CheckoutProcessor():
    def __init__(self, camera_index=1) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.checkout_analyzer = CheckoutAnalyzer()
        
    def count_people(self):
        success, frame = self.cap.read()
        if not success:
            return None
        
        count = self.checkout_analyzer.count_people(frame)
        return count

    def get_sentiments(self):
        success, frame = self.cap.read()
        if not success:
            return None
        
        sentiments = self.checkout_analyzer.get_sentiments(frame)
        return sentiments