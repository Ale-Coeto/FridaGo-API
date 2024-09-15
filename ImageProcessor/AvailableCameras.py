import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        # Use the camera for your application
        # ...
        cap.release()
    else:
        print(f"Camera {i} is not available")