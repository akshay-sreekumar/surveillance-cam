import cv2
import os

os.makedirs("diag_photos", exist_ok=True)

def diag_cameras():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                path = f"diag_photos/cam_{i}.jpg"
                cv2.imwrite(path, frame)
                print(f"Captured cam_{i}.jpg")
            cap.release()
        else:
            print(f"Camera {i} not available")

diag_cameras()
