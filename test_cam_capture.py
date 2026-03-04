import cv2
import os

def test_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {index}")
        return False
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("test_camera_output.jpg", frame)
        print(f"Success: Frame captured from camera {index}")
    else:
        print(f"Error: Could not read frame from camera {index}")
    
    cap.release()
    return ret

if __name__ == "__main__":
    # Test index 3 as it's the one in .env and likely DroidCam
    test_camera(3)
