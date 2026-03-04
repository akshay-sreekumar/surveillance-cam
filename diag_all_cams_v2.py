import cv2
import os

os.makedirs("diag_photos", exist_ok=True)

for i in range(11):
    print(f"Testing index {i}...")
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Index {i} could not be opened.")
        continue
    
    # Try multiple times to get a frame (some cameras take time to wake up)
    ret = False
    for j in range(10):
        cap.grab()
        ret, frame = cap.read()
        if ret:
            break
            
    if ret:
        cv2.imwrite(f"diag_photos/cam_{i}.jpg", frame)
        print(f"Captured frame from index {i}")
    else:
        print(f"Could not read frame from index {i}")
        
    cap.release()
