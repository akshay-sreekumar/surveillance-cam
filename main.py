import cv2
import time
import os
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO
import collections

# Try to import supabase, but gracefully handle if credentials are not yet set up
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("Warning: supabase is not installed. Alerts will only be printed locally.")

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "alerts")

raw_camera_indices = os.getenv("CAMERA_INDEX", "0").split(",")
CAMERA_INDICES = [int(i.strip()) if i.strip().isdigit() else i.strip() for i in raw_camera_indices]

raw_camera_ids = os.getenv("CAMERA_ID", "unknown_camera").split(",")
CAMERA_IDS = [i.strip() for i in raw_camera_ids]

# Ensure lists are of equal length if possible
if len(CAMERA_IDS) < len(CAMERA_INDICES):
    CAMERA_IDS.extend([f"camera_{i}" for i in range(len(CAMERA_IDS), len(CAMERA_INDICES))])

CROWD_DENSITY_THRESHOLD = int(os.getenv("CROWD_DENSITY_THRESHOLD", 50))
VIOLENCE_MOTION_THRESHOLD = float(os.getenv("VIOLENCE_MOTION_THRESHOLD", 15.0))
FALL_DETECTION_ASPECT_RATIO = float(os.getenv("FALL_DETECTION_ASPECT_RATIO", 1.2))
DECISION_CONFIDENCE_THRESHOLD = float(os.getenv("DECISION_CONFIDENCE_THRESHOLD", 0.85))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", 30))

# Initialize Supabase
supabase_initialized = False
if SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        supabase_initialized = True
        print("Supabase initialized successfully.")
    except Exception as e:
        print(f"Error initializing Supabase: {e}")
else:
    print("Supabase credentials not fully set in .env. Running in local mode.")

# Load YOLOv8 Pose model (Nano version for speed)
print("Loading YOLO Pose model...")
model = YOLO("yolov8n-pose.pt")
print("YOLO model loaded.")

class CrowdPredictor:
    """
    Uses simple linear regression on recent historical data to predict future crowd density.
    This acts as a basic 'previously trained' context tracker for the current session.
    """
    def __init__(self, history_size=150):
        self.history = collections.deque(maxlen=history_size)

    def add_data(self, current_time, count):
        self.history.append((current_time, count))

    def predict_future(self, future_seconds=60):
        if len(self.history) < 10:
            return None # Not enough data to confidently predict
            
        x = np.array([t for t, c in self.history])
        y = np.array([c for t, c in self.history])
        
        # Normalize x to start at 0 for numeric stability
        x = x - x[0]
        x_mean, y_mean = np.mean(x), np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean)**2)
        
        if denominator == 0:
            return y[-1]
            
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        future_x = x[-1] + future_seconds
        predicted_count = slope * future_x + intercept
        return max(0, predicted_count)

def send_alert(camera_id, alert_type, frame, confidence_score, instruction=""):
    """
    Handles sending the alert to Firebase (Storage + Firestore)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    temp_image_path = f"temp_alert_{camera_id}_{timestamp}.jpg"
    
    # Save the frame locally first
    cv2.imwrite(temp_image_path, frame)
    
    print(f"🚨 [EMERGENCY DETECTED] Camera: {camera_id} | Type: {alert_type} | Confidence: {confidence_score*100:.1f}% 🚨")
    
    if not supabase_initialized:
        print(f"Local Alert: Saved temporarily as {temp_image_path} (Supabase not configured)")
        # Since Supabase isn't configured, we just delete it so it doesn't clutter
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return
        
    try:
        # 1. Upload snapshot to Supabase Storage
        file_path = f"snapshot_{camera_id}_{timestamp}.jpg"
        supabase.storage.from_(SUPABASE_STORAGE_BUCKET).upload(path=file_path, file=temp_image_path, file_options={"content-type": "image/jpeg"})
        image_url = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).get_public_url(file_path)
        
        # Determine priority for dashboard display
        priority = "HIGH"
        if alert_type == "Violence":
            priority = "EMERGENCY"
        elif "Warning" in alert_type:
            priority = "WARNING"
            
        # 2. Create document in Supabase 'alerts' table
        alert_data = {
            "camera_id": camera_id,
            "type": alert_type,  # e.g., "Violence", "Crowd Density", or "Crowd Density Warning"
            "priority": priority,
            "image_url": image_url,
            "confidence": float(confidence_score),
            "instruction": instruction
        }
        supabase.table("alerts").insert(alert_data).execute()
        
        print(f"✅ Alert data uploaded to Supabase. URL: {image_url}")
        
    except Exception as e:
        print(f"❌ Failed to upload alert to Supabase: {e}")
    finally:
        # ALWAYS clean up local file if we wrote one
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


class CameraState:
    def __init__(self, index, cam_id):
        self.index = index
        self.camera_id = cam_id
        self.cap = cv2.VideoCapture(index)
        self.last_alert_time = 0
        self.last_warning_time = 0
        self.prev_gray = None
        self.crowd_predictor = CrowdPredictor(history_size=150)
        self.active = self.cap.isOpened()
        if not self.active:
            print(f"Error: Cannot open camera at index {index} (ID: {cam_id})")

def main():
    cameras = []
    for idx, cam_id in zip(CAMERA_INDICES, CAMERA_IDS):
        cam_state = CameraState(idx, cam_id)
        if cam_state.active:
            cameras.append(cam_state)

    if not cameras:
        print("Error: No cameras could be opened.")
        return

    print(f"Started camera surveillance for {len(cameras)} cameras. Press 'q' to quit.")

    while True:
        for cam in cameras:
            ret, frame = cam.cap.read()
            if not ret:
                continue

            current_time = time.time()
            
            # Convert to grayscale for optical flow (violence detection)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Crowd Density Analysis
            # Run YOLO inference
            results = model(frame, verbose=False, classes=[0]) # class 0 is person
            detections = results[0].boxes
            
            person_count = len(detections)
            crowd_confidence = 0.0
            
            if person_count > 0:
                # Average confidence of detected people
                crowd_confidence = float(sum(detections.conf).item() / person_count)
                
            cam.crowd_predictor.add_data(current_time, person_count)

            # 2. Violence Detection (Erratic Motion Heuristic via Optical Flow)
            motion_magnitude = 0.0
            violence_confidence = 0.0
            
            if cam.prev_gray is not None:
                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowFarneback(cam.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Simple heuristic: if the average motion magnitude is exceptionally high, flag as erratic
                # In a real scenario, you'd mask this to only check motion WITHIN person bounding boxes
                motion_magnitude = np.mean(mag)
                
                # Normalize it arbitrarily to a 0.0 - 1.0 confidence score for the decision engine
                # e.g., if threshold is 15.0, a magnitude of 15.0 => 0.85 confidence
                violence_confidence = min(motion_magnitude / (VIOLENCE_MOTION_THRESHOLD / DECISION_CONFIDENCE_THRESHOLD), 1.0)
                
            cam.prev_gray = gray

            # 3. Health Emergency / Fall Detection
            fall_confidence = 0.0
            fallen_count = 0
            
            # We need to analyze posture and bounding boxes
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                # Check bounding boxes for aspect ratio anomaly (wider than they are tall = slumped/fallen)
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Simple Heuristic: If bounding box width is significantly greater than height
                    if height > 0 and (width / height) > FALL_DETECTION_ASPECT_RATIO:
                        fallen_count += 1
                        # Base confidence on how extreme the slump is
                        confidence = min(((width / height) - 1.0) / 1.5, 1.0)
                        fall_confidence = max(fall_confidence, confidence)

            # Draw info on frame
            display_frame = frame.copy()
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # If width > height * threshold, draw it Red to indicate a fall
                width, height = x2 - x1, y2 - y1
                color = (0, 0, 255) if height > 0 and (width / height) > FALL_DETECTION_ASPECT_RATIO else (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
            cv2.putText(display_frame, f"People: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Motion: {motion_magnitude:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Fallen: {fallen_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if fallen_count > 0 else (0, 255, 255), 2)

            cv2.imshow(f"Surveillance feed - {cam.camera_id}", display_frame)

            # 3. Decision Engine: Bridge and Database trigger
            # Only evaluate if cooldown period has passed
            if current_time - cam.last_alert_time >= COOLDOWN_SECONDS:
                
                # Decision 1: Overcrowding
                if person_count > CROWD_DENSITY_THRESHOLD and crowd_confidence >= DECISION_CONFIDENCE_THRESHOLD:
                    send_alert(cam.camera_id, "Crowd Density", display_frame, crowd_confidence, 
                               instruction="Overcrowding detected. Dispatch staff to manage flow and open immediate exits.")
                    cam.last_alert_time = current_time
                    cam.last_warning_time = current_time # Reset warning cooldown too
                else:
                    # Decision 1.5: Early Warning for Overcrowding
                    if current_time - cam.last_warning_time >= COOLDOWN_SECONDS:
                        # Predict crowd size 60 seconds from now
                        predicted_count = cam.crowd_predictor.predict_future(future_seconds=60)
                        if predicted_count is not None and predicted_count > CROWD_DENSITY_THRESHOLD:
                            send_alert(cam.camera_id, "Crowd Density Warning", display_frame, 0.75, 
                                       instruction="Monitor area closely. Consider opening more queues/exits.")  # Moderate confidence
                            cam.last_warning_time = current_time
                    
                # Decision 2: Violence / High Erratic Motion
                if violence_confidence >= DECISION_CONFIDENCE_THRESHOLD:
                    # Add a sanity check that people are actually present before assuming violence
                    if person_count >= 2: 
                        send_alert(cam.camera_id, "Violence", display_frame, violence_confidence, 
                                   instruction="CRITICAL: Dispatch security personnel to camera location immediately.")
                        cam.last_alert_time = current_time
                        
                # Decision 3: Health Emergency (Fall/Slump Detection)
                elif fall_confidence >= DECISION_CONFIDENCE_THRESHOLD:
                    send_alert(cam.camera_id, "Health Emergency (Fall/Slump)", display_frame, fall_confidence, 
                               instruction="Dispatch medical personnel/first aid to camera location.")
                    cam.last_alert_time = current_time

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cam in cameras:
        cam.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
