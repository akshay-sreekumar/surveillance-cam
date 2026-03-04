import cv2
import time
import os
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO
import collections
import threading

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
SUPABASE_RECORDINGS_BUCKET = os.getenv("SUPABASE_RECORDINGS_BUCKET", "recordings")

RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)
RECORDING_DURATION_SECONDS = 60
MAX_RECORDINGS = 3

def upload_and_rotate_cloud_recording(camera_id, local_filepath):
    if not supabase_initialized:
        return
    try:
        filename = os.path.basename(local_filepath)
        cloud_path = f"{camera_id}/{filename}"
        
        print(f"[{camera_id}] Uploading recording {filename} to Supabase in background...")
        with open(local_filepath, "rb") as f:
            supabase.storage.from_(SUPABASE_RECORDINGS_BUCKET).upload(path=cloud_path, file=f, file_options={"content-type": "video/mp4"})
            
        # Get public URL for the video
        video_url = supabase.storage.from_(SUPABASE_RECORDINGS_BUCKET).get_public_url(cloud_path)
        
        # Save record to database
        recording_data = {
            "camera_id": camera_id,
            "video_url": video_url,
            "filename": filename,
            "created_at": datetime.now().isoformat()
        }
        supabase.table("recordings").insert(recording_data).execute()
        
        print(f"[{camera_id}] Upload complete and logged to DB: {cloud_path}")
        
        # We rely on Supabase listing or just upload the newest. For MVP, we upload newest.
        # Handling cloud deletion of oldest (circular queue)
        res = supabase.storage.from_(SUPABASE_RECORDINGS_BUCKET).list(camera_id)
        if res:
            # res is a list of dicts, sorted by name
            files = sorted(res, key=lambda x: x.get('name', ''))
            files = [f for f in files if f.get('name', '').endswith('.mp4')]
            
            while len(files) > MAX_RECORDINGS:
                oldest = files.pop(0)
                oldest_filename = oldest['name']
                oldest_path = f"{camera_id}/{oldest_filename}"
                print(f"[{camera_id}] Deleting oldest cloud recording: {oldest_path}")
                # Delete from storage
                supabase.storage.from_(SUPABASE_RECORDINGS_BUCKET).remove([oldest_path])
                # Delete from database
                supabase.table("recordings").delete().eq("filename", oldest_filename).eq("camera_id", camera_id).execute()
                
    except Exception as e:
        print(f"[{camera_id}] Failed to handle cloud recording rotation: {e}")

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
BLOOD_PIXEL_THRESHOLD = int(os.getenv("BLOOD_PIXEL_THRESHOLD", 500))

# Virtual boundary line: 0.0 = left edge, 1.0 = right edge of frame
# Crossing right→left  = ENTRY,  left→right = EXIT
BOUNDARY_X_RATIO = float(os.getenv("BOUNDARY_X_RATIO", 0.5))

# How often (seconds) to push people-count data to Supabase
COUNT_PUSH_INTERVAL = int(os.getenv("COUNT_PUSH_INTERVAL", 60))

# Buffer zone around boundary to prevent jitter (pixels)
# A person must fully cross the +/- buffer to be counted
BOUNDARY_BUFFER_PX = int(os.getenv("BOUNDARY_BUFFER_PX", 30))

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
print("Loading YOLO models...")
model = YOLO("yolov8n-pose.pt")
model_detect = YOLO("yolov8n.pt") # Standard model for weapon detecting (knife, bat, etc.)
WEAPON_CLASSES = [34, 43] # COCO indices: 34=baseball bat, 43=knife
print("YOLO models loaded.")

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
            "zone_name": camera_id.replace("_", " ").title(), # Use camera_id as zone name if not mapped
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
        try:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        except OSError as e:
            print(f"[{camera_id}] Could not delete temp file {temp_image_path} (it may still be locked): {e}")


def send_people_count(cam):
    """Insert current entry/exit counts into the 'people_count' Supabase table."""
    if not supabase_initialized:
        print(f"[{cam.camera_id}] People count (local): IN={cam.total_entered} OUT={cam.total_exited} INSIDE={cam.current_inside}")
        return
    try:
        data = {
            "camera_id": cam.camera_id,
            "entered": cam.total_entered,
            "exited": cam.total_exited,
            "current_inside": cam.current_inside,
            "recorded_at": datetime.now().isoformat()
        }
        supabase.table("people_count").insert(data).execute()
        print(f"[{cam.camera_id}] ✅ People count pushed to DB: IN={cam.total_entered} OUT={cam.total_exited} INSIDE={cam.current_inside}")
    except Exception as e:
        print(f"[{cam.camera_id}] ❌ Failed to push people count: {e}")


class CameraState:
    def __init__(self, index, cam_id):
        self.index = index
        self.camera_id = cam_id
        self.cap = cv2.VideoCapture(index)
        self.last_crowd_alert_time = 0
        self.last_violence_alert_time = 0
        self.last_fall_alert_time = 0
        self.last_weapon_alert_time = 0
        self.last_warning_time = 0
        self.first_fall_time = 0
        self.prev_gray = None
        self.active = self.cap.isOpened()
        
        # Video recording state
        # Not using maxlen here because we want to manually delete the file from disk when dropping
        self.recordings = collections.deque()
        self.current_writer = None
        self.recording_start_time = 0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0 or self.fps > 120:
            self.fps = 20.0  # Fallback FPS

        # --- Entry / Exit counter state ---
        # track_history: track_id -> list of recent X positions for smoothing
        self.track_history = {}
        # track_side: track_id -> last known "certain" side ('LEFT' or 'RIGHT')
        self.track_side = {}
        
        self.total_entered = 0    # cumulative people who entered
        self.total_exited  = 0    # cumulative people who exited
        self.current_inside = 0   # currently inside = entered - exited
        self.last_count_push_time = 0  # last time we pushed to DB
            
        if not self.active:
            print(f"Error: Cannot open camera at index {index} (ID: {cam_id})")

surveillance_stop_event = threading.Event()
surveillance_is_running = False

def run_surveillance(camera_configs=None, show_display=True):
    global surveillance_is_running
    surveillance_is_running = True
    surveillance_stop_event.clear()
    
    if camera_configs is None:
        camera_configs = list(zip(CAMERA_INDICES, CAMERA_IDS))
        
    cameras = []
    for idx, cam_id in camera_configs:
        cam_state = CameraState(idx, cam_id)
        if cam_state.active:
            cameras.append(cam_state)

    if not cameras:
        print("Error: No cameras could be opened.")
        surveillance_is_running = False
        return

    print(f"Started camera surveillance for {len(cameras)} cameras. Running in thread.")

    while not surveillance_stop_event.is_set():
        for cam in cameras:
            ret, frame = cam.cap.read()
            if not ret:
                continue

            current_time = time.time()
            
            # --- Continuous Video Recording Logic ---
            if cam.current_writer is None or (current_time - cam.recording_start_time >= RECORDING_DURATION_SECONDS):
                if cam.current_writer is not None:
                    cam.current_writer.release()
                    if len(cam.recordings) > 0:
                        finished_file = cam.recordings[-1]
                        # Start upload in background thread
                        threading.Thread(target=upload_and_rotate_cloud_recording, args=(cam.camera_id, finished_file)).start()
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = os.path.join(RECORDINGS_DIR, f"rec_{cam.camera_id}_{timestamp}.mp4")
                
                if len(cam.recordings) >= MAX_RECORDINGS:
                    oldest_file = cam.recordings.popleft()
                    if os.path.exists(oldest_file):
                        try:
                            os.remove(oldest_file)
                            print(f"[{cam.camera_id}] Deleted old local recording: {oldest_file}")
                        except OSError as e:
                            print(f"Error deleting file {oldest_file}: {e}")
                
                cam.recordings.append(filename)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                cam.current_writer = cv2.VideoWriter(filename, fourcc, cam.fps, (cam.frame_width, cam.frame_height))
                cam.recording_start_time = current_time
                print(f"[{cam.camera_id}] Started new local recording: {filename}")
                
            if cam.current_writer is not None:
                cam.current_writer.write(frame)
            # ----------------------------------------
            
            # Convert to grayscale for optical flow (violence detection)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Crowd Density Analysis + Entry/Exit Tracking
            # Use model.track() to get persistent IDs for crossing detection
            results = model.track(frame, verbose=False, classes=[0], persist=True)  # class 0 = person
            detections = results[0].boxes

            person_count = len(detections)
            crowd_confidence = 0.0

            if person_count > 0:
                crowd_confidence = float(sum(detections.conf).item() / person_count)

            # --- Virtual boundary line crossing detection (Robust State-Based) ---
            boundary_x = int(cam.frame_width * BOUNDARY_X_RATIO)
            left_limit = boundary_x - BOUNDARY_BUFFER_PX
            right_limit = boundary_x + BOUNDARY_BUFFER_PX
            
            active_track_ids = set()

            if detections.id is not None:  # track IDs available
                for box in detections:
                    track_id = int(box.id[0].item())
                    active_track_ids.add(track_id)

                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    raw_x = (x1 + x2) / 2.0  # centroid x

                    # 1. Update history and get smoothed X
                    if track_id not in cam.track_history:
                        cam.track_history[track_id] = []
                    cam.track_history[track_id].append(raw_x)
                    if len(cam.track_history[track_id]) > 3:
                        cam.track_history[track_id].pop(0)
                    
                    curr_x = sum(cam.track_history[track_id]) / len(cam.track_history[track_id])

                    # 2. Determine Initial Side if unknown
                    if track_id not in cam.track_side:
                        if curr_x < left_limit:
                            cam.track_side[track_id] = 'LEFT'
                        elif curr_x > right_limit:
                            cam.track_side[track_id] = 'RIGHT'
                        continue # Start tracking from a clear side

                    # 3. Check for transitions
                    orig_side = cam.track_side[track_id]

                    # RIGHT → LEFT crossing (ENTRY)
                    if orig_side == 'RIGHT' and curr_x < left_limit:
                        cam.total_entered += 1
                        cam.current_inside = max(0, cam.total_entered - cam.total_exited)
                        cam.track_side[track_id] = 'LEFT' # Update state to prevent re-trigger
                        print(f"[{cam.camera_id}] 🟢 ENTRY  id={track_id}  IN={cam.total_entered} INSIDE={cam.current_inside}")

                    # LEFT → RIGHT crossing (EXIT)
                    elif orig_side == 'LEFT' and curr_x > right_limit:
                        cam.total_exited += 1
                        cam.current_inside = max(0, cam.total_entered - cam.total_exited)
                        cam.track_side[track_id] = 'RIGHT'
                        print(f"[{cam.camera_id}] 🔴 EXIT   id={track_id}  OUT={cam.total_exited} INSIDE={cam.current_inside}")

            # Remove stale track IDs
            stale_ids = set(cam.track_history.keys()) - active_track_ids
            for stale_id in stale_ids:
                del cam.track_history[stale_id]
                if stale_id in cam.track_side:
                    del cam.track_side[stale_id]

            # Periodic push to Supabase every COUNT_PUSH_INTERVAL seconds
            if current_time - cam.last_count_push_time >= COUNT_PUSH_INTERVAL:
                threading.Thread(
                    target=send_people_count, args=(cam,), daemon=True
                ).start()
                cam.last_count_push_time = current_time

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

            # 3. Health Emergency / Fall Detection & Pose-based Violence Detection
            fall_confidence = 0.0
            fallen_count = 0
            pose_violence_score = 0.0
            
            # We need to analyze posture and bounding boxes
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                kpts = results[0].keypoints.data.cpu().numpy() # Shape (N, 17, 3)
                
                # Check bounding boxes for aspect ratio anomaly (wider than they are tall = slumped/fallen)
                for i, box in enumerate(detections):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Simple Heuristic: If bounding box width is significantly greater than height
                    if height > 0 and (width / height) > FALL_DETECTION_ASPECT_RATIO:
                        fallen_count += 1
                        # Base confidence on how extreme the slump is
                        confidence = min(((width / height) - 1.0) / 1.5, 1.0)
                        fall_confidence = max(fall_confidence, confidence)

                    # --- New Pose-based Violence Heuristics ---
                    if i < len(kpts):
                        person_kpts = kpts[i]
                        
                        # 1. Kicking: Ankle is above hip (Y goes DOWN in images, so 'above' means smaller Y)
                        # Left Ankle(15) vs Hip(11), Right Ankle(16) vs Hip(12)
                        for ankle_idx, hip_idx in [(15, 11), (16, 12)]:
                            if person_kpts[ankle_idx][2] > 0.5 and person_kpts[hip_idx][2] > 0.5:
                                if person_kpts[ankle_idx][1] < person_kpts[hip_idx][1]: 
                                    pose_violence_score += 0.5 # High suspicion of kicking

                        # 2. Instant Hand Raise / Punching: Wrist above nose
                        # Wrists(9,10), Nose(0)
                        for wrist_idx in [9, 10]:
                            if person_kpts[wrist_idx][2] > 0.5 and person_kpts[0][2] > 0.5:
                                if person_kpts[wrist_idx][1] < person_kpts[0][1]: 
                                    pose_violence_score += 0.3
                
                # 3. Fighting / Interpersonal Violence
                # Check bounding box overlaps and wrist-to-face proximity if >= 2 people
                if person_count >= 2:
                    for i in range(person_count):
                        for j in range(i + 1, person_count):
                            boxA = detections[i].xyxy[0]
                            boxB = detections[j].xyxy[0]
                            
                            # Simple overlap check
                            xA = max(boxA[0], boxB[0])
                            yA = max(boxA[1], boxB[1])
                            xB = min(boxA[2], boxB[2])
                            yB = min(boxA[3], boxB[3])
                            interArea = max(0, float((xB - xA) * (yB - yA)))
                            
                            if interArea > 0:
                                pose_violence_score += 0.2
                                # If wrist is touching/near another person's head
                                if i < len(kpts) and j < len(kpts):
                                    noseB = kpts[j][0]
                                    if noseB[2] > 0.5:
                                        for wrist_idx in [9, 10]:
                                            wristA = kpts[i][wrist_idx]
                                            if wristA[2] > 0.5:
                                                dist = np.linalg.norm([wristA[0]-noseB[0], wristA[1]-noseB[1]])
                                                # Threshold representing striking distance (pixels)
                                                if dist < 50:
                                                    pose_violence_score += 0.6
                                                    
            # Increase master violence confidence if pose heuristics are high
            violence_confidence = max(violence_confidence, min(pose_violence_score, 1.0))
            
            # --- Weapon Detection ---
            weapon_confidence = 0.0
            detect_results = model_detect(frame, verbose=False, classes=WEAPON_CLASSES)
            weapons = detect_results[0].boxes if len(detect_results) > 0 else []

            # --- Prolonged Fall and Blood Detection ---
            blood_detected = False
            blood_pixel_count = 0
            if fallen_count > 0:
                if cam.first_fall_time == 0:
                    cam.first_fall_time = current_time
                
                # Check for redness/blood in frame
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
                red_mask = mask1 + mask2
                blood_pixel_count = cv2.countNonZero(red_mask)
                if blood_pixel_count > BLOOD_PIXEL_THRESHOLD:
                    blood_detected = True
            else:
                cam.first_fall_time = 0

            # Draw info on frame
            display_frame = frame.copy()
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # If width > height * threshold, draw it Red to indicate a fall
                width, height = x2 - x1, y2 - y1
                color = (0, 0, 255) if height > 0 and (width / height) > FALL_DETECTION_ASPECT_RATIO else (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
            for w_box in weapons:
                conf = w_box.conf[0].item()
                if conf > 0.5:
                    weapon_confidence = max(weapon_confidence, conf)
                    wx1, wy1, wx2, wy2 = map(int, w_box.xyxy[0])
                    cv2.rectangle(display_frame, (wx1, wy1), (wx2, wy2), (0, 0, 255), 3)
                    cv2.putText(display_frame, "WEAPON DETECTED", (wx1, wy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            cv2.putText(display_frame, f"People: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Violence Score: {pose_violence_score:.2f} | Motion: {motion_magnitude:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Fallen: {fallen_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if fallen_count > 0 else (0, 255, 255), 2)

            # --- Draw virtual boundary line + entry/exit counters ---
            boundary_x_draw = int(cam.frame_width * BOUNDARY_X_RATIO)
            left_limit_draw = boundary_x_draw - BOUNDARY_BUFFER_PX
            right_limit_draw = boundary_x_draw + BOUNDARY_BUFFER_PX

            # Draw "Neutral Zone" shadow
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (left_limit_draw, 0), (right_limit_draw, cam.frame_height), (100, 100, 100), -1)
            cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)

            # Central Line
            cv2.line(display_frame, (boundary_x_draw, 0), (boundary_x_draw, cam.frame_height), (0, 255, 255), 2)
            
            # Entry label (left side of line)
            cv2.putText(display_frame, "< ENTRY", (boundary_x_draw - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Exit label (right side of line)
            cv2.putText(display_frame, "EXIT >", (boundary_x_draw + 20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Counter HUD at the bottom of the frame
            hud_y = cam.frame_height - 20
            cv2.rectangle(display_frame, (0, hud_y - 30), (420, cam.frame_height), (0, 0, 0), -1)
            cv2.putText(display_frame,
                        f"IN: {cam.total_entered}  OUT: {cam.total_exited}  INSIDE: {cam.current_inside}",
                        (10, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if show_display:
                cv2.imshow(f"Surveillance feed - {cam.camera_id}", display_frame)

            # 3. Decision Engine: Bridge and Database trigger
            
            # Decision 1: Overcrowding (Evaluated with its own cooldown)
            EXTREME_THRESHOLD = int(CROWD_DENSITY_THRESHOLD * 1.5)
            HIGH_THRESHOLD = CROWD_DENSITY_THRESHOLD
            MODERATE_THRESHOLD = int(CROWD_DENSITY_THRESHOLD * 0.7)

            if current_time - cam.last_crowd_alert_time >= COOLDOWN_SECONDS:
                # We do not use crowd_confidence strict threshold because YOLO confidence drops for dense, occluded crowds.
                if person_count >= EXTREME_THRESHOLD:
                    send_alert(cam.camera_id, "Extreme Crowd Density", display_frame, crowd_confidence, 
                               instruction="CRITICAL: Extreme overcrowding! Immediately halt entry and open emergency exits!")
                    cam.last_crowd_alert_time = current_time
                    cam.last_warning_time = current_time # Reset warning cooldown too
                elif person_count >= HIGH_THRESHOLD:
                    send_alert(cam.camera_id, "High Crowd Density", display_frame, crowd_confidence, 
                               instruction="High crowd density detected. Dispatch staff to manage flow and prepare overflow areas.")
                    cam.last_crowd_alert_time = current_time
                    cam.last_warning_time = current_time
                elif person_count >= MODERATE_THRESHOLD:
                    send_alert(cam.camera_id, "Moderate Crowd Density", display_frame, crowd_confidence, 
                               instruction="Crowd density is reaching moderate levels. Monitor the situation closely.")
                    cam.last_warning_time = current_time
            # Decision 2 & 4: Violence / High Erratic Motion / Weapon / Fighting
            # We use an OR logic to determine if violence is occurring.
            # Conditions for violence:
            # - Motion confidence is high (erratic motion)
            # - Pose violence score is high (kicking, punching, fighting)
            # - Weapon is detected
            
            is_violence_detected = False
            violence_flags = []
            max_violence_confidence = 0.0

            if violence_confidence >= DECISION_CONFIDENCE_THRESHOLD:
                 # Original check included person_count >= 2, we can keep or remove, but for erratic motion it helps reduce false positives
                 if person_count >= 2 or pose_violence_score >= 0.5:
                     is_violence_detected = True
                     violence_flags.append("Erratic Motion/Fighting")
                     max_violence_confidence = max(max_violence_confidence, violence_confidence)

            if weapon_confidence >= 0.5:
                 is_violence_detected = True
                 violence_flags.append("Weapon Detected")
                 max_violence_confidence = max(max_violence_confidence, weapon_confidence)
                 
            if pose_violence_score >= 0.5:
                 is_violence_detected = True
                 if "Erratic Motion/Fighting" not in violence_flags:
                      violence_flags.append("Violent Pose/Fighting")
                 max_violence_confidence = max(max_violence_confidence, pose_violence_score)

            if is_violence_detected and (current_time - cam.last_violence_alert_time >= COOLDOWN_SECONDS):
                alert_reason = " + ".join(violence_flags)
                send_alert(cam.camera_id, f"Violence ({alert_reason})", display_frame, max_violence_confidence, 
                           instruction="CRITICAL: Dispatch security personnel to camera location immediately.")
                cam.last_violence_alert_time = current_time
                # Also reset weapon alert time to avoid double alerting if weapon was part of violence
                cam.last_weapon_alert_time = current_time
                        
            # Decision 3: Health Emergency (Fall/Slump Detection)
            fall_duration = (current_time - cam.first_fall_time) if cam.first_fall_time > 0 else 0
            is_health_emergency = False
            health_flags = []
            
            if fall_duration >= 8.0:
                 is_health_emergency = True
                 health_flags.append(f"Prolonged Fall ({fall_duration:.1f}s)")
                 
            if blood_detected:
                 is_health_emergency = True
                 health_flags.append("Blood Detected")
                 
            if is_health_emergency and (current_time - cam.last_fall_alert_time >= COOLDOWN_SECONDS):
                 alert_reason = " + ".join(health_flags)
                 send_alert(cam.camera_id, f"SEVERE Health Emergency ({alert_reason})", display_frame, max(0.9, fall_confidence), 
                            instruction="CRITICAL: Dispatch medical personnel/first aid immediately. Possible severe injury or unconsciousness.")
                 cam.last_fall_alert_time = current_time
            elif (current_time - cam.last_fall_alert_time >= COOLDOWN_SECONDS) and fall_confidence >= DECISION_CONFIDENCE_THRESHOLD:
                 send_alert(cam.camera_id, "Health Emergency (Fall/Slump)", display_frame, fall_confidence, 
                            instruction="Dispatch medical personnel/first aid to camera location.")
                 cam.last_fall_alert_time = current_time

        if show_display:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                surveillance_stop_event.set()
                break
        else:
            # When headless, check if stop event is set to break out early if needed,
            # though the while loop condition handles normal stops.
            if surveillance_stop_event.is_set():
                break

    for cam in cameras:
        if hasattr(cam, 'current_writer') and cam.current_writer is not None:
            cam.current_writer.release()
            # Optionally upload the final partial clip
            if len(cam.recordings) > 0:
                finished_file = cam.recordings[-1]
                threading.Thread(target=upload_and_rotate_cloud_recording, args=(cam.camera_id, finished_file)).start()
        cam.cap.release()
    cv2.destroyAllWindows()

    surveillance_is_running = False
if __name__ == "__main__":
    run_surveillance(show_display=True)
