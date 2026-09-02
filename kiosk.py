import cv2
import time
import numpy as np
from datetime import datetime
import database
import vision

def run_kiosk(camera_index: int = 0, tolerance: float = 0.50, cooldown_mins: int = 5, process_every_n_frames: int = 3):
    """
    Runs a real-time continuous OpenCV roll-call kiosk application.
    
    Features:
        - 30+ FPS live webcam streaming with frame-skipping recognition.
        - Automatic attendance logging directly to SQLite with cooldown.
        - Real-time on-screen HUD with attendance status, FPS, and detection boxes.
        - Hotkey 'r' to hot-reload newly enrolled workers from DB.
        - Hotkey 'q' or 'ESC' to exit.
    """
    database.init_db()
    
    print("=" * 60)
    print("  RollCall AI - Continuous Desktop Kiosk Mode")
    print("=" * 60)
    print("  Controls:")
    print("    [Q] or [ESC] : Quit Kiosk")
    print("    [R]          : Reload enrolled workers from Database")
    print("=" * 60)
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}. Please check connection.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Cache known workers
    known_workers = database.get_all_workers()
    print(f"Loaded {len(known_workers)} enrolled worker(s) from database.")
    
    frame_count = 0
    detections = []
    recent_logs = []  # Stores (message, timestamp, is_success) for on-screen notification feed
    fps = 0.0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break
                
            frame_count += 1
            h, w, _ = frame.shape
            
            # Convert BGR (OpenCV) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run face recognition on every N-th frame for optimal performance
            if frame_count % process_every_n_frames == 0:
                # Downscale 0.5x for faster recognition
                small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
                
                raw_detections, _ = vision.recognize_faces(
                    small_frame, 
                    known_workers=known_workers, 
                    tolerance=tolerance
                )
                
                # Scale face locations back to original frame size
                detections = []
                for det in raw_detections:
                    top, right, bottom, left = det["location"]
                    scaled_loc = (top * 2, right * 2, bottom * 2, left * 2)
                    det_copy = dict(det)
                    det_copy["location"] = scaled_loc
                    detections.append(det_copy)
                    
                    # If recognized, log attendance
                    if det["is_match"] and det["worker_id"] is not None:
                        logged, msg = database.log_attendance(
                            worker_id=det["worker_id"],
                            cooldown_minutes=cooldown_mins
                        )
                        time_now = datetime.now().strftime("%H:%M:%S")
                        if logged:
                            notice = f"[{time_now}] {det['name']} marked present"
                            recent_logs.append((notice, time.time(), True))
                        else:
                            notice = f"[{time_now}] {det['name']} (cooldown active)"
                            recent_logs.append((notice, time.time(), False))
                            
                # Keep only recent notifications from last 6 seconds
                recent_logs = [log for log in recent_logs if time.time() - log[1] < 6.0]

            # Calculate FPS
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = 10.0 / elapsed if elapsed > 0 else 0.0
                start_time = time.time()

            # Render detection boxes on the BGR frame
            for det in detections:
                top, right, bottom, left = det["location"]
                is_match = det["is_match"]
                name = det["name"]
                dist = det["distance"]
                
                # Color in BGR: Emerald Green (50, 205, 50) / Coral Red (50, 50, 220)
                box_color = (50, 205, 50) if is_match else (50, 50, 220)
                
                # Bounding box
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                
                # Name Badge
                label = f"{name} [{dist:.2f}]" if is_match else "Unknown"
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
                
                label_y1 = max(top - th - 12, 0)
                label_y2 = max(top, th + 12)
                cv2.rectangle(frame, (left, label_y1), (left + tw + 12, label_y2), box_color, cv2.FILLED)
                cv2.putText(frame, label, (left + 6, label_y2 - 6), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

            # --- HUD OVERLAYS ---
            # Top Banner
            cv2.rectangle(frame, (0, 0), (w, 50), (20, 24, 30), cv2.FILLED)
            cv2.putText(frame, "RollCall AI - Kiosk Feed", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f} | Enrolled: {len(known_workers)} | Press 'Q' to Exit | 'R' to Reload", (w - 550, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Bottom Notification Feed
            y_offset = h - 25
            for msg, _, is_success in reversed(recent_logs[-4:]):
                bg_color = (30, 120, 40) if is_success else (40, 80, 140)
                cv2.rectangle(frame, (10, y_offset - 20), (min(500, w - 20), y_offset + 5), bg_color, cv2.FILLED)
                cv2.putText(frame, msg, (15, y_offset - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                y_offset -= 30

            cv2.imshow("RollCall AI - Face Recognition Kiosk", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('r'):  # Reload workers
                known_workers = database.get_all_workers()
                print(f"[RELOAD] Updated known workers: {len(known_workers)} enrolled.")
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Kiosk closed successfully.")

if __name__ == "__main__":
    run_kiosk()
