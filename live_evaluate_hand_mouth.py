#!/usr/bin/env python
"""
Live Evaluation for Hand-Mouth Detection System
Run this script to test your detection in real-time with your webcam.
"""

import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

# ==================== MEDIAPIPE SETUP ====================

hand_landmarker = None
face_landmarker = None

def initialize_mediapipe():
    """Initialize MediaPipe models"""
    global hand_landmarker, face_landmarker
    
    try:
        # Download models if not present
        hand_model_path = "hand_landmarker.task"
        face_model_path = "face_landmarker.task"
        
        if not os.path.exists(hand_model_path):
            print("📥 Downloading hand model...")
            hand_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(hand_url, hand_model_path)
            print("✅ Hand model downloaded")
        
        if not os.path.exists(face_model_path):
            print("📥 Downloading face model...")
            face_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(face_url, face_model_path)
            print("✅ Face model downloaded")
        
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model_path),
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=face_model_path),
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        
        print("✅ MediaPipe initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing MediaPipe: {e}")
        return False


def detect_hand_mouth(frame):
    """
    Detect hand and mouth positions and check proximity.
    Returns: (is_near_mouth, hand_landmarks_list, mouth_center, distance)
    """
    global hand_landmarker, face_landmarker
    
    if hand_landmarker is None or face_landmarker is None:
        return False, None, None, None
    
    try:
        INDEX_TIP = 8
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        hand_result = hand_landmarker.detect(mp_image)
        
        # Detect face
        face_result = face_landmarker.detect(mp_image)
        
        mouth_center = None
        hand_landmarks_list = None
        distance = None
        is_near = False
        
        # Get mouth center
        if face_result and face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]
            
            left_corner = face_landmarks[78]
            right_corner = face_landmarks[308]
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]
            
            mouth_center = (
                int(((left_corner.x + right_corner.x) / 2) * w),
                int(((upper_lip.y + lower_lip.y) / 2) * h)
            )
        
        # Check hand proximity to mouth
        if hand_result and hand_result.hand_landmarks and mouth_center:
            for hand_lms in hand_result.hand_landmarks:
                # Store landmarks for drawing
                hand_landmarks_list = []
                for lm in hand_lms:
                    hand_landmarks_list.append((int(lm.x * w), int(lm.y * h)))
                
                index_tip = hand_lms[INDEX_TIP]
                finger_pos = (int(index_tip.x * w), int(index_tip.y * h))
                
                distance = np.sqrt((finger_pos[0] - mouth_center[0])**2 + 
                                  (finger_pos[1] - mouth_center[1])**2)
                
                threshold = w * 0.08
                is_near = distance < threshold
                
                if is_near:
                    break
        
        return is_near, hand_landmarks_list, mouth_center, distance
        
    except Exception as e:
        print(f"Detection error: {e}")
        return False, None, None, None


# ==================== LIVE EVALUATOR ====================

class LiveEvaluator:
    def __init__(self):
        self.results = []
        self.frame_times = []
        self.detection_times = []
        self.hand_detected_count = 0
        self.mouth_detected_count = 0
        self.near_count = 0
        self.total_frames = 0
        
    def run(self, duration=None, save_video=False, show_landmarks=True):
        """
        Run live evaluation.
        duration: seconds to run (None = run until 'q' pressed)
        save_video: save recording to file
        show_landmarks: overlay landmarks on display
        """
        print("\n" + "=" * 60)
        print("🎥 LIVE EVALUATION - Hand-Mouth Detection")
        print("=" * 60)
        print("📋 Instructions:")
        print("   - Put your hand near your mouth (simulate pill taking)")
        print("   - Move your hand away from your mouth")
        print("   - Try different angles and lighting")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save a screenshot")
        print("=" * 60)
        
        if not initialize_mediapipe():
            print("❌ Failed to initialize. Please check your setup.")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        # Video writer
        out = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = cv2.VideoWriter(f'evaluation_{timestamp}.avi', fourcc, 20.0, (640, 480))
            print(f"📹 Recording to: evaluation_{timestamp}.avi")
        
        # FPS counter
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0  # 👈 INITIALIZE HERE
        
        # Detection stats
        hand_found = 0
        mouth_found = 0
        near_found = 0
        total = 0
        distances = []
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            total += 1
            
            # Process frame
            frame_start = time.time()
            is_near, hand_lms, mouth_center, distance = detect_hand_mouth(frame)
            elapsed_ms = (time.time() - frame_start) * 1000
            
            self.frame_times.append(elapsed_ms)
            self.detection_times.append(elapsed_ms)
            
            # Record results
            if hand_lms:
                hand_found += 1
            if mouth_center:
                mouth_found += 1
            if is_near:
                near_found += 1
                distances.append(distance)
            
            # FPS
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Draw overlay
            display_frame = frame.copy()
            
            # Draw hand landmarks
            if hand_lms:
                for i, (x, y) in enumerate(hand_lms):
                    cv2.circle(display_frame, (x, y), 3, (0, 255, 0), -1)
                # Draw finger tip with larger circle
                if len(hand_lms) > 8:
                    tip = hand_lms[8]
                    cv2.circle(display_frame, tip, 8, (0, 255, 0), 2)
            
            # Draw mouth center
            if mouth_center:
                cv2.circle(display_frame, mouth_center, 8, (0, 0, 255), 2)
                cv2.circle(display_frame, mouth_center, 4, (0, 0, 255), -1)
            
            # Draw proximity line
            if hand_lms and mouth_center:
                tip = hand_lms[8]
                cv2.line(display_frame, tip, mouth_center, (255, 255, 0), 2)
                if distance:
                    cv2.putText(display_frame, f"Dist: {distance:.1f}px", 
                                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Status text
            status_color = (0, 255, 0) if is_near else (0, 0, 255)
            status_text = "✅ HAND NEAR MOUTH" if is_near else "❌ HAND AWAY"
            if not hand_lms or not mouth_center:
                status_color = (255, 255, 0)
                status_text = "⏳ DETECTING..."
                if not hand_lms:
                    status_text += " (No hand)"
                if not mouth_center:
                    status_text += " (No face)"
            
            cv2.putText(display_frame, f"Status: {status_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_frame, f"Frames: {total}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Time: {time.time() - start_time:.1f}s", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Controls
            cv2.putText(display_frame, "q=quit  s=screenshot", (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Live Evaluation - Hand-Mouth Detection", display_frame)
            
            if save_video and out:
                out.write(display_frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⏹️ Stopped by user")
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"screenshot_{timestamp}.jpg", display_frame)
                print(f"📸 Screenshot saved: screenshot_{timestamp}.jpg")
            
            # Check duration
            if duration and (time.time() - start_time) > duration:
                print(f"\n⏰ Time limit ({duration}s) reached")
                break
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # ====== RESULTS ======
        self.total_frames = total
        self.hand_detected_count = hand_found
        self.mouth_detected_count = mouth_found
        self.near_count = near_found
        
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS")
        print("=" * 60)
        print(f"\n📈 SPEED:")
        avg_latency = np.mean(self.frame_times) if self.frame_times else 0
        print(f"   Average Latency: {avg_latency:.1f} ms")
        print(f"   Estimated FPS: {1000/avg_latency:.1f}" if avg_latency > 0 else "   (No data)")
        
        print(f"\n🎯 DETECTION:")
        if total > 0:
            hand_rate = hand_found / total * 100
            mouth_rate = mouth_found / total * 100
            near_rate = near_found / total * 100
            print(f"   Hand Detection: {hand_rate:.1f}% ({hand_found}/{total})")
            print(f"   Mouth Detection: {mouth_rate:.1f}% ({mouth_found}/{total})")
            print(f"   Hand Near Mouth: {near_rate:.1f}% ({near_found}/{total})")
        else:
            print("   No frames processed.")
        
        if distances:
            print(f"\n📏 DISTANCE METRICS:")
            print(f"   Average Distance: {np.mean(distances):.1f} pixels")
            print(f"   Min Distance: {np.min(distances):.1f} pixels")
            print(f"   Max Distance: {np.max(distances):.1f} pixels")
        
        # Confusion matrix
        print(f"\n📊 PROXIMITY CLASSIFICATION:")
        print("   (Based on hand-near-mouth detection)")
        print(f"   Positive (near): {near_found}")
        print(f"   Negative (away): {total - near_found}")
        
        # Save results
        self.save_results()
        
        return self.results

    def save_results(self):
        """Save results to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.csv"
        
        # Simple summary
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Frames', self.total_frames])
            writer.writerow(['Hand Detected', self.hand_detected_count])
            writer.writerow(['Mouth Detected', self.mouth_detected_count])
            writer.writerow(['Hand Near Mouth', self.near_count])
            if self.frame_times:
                writer.writerow(['Avg Latency (ms)', np.mean(self.frame_times)])
                writer.writerow(['Min Latency (ms)', np.min(self.frame_times)])
                writer.writerow(['Max Latency (ms)', np.max(self.frame_times)])
        
        print(f"\n💾 Results saved to: {filename}")


# ==================== MAIN ====================

if __name__ == "__main__":
    evaluator = LiveEvaluator()
    
    # Run evaluation (run indefinitely, press 'q' to quit)
    evaluator.run(
        duration=None,      # None = run until 'q' pressed
        save_video=False,   # Set to True to record
        show_landmarks=True
    )