#!/usr/bin/env python
"""
Manual Accuracy Test for Hand-Mouth Detection
Press 'n' when hand IS near mouth (ground truth = 1)
Press 'f' when hand IS NOT near mouth (ground truth = 0)
Press 'q' to quit
Results saved to CSV with accuracy metrics
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
    """Detect hand and mouth positions and check proximity"""
    global hand_landmarker, face_landmarker
    
    if hand_landmarker is None or face_landmarker is None:
        return False, None, None, None
    
    try:
        INDEX_TIP = 8
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        hand_result = hand_landmarker.detect(mp_image)
        face_result = face_landmarker.detect(mp_image)
        
        mouth_center = None
        hand_landmarks_list = None
        distance = None
        is_near = False
        
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
        
        if hand_result and hand_result.hand_landmarks and mouth_center:
            for hand_lms in hand_result.hand_landmarks:
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


# ==================== MANUAL ACCURACY TEST ====================

class ManualAccuracyTest:
    def __init__(self):
        self.results = []
        self.frame_times = []
        self.total_frames = 0
        self.ground_truth_labels = []
        self.system_predictions = []
        self.hand_detected_count = 0
        self.mouth_detected_count = 0
        
    def run(self, duration=None):
        """Run manual accuracy test"""
        
        print("\n" + "=" * 70)
        print("🎯 MANUAL ACCURACY TEST - Hand-Mouth Detection")
        print("=" * 70)
        print("📋 INSTRUCTIONS:")
        print("   - Put your hand near your mouth (simulate pill taking)")
        print("   - Move your hand away from your mouth")
        print("   - PRESS 'n' when hand IS NEAR mouth (ground truth = PILL TAKEN)")
        print("   - PRESS 'f' when hand IS NOT NEAR mouth (ground truth = NOT TAKEN)")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save screenshot")
        print("=" * 70)
        print("⏳ You need to label at least 50 frames for accuracy metrics")
        print("   'n' = Hand NEAR mouth (PILL TAKEN)")
        print("   'f' = Hand NOT near mouth (NOT TAKEN)")
        print("=" * 70)
        
        if not initialize_mediapipe():
            print("❌ Failed to initialize.")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        # Stats
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0
        labeled_count = 0
        near_count = 0
        away_count = 0
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.total_frames += 1
            
            # Process frame
            frame_start = time.time()
            is_near, hand_lms, mouth_center, distance = detect_hand_mouth(frame)
            elapsed_ms = (time.time() - frame_start) * 1000
            
            self.frame_times.append(elapsed_ms)
            
            # Record system prediction (0 = away, 1 = near)
            system_prediction = 1 if is_near else 0
            self.system_predictions.append(system_prediction)
            
            if hand_lms:
                self.hand_detected_count += 1
            if mouth_center:
                self.mouth_detected_count += 1
            
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
                if len(hand_lms) > 8:
                    tip = hand_lms[8]
                    cv2.circle(display_frame, tip, 8, (0, 255, 0), 2)
            
            # Draw mouth center
            if mouth_center:
                cv2.circle(display_frame, mouth_center, 8, (0, 0, 255), 2)
                cv2.circle(display_frame, mouth_center, 4, (0, 0, 255), -1)
            
            # Draw proximity line
            if hand_lms and mouth_center and distance is not None:
                tip = hand_lms[8]
                cv2.line(display_frame, tip, mouth_center, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Dist: {distance:.1f}px", 
                            (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # System status
            status_color = (0, 255, 0) if is_near else (0, 0, 255)
            status_text = "🟢 NEAR" if is_near else "🔴 AWAY"
            cv2.putText(display_frame, f"System: {status_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Last labeled ground truth
            if labeled_count > 0:
                last_gt = self.ground_truth_labels[-1]
                gt_text = "✅ PILL TAKEN" if last_gt == 1 else "❌ NOT TAKEN"
                gt_color = (0, 255, 0) if last_gt == 1 else (0, 0, 255)
                cv2.putText(display_frame, f"Last Label: {gt_text}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, gt_color, 2)
            
            cv2.putText(display_frame, f"Frames: {self.total_frames}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Labeled: {labeled_count}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Legend
            cv2.putText(display_frame, "n=NEAR(ground truth)  f=AWAY(ground truth)  q=quit  s=save", 
                       (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Manual Accuracy Test - Press 'n'(near) or 'f'(away)", display_frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n'):
                # Ground truth: Hand IS near mouth (PILL TAKEN)
                self.ground_truth_labels.append(1)
                labeled_count += 1
                near_count += 1
                # Save result with ground truth
                self.results.append({
                    'frame': self.total_frames,
                    'system_prediction': system_prediction,
                    'ground_truth': 1,
                    'is_correct': system_prediction == 1,
                    'distance': distance,
                    'latency_ms': elapsed_ms
                })
                print(f"📍 Frame {self.total_frames}: Labeled NEAR (pill taken) | System: {'✅' if system_prediction == 1 else '❌'}")
                
            elif key == ord('f'):
                # Ground truth: Hand IS NOT near mouth (NOT TAKEN)
                self.ground_truth_labels.append(0)
                labeled_count += 1
                away_count += 1
                self.results.append({
                    'frame': self.total_frames,
                    'system_prediction': system_prediction,
                    'ground_truth': 0,
                    'is_correct': system_prediction == 0,
                    'distance': distance,
                    'latency_ms': elapsed_ms
                })
                print(f"📍 Frame {self.total_frames}: Labeled AWAY (not taken) | System: {'✅' if system_prediction == 0 else '❌'}")
                
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"screenshot_{timestamp}.jpg", display_frame)
                print(f"📸 Screenshot saved: screenshot_{timestamp}.jpg")
                
            elif key == ord('q'):
                print("\n⏹️ Stopped by user")
                break
            
            # Check duration
            if duration and (time.time() - start_time) > duration:
                print(f"\n⏰ Time limit ({duration}s) reached")
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # ====== RESULTS ======
        self.print_results(labeled_count, near_count, away_count)
        self.save_results()
        
        return self.results
    
    def print_results(self, labeled_count, near_count, away_count):
        """Print accuracy metrics"""
        
        print("\n" + "=" * 70)
        print("📊 ACCURACY TEST RESULTS")
        print("=" * 70)
        
        if labeled_count == 0:
            print("⚠️ No frames were labeled. Please run the test and press 'n' or 'f'.")
            return
        
        # Calculate metrics
        correct = sum(1 for r in self.results if r['is_correct'])
        accuracy = correct / labeled_count * 100
        
        # Confusion Matrix
        tp = sum(1 for r in self.results if r['system_prediction'] == 1 and r['ground_truth'] == 1)
        tn = sum(1 for r in self.results if r['system_prediction'] == 0 and r['ground_truth'] == 0)
        fp = sum(1 for r in self.results if r['system_prediction'] == 1 and r['ground_truth'] == 0)
        fn = sum(1 for r in self.results if r['system_prediction'] == 0 and r['ground_truth'] == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n📈 METRICS:")
        print(f"   Total Frames: {self.total_frames}")
        print(f"   Labeled Frames: {labeled_count}")
        print(f"   Near (pill taken): {near_count}")
        print(f"   Away (not taken): {away_count}")
        print(f"   Correct Predictions: {correct}/{labeled_count}")
        
        print("\n🎯 ACCURACY:")
        print(f"   Accuracy: {accuracy:.1f}%")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   Specificity: {specificity:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        
        print("\n📊 CONFUSION MATRIX:")
        print(f"   True Positive (near → near):  {tp}")
        print(f"   True Negative (away → away):  {tn}")
        print(f"   False Positive (away → near): {fp}")
        print(f"   False Negative (near → away): {fn}")
        
        print("\n⏱️ SPEED:")
        avg_latency = np.mean(self.frame_times) if self.frame_times else 0
        print(f"   Avg Latency: {avg_latency:.1f} ms")
        print(f"   Estimated FPS: {1000/avg_latency:.1f}" if avg_latency > 0 else "   (No data)")
        
        # Interpretation
        print("\n💡 INTERPRETATION:")
        if accuracy >= 90:
            print("   ✅ Excellent! System is highly accurate.")
        elif accuracy >= 75:
            print("   👍 Good. System is reliable with room for improvement.")
        elif accuracy >= 60:
            print("   ⚠️ Moderate. Consider improving lighting or camera position.")
        else:
            print("   ❌ Needs improvement. Check detection thresholds and lighting.")
        
        print("=" * 70)
    
    def save_results(self):
        """Save results to CSV"""
        if not self.results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"accuracy_results_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'system_prediction', 'ground_truth', 'is_correct', 'distance', 'latency_ms'])
            for r in self.results:
                writer.writerow([
                    r['frame'],
                    r['system_prediction'],
                    r['ground_truth'],
                    r['is_correct'],
                    r['distance'],
                    r['latency_ms']
                ])
        
        print(f"\n💾 Detailed results saved to: {filename}")


# ==================== MAIN ====================

if __name__ == "__main__":
    test = ManualAccuracyTest()
    test.run(duration=None)  # Run until 'q' is pressed