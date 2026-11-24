#!/usr/bin/env python
"""
Manual Accuracy Test - 5 Labels (AUTO-FILL + METRICS)
Press keys to label frames:
  h = hand only (no mouth)
  m = mouth only (no hand)
  b = both hand + mouth, NO pill intake (hand AWAY from mouth)
  p = pill intake (hand NEAR mouth)
  n = nothing visible
  q = quit

AUTO-FILLS:
  - system_label = what MediaPipe detected
  - correct = T if user_label == system_label, F otherwise
  - correct_label = system_label if wrong, empty if correct

CALCULATES:
  - Accuracy, Precision, Recall, F1 per label
  - Confusion Matrix
  - Hand/Mouth Detection Rates
  - Distance Stats (avg, min, max)
  - Latency Stats (avg, min, max)
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

# ==================== SETTINGS ====================
SCREENSHOT_FOLDER = "eval_screenshots"

if not os.path.exists(SCREENSHOT_FOLDER):
    os.makedirs(SCREENSHOT_FOLDER)
    print(f"Created folder: {SCREENSHOT_FOLDER}")

# ==================== LABEL MAPPING ====================
LABEL_MAP = {
    ord('h'): 'hand_only',
    ord('m'): 'mouth_only',
    ord('b'): 'both_no_pill',
    ord('p'): 'pill_intake',
    ord('n'): 'nothing'
}

LABELS = ['hand_only', 'mouth_only', 'both_no_pill', 'pill_intake', 'nothing']

# ==================== MEDIAPIPE SETUP ====================

hand_landmarker = None
face_landmarker = None

def initialize_mediapipe():
    global hand_landmarker, face_landmarker
    try:
        hand_model_path = "hand_landmarker.task"
        face_model_path = "face_landmarker.task"
        
        if not os.path.exists(hand_model_path):
            print("Downloading hand model...")
            hand_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(hand_url, hand_model_path)
            print("Hand model downloaded")
        
        if not os.path.exists(face_model_path):
            print("Downloading face model...")
            face_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(face_url, face_model_path)
            print("Face model downloaded")
        
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
        
        print("MediaPipe initialized")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def detect_hand_mouth(frame):
    """Returns: system_label, hand_detected, mouth_detected, distance, mouth_center"""
    global hand_landmarker, face_landmarker
    if hand_landmarker is None or face_landmarker is None:
        return 'nothing', False, False, None, None
    
    try:
        INDEX_TIP = 8
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        hand_result = hand_landmarker.detect(mp_image)
        face_result = face_landmarker.detect(mp_image)
        
        hand_detected = hand_result is not None and hand_result.hand_landmarks
        mouth_detected = face_result is not None and face_result.face_landmarks
        
        mouth_center = None
        distance = None
        is_near = False
        
        if mouth_detected:
            face_landmarks = face_result.face_landmarks[0]
            left_corner = face_landmarks[78]
            right_corner = face_landmarks[308]
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]
            mouth_center = (
                int(((left_corner.x + right_corner.x) / 2) * w),
                int(((upper_lip.y + lower_lip.y) / 2) * h)
            )
        
        if hand_detected and mouth_center:
            for hand_lms in hand_result.hand_landmarks:
                index_tip = hand_lms[INDEX_TIP]
                finger_pos = (int(index_tip.x * w), int(index_tip.y * h))
                distance = np.sqrt((finger_pos[0] - mouth_center[0])**2 + 
                                  (finger_pos[1] - mouth_center[1])**2)
                threshold = w * 0.08
                is_near = distance < threshold
                break
        
        # ===== SYSTEM LABEL =====
        if hand_detected and mouth_detected and is_near:
            system_label = 'pill_intake'
        elif hand_detected and mouth_detected and not is_near:
            system_label = 'both_no_pill'
        elif hand_detected and not mouth_detected:
            system_label = 'hand_only'
        elif mouth_detected and not hand_detected:
            system_label = 'mouth_only'
        else:
            system_label = 'nothing'
        
        return system_label, hand_detected, mouth_detected, distance, mouth_center
        
    except Exception as e:
        print(f"Error: {e}")
        return 'nothing', False, False, None, None


# ==================== MAIN TEST CLASS ====================

class ManualAccuracyTest:
    def __init__(self):
        self.results = []
        self.total_frames = 0
        self.frame_times = []
        self.hand_detected_count = 0
        self.mouth_detected_count = 0
        
    def run(self):
        print("\n" + "=" * 70)
        print("MANUAL ACCURACY TEST - 5 Labels (AUTO-FILL)")
        print("=" * 70)
        print("INSTRUCTIONS:")
        print("   h = HAND only (no mouth)")
        print("   m = MOUTH only (no hand)")
        print("   b = BOTH hand + mouth, NO pill intake (hand AWAY)")
        print("   p = PILL INTAKE (hand NEAR mouth)")
        print("   n = NOTHING visible")
        print("   q = QUIT")
        print("=" * 70)
        print("AUTO-FILL: system_label, correct, correct_label")
        print("CALCULATES: Accuracy, Precision, Recall, F1, Confusion Matrix")
        print("CALCULATES: Detection Rates, Distance Stats, Latency")
        print(f"Screenshots saved to: {SCREENSHOT_FOLDER}/")
        print("=" * 70)
        
        if not initialize_mediapipe():
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam")
            return
        
        labeled_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.total_frames += 1
            
            # Detect
            frame_start = time.time()
            system_label, hand_detected, mouth_detected, distance, mouth_center = detect_hand_mouth(frame)
            elapsed_ms = (time.time() - frame_start) * 1000
            self.frame_times.append(elapsed_ms)
            
            if hand_detected:
                self.hand_detected_count += 1
            if mouth_detected:
                self.mouth_detected_count += 1
            
            # Draw overlay
            display_frame = frame.copy()
            
            if hand_detected:
                cv2.putText(display_frame, "HAND", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if mouth_detected and mouth_center:
                cv2.circle(display_frame, mouth_center, 8, (0, 0, 255), 2)
                cv2.putText(display_frame, "MOUTH", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            if distance is not None:
                cv2.putText(display_frame, f"Dist: {distance:.1f}px", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            system_color = {
                'pill_intake': (0, 255, 0),
                'both_no_pill': (255, 255, 0),
                'hand_only': (0, 165, 255),
                'mouth_only': (255, 0, 0),
                'nothing': (128, 128, 128)
            }.get(system_label, (255, 255, 255))
            
            cv2.putText(display_frame, f"System: {system_label}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, system_color, 2)
            
            cv2.putText(display_frame, f"Labeled: {labeled_count}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(display_frame, "h=hand  m=mouth  b=both(no pill)  p=pill  n=none  q=quit", 
                       (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Accuracy Test - Press h, m, b, p, or n", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key in LABEL_MAP:
                user_label = LABEL_MAP[key]
                labeled_count += 1
                
                # ===== AUTO-FILL =====
                if user_label == system_label:
                    correct = 'T'
                    correct_label = ''
                else:
                    correct = 'F'
                    correct_label = system_label
                
                # Save result
                self.results.append({
                    'frame': self.total_frames,
                    'label': user_label,
                    'system_label': system_label,
                    'hand_detected': hand_detected,
                    'mouth_detected': mouth_detected,
                    'distance': distance,
                    'correct': correct,
                    'correct_label': correct_label
                })
                
                # Save screenshot
                timestamp = datetime.now().strftime("%H%M%S")
                screenshot_name = f"eval_{user_label}_{labeled_count}_{timestamp}.jpg"
                screenshot_path = os.path.join(SCREENSHOT_FOLDER, screenshot_name)
                cv2.imwrite(screenshot_path, display_frame)
                
                print(f"{labeled_count}: {user_label} → System: {system_label} → {correct}")
                
            elif key == ord('q'):
                print("\nStopped")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # ===== SAVE RESULTS =====
        self.save_results()
        
        # ===== CALCULATE METRICS =====
        self.calculate_metrics()
    
    def save_results(self):
        if not self.results:
            return
        
        filename = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'label', 'system_label', 'hand_detected', 'mouth_detected', 'distance', 'correct', 'correct_label'])
            for r in self.results:
                writer.writerow([
                    r['frame'],
                    r['label'],
                    r['system_label'],
                    r['hand_detected'],
                    r['mouth_detected'],
                    r['distance'],
                    r['correct'],
                    r['correct_label']
                ])
        
        print(f"\nCSV saved: {filename}")
        print(f"   Total labeled: {len(self.results)}")
        print(f"   Screenshots saved in: {SCREENSHOT_FOLDER}/")
    
    def calculate_metrics(self):
        """Calculate ALL metrics from results"""
        if not self.results:
            print("No results to calculate")
            return
        
        total = len(self.results)
        
        # ===== CONFUSION MATRIX =====
        confusion = {l: {l2: 0 for l2 in LABELS} for l in LABELS}
        
        for r in self.results:
            user = r['label']
            system = r['system_label']
            confusion[user][system] = confusion[user].get(system, 0) + 1
        
        # ===== ACCURACY =====
        correct_count = sum(1 for r in self.results if r['correct'] == 'T')
        accuracy = correct_count / total * 100
        
        # ===== PER-LABEL METRICS =====
        metrics = {}
        for label in LABELS:
            tp = confusion[label][label]
            fp = sum(confusion[l][label] for l in LABELS if l != label)
            fn = sum(confusion[label][l] for l in LABELS if l != label)
            tn = total - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[label] = {
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1
            }
        
        # ===== DETECTION RATES =====
        hand_rate = self.hand_detected_count / self.total_frames * 100
        mouth_rate = self.mouth_detected_count / self.total_frames * 100
        
        # ===== DISTANCE STATS =====
        distances = [r['distance'] for r in self.results if r['distance'] is not None]
        avg_distance = sum(distances) / len(distances) if distances else 0
        min_distance = min(distances) if distances else 0
        max_distance = max(distances) if distances else 0
        
        # ===== LATENCY =====
        avg_latency = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        min_latency = min(self.frame_times) if self.frame_times else 0
        max_latency = max(self.frame_times) if self.frame_times else 0
        
        # ===== PRINT =====
        print("\n" + "=" * 70)
        print("ACCURACY TEST RESULTS")
        print("=" * 70)
        
        print(f"\nOVERALL METRICS:")
        print(f"   Total Frames: {self.total_frames}")
        print(f"   Labeled Frames: {total}")
        print(f"   Correct Predictions: {correct_count}/{total}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        print(f"\nPER-LABEL METRICS:")
        for label in LABELS:
            m = metrics[label]
            print(f"\n   {label}:")
            print(f"      Precision: {m['precision']:.3f}")
            print(f"      Recall:    {m['recall']:.3f}")
            print(f"      F1-Score:  {m['f1']:.3f}")
        
        print(f"\nCONFUSION MATRIX:")
        print("                   Actual (User)")
        print("                  H    M    B    P    N")
        print("   Predicted H   ", end="")
        for l in LABELS:
            print(f"{confusion['hand_only'][l]:>4}", end="")
        print()
        print("   (System)  M   ", end="")
        for l in LABELS:
            print(f"{confusion['mouth_only'][l]:>4}", end="")
        print()
        print("              B   ", end="")
        for l in LABELS:
            print(f"{confusion['both_no_pill'][l]:>4}", end="")
        print()
        print("              P   ", end="")
        for l in LABELS:
            print(f"{confusion['pill_intake'][l]:>4}", end="")
        print()
        print("              N   ", end="")
        for l in LABELS:
            print(f"{confusion['nothing'][l]:>4}", end="")
        print()
        print("   H=hand_only, M=mouth_only, B=both_no_pill, P=pill_intake, N=nothing")
        
        print(f"\nDETECTION RATES:")
        print(f"   Hand Detected: {self.hand_detected_count}/{self.total_frames} ({hand_rate:.1f}%)")
        print(f"   Mouth Detected: {self.mouth_detected_count}/{self.total_frames} ({mouth_rate:.1f}%)")
        
        print(f"\nDISTANCE STATS:")
        print(f"   Average Distance: {avg_distance:.1f} px")
        print(f"   Min Distance: {min_distance:.1f} px")
        print(f"   Max Distance: {max_distance:.1f} px")
        
        print(f"\nSPEED:")
        print(f"   Avg Latency: {avg_latency:.1f} ms")
        print(f"   Min Latency: {min_latency:.1f} ms")
        print(f"   Max Latency: {max_latency:.1f} ms")
        print(f"   Estimated FPS: {1000/avg_latency:.1f}" if avg_latency > 0 else "   (No data)")
        
        # ===== INTERPRETATION =====
        print(f"\nINTERPRETATION:")
        if accuracy >= 90:
            print("Excellent! System is highly accurate.")
        elif accuracy >= 75:
            print("Good. System is reliable with room for improvement.")
        elif accuracy >= 60:
            print("Moderate. Consider improving lighting or camera position.")
        else:
            print("Needs improvement. Check detection thresholds and lighting.")
        
        print("=" * 70)
        
        # ===== SAVE METRICS =====
        metrics_filename = f"metrics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(metrics_filename, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("📊 ACCURACY TEST RESULTS\n")
            f.write("=" * 70 + "\n")
            f.write(f"\nOverall Accuracy: {accuracy:.1f}%\n")
            f.write(f"Correct: {correct_count}/{total}\n\n")
            
            f.write("Per-Label Metrics:\n")
            for label in LABELS:
                m = metrics[label]
                f.write(f"  {label}: Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, F1={m['f1']:.3f}\n")
            
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"  TP: {metrics['pill_intake']['tp']}, FP: {metrics['pill_intake']['fp']}\n")
            f.write(f"  FN: {metrics['pill_intake']['fn']}, TN: {metrics['pill_intake']['tn']}\n")
            
            f.write(f"\nHand Detection Rate: {hand_rate:.1f}%\n")
            f.write(f"Mouth Detection Rate: {mouth_rate:.1f}%\n")
            f.write(f"Avg Distance: {avg_distance:.1f} px\n")
            f.write(f"Avg Latency: {avg_latency:.1f} ms\n")
        
        print(f"\n💾 Metrics saved to: {metrics_filename}")


# ==================== MAIN ====================

if __name__ == "__main__":
    test = ManualAccuracyTest()
    test.run()