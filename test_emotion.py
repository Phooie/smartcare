# test_emotion.py - Standalone emotion detection test
# Run this to test emotion detection without the website

import cv2
import numpy as np
from datetime import datetime
import time

# Try to import emotion libraries
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded successfully")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not available")

try:
    from fer import FER
    FER_AVAILABLE = True
    print("✅ FER loaded successfully")
except ImportError:
    FER_AVAILABLE = False
    print("⚠️ FER not available")

# ============================================================
# EMOTION DETECTION CLASS
# ============================================================

class EmotionDetector:
    """Simple emotion detector for testing"""
    
    # Emotion labels with emojis
    EMOTION_LABELS = {
        'happy': ('😊 Happy', '#00ff00'),
        'neutral': ('😐 Neutral', '#ffff00'),
        'sad': ('😢 Sad', '#4444ff'),
        'angry': ('😡 Angry', '#ff0000'),
        'fear': ('😨 Fear', '#ff8800'),
        'surprise': ('😮 Surprise', '#ff44ff'),
        'disgust': ('🤢 Disgust', '#00ff44'),
    }
    
    def __init__(self):
        self.current_emotion = 'neutral'
        self.confidence = 0.0
        self.emotion_history = []
        self.use_deepface = DEEPFACE_AVAILABLE
        self.use_fer = FER_AVAILABLE
        
        # Initialize FER if available
        if self.use_fer and not self.use_deepface:
            self.fer = FER(mtcnn=True)
            print("✅ Using FER for emotion detection")
        elif self.use_deepface:
            print("✅ Using DeepFace for emotion detection")
        else:
            print("⚠️ No emotion detection library available!")
            print("   Install with: pip install deepface fer")
    
    def detect_emotion(self, face_roi):
        """Detect emotion from face region"""
        try:
            # Convert to RGB (OpenCV uses BGR)
            if len(face_roi.shape) == 3:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_roi
            
            # Try DeepFace first
            if self.use_deepface:
                try:
                    result = DeepFace.analyze(
                        img_path=face_rgb,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True
                    )
                    if result and len(result) > 0:
                        emotions = result[0].get('emotion', {})
                        if emotions:
                            emotion = max(emotions, key=emotions.get)
                            confidence = emotions[emotion] / 100.0
                            return emotion, confidence
                except Exception as e:
                    print(f"⚠️ DeepFace error: {e}")
            
            # Try FER if DeepFace failed
            if self.use_fer:
                try:
                    result = self.fer.detect_emotions(face_rgb)
                    if result and len(result) > 0:
                        emotions = result[0].get('emotions', {})
                        if emotions:
                            emotion = max(emotions, key=emotions.get)
                            confidence = emotions[emotion] / 100.0
                            return emotion, confidence
                except Exception as e:
                    print(f"⚠️ FER error: {e}")
            
            return 'neutral', 0.0
            
        except Exception as e:
            print(f"❌ Emotion detection error: {e}")
            return 'neutral', 0.0
    
    def process_frame(self, frame):
        """Process a frame and detect emotions"""
        # Simple face detection using OpenCV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Detect emotion
            emotion, confidence = self.detect_emotion(face_roi)
            
            # Store result
            results.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'emotion': emotion,
                'confidence': confidence
            })
        
        # Update current emotion (use first face)
        if results:
            self.current_emotion = results[0]['emotion']
            self.confidence = results[0]['confidence']
            self.emotion_history.append({
                'timestamp': datetime.now(),
                'emotion': self.current_emotion,
                'confidence': self.confidence
            })
            
            # Keep last 100 records
            if len(self.emotion_history) > 100:
                self.emotion_history = self.emotion_history[-100:]
        
        return results, faces
    
    def draw_results(self, frame, results):
        """Draw emotion results on frame"""
        for result in results:
            x, y, w, h = result['x'], result['y'], result['w'], result['h']
            emotion = result['emotion']
            confidence = result['confidence']
            
            # Get label and color
            label, color = self.EMOTION_LABELS.get(emotion, ('Unknown', '#888888'))
            color_bgr = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))[::-1]
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 3)
            
            # Draw background for text
            text = f"{label} ({confidence*100:.1f}%)"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Background rectangle for text
            cv2.rectangle(frame, 
                         (x, y - text_h - 15), 
                         (x + text_w + 20, y), 
                         color_bgr, -1)
            
            # Draw text
            cv2.putText(frame, text, 
                       (x + 10, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw confidence bar
            bar_x = x
            bar_y = y + h + 10
            bar_w = w
            bar_h = 8
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                         (50, 50, 50), -1)
            
            # Confidence bar
            conf_w = int(bar_w * confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_w, bar_y + bar_h), 
                         color_bgr, -1)
            
            # Confidence text
            cv2.putText(frame, f"{confidence*100:.0f}%", 
                       (bar_x + bar_w + 5, bar_y + bar_h),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame


# ============================================================
# MAIN TEST FUNCTION
# ============================================================

def main():
    """Main test function"""
    print("\n" + "="*60)
    print("🧠 EMOTION DETECTION TEST")
    print("="*60)
    print("Press 'q' or ESC to quit")
    print("Press 'r' to reset history")
    print("Press 's' to save a screenshot")
    print("Press 'h' to show emotion history")
    print("="*60 + "\n")
    
    # Initialize detector
    detector = EmotionDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return
    
    # Get frame dimensions
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        print(f"📷 Camera: {width}x{height}")
    
    # FPS tracking
    fps_start = time.time()
    fps_count = 0
    fps = 0
    
    # Statistics
    emotion_stats = {}
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        # Flip horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Process frame
        results, faces = detector.process_frame(frame)
        
        # Draw results
        frame = detector.draw_results(frame, results)
        
        # Calculate FPS
        fps_count += 1
        if time.time() - fps_start >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_start = time.time()
        
        # Update emotion statistics
        if results:
            for r in results:
                emotion = r['emotion']
                emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1
        
        # Draw FPS and info
        cv2.putText(frame, f"FPS: {fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw emotion count
        y_pos = 60
        for emotion, count in sorted(emotion_stats.items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
            emoji = detector.EMOTION_LABELS.get(emotion, ('', ''))[0]
            cv2.putText(frame, f"{emoji} {emotion}: {count}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 20
        
        # Draw current emotion big on screen
        if results:
            emotion = results[0]['emotion']
            label, color = detector.EMOTION_LABELS.get(emotion, ('Unknown', '#888888'))
            color_bgr = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))[::-1]
            
            # Big text at bottom
            cv2.putText(frame, f"{label}", (width//2 - 100, height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_bgr, 4)
            
            # Confidence
            cv2.putText(frame, f"{detector.confidence*100:.1f}%", 
                       (width//2 - 80, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Show controls
        cv2.putText(frame, "q:Quit r:Reset s:Screenshot h:History", 
                   (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('🧠 Emotion Detection Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC
            break
        
        elif key == ord('r'):  # Reset history
            detector.emotion_history = []
            emotion_stats = {}
            print("🔄 History reset")
        
        elif key == ord('s'):  # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
        
        elif key == ord('h'):  # Show history
            print("\n📊 EMOTION HISTORY")
            print("="*50)
            if detector.emotion_history:
                for i, record in enumerate(detector.emotion_history[-10:]):
                    time_str = record['timestamp'].strftime("%H:%M:%S")
                    emoji = detector.EMOTION_LABELS.get(record['emotion'], ('', ''))[0]
                    print(f"  {i+1}. {time_str} - {emoji} {record['emotion']} ({record['confidence']*100:.1f}%)")
            else:
                print("  No history yet")
            print("="*50 + "\n")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "="*60)
    print("📊 FINAL STATISTICS")
    print("="*60)
    print(f"Total frames processed: {sum(emotion_stats.values())}")
    print(f"Running time: {time.time() - start_time:.1f} seconds")
    
    if emotion_stats:
        print("\nEmotion distribution:")
        total = sum(emotion_stats.values())
        for emotion, count in sorted(emotion_stats.items(), 
                                     key=lambda x: x[1], reverse=True):
            emoji = detector.EMOTION_LABELS.get(emotion, ('', ''))[0]
            percentage = (count / total) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {emoji} {emotion:10s}: {bar} {percentage:.1f}% ({count})")
    
    print("="*60)
    print("✅ Test complete!")


# ============================================================
# RUN THE TEST
# ============================================================

if __name__ == "__main__":
    main()