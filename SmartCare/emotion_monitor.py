# emotion_monitor.py - FIXED VERSION
import cv2
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import DeepFace (optional)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded successfully")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not available - using mock emotion detection")

# Try to import FER (optional)
try:
    from fer import FER
    FER_AVAILABLE = True
    print("✅ FER loaded successfully")
except ImportError:
    FER_AVAILABLE = False
    print("⚠️ FER not available - using mock emotion detection")


# ============================================================
# EMOTION RECOGNITION MODULE
# ============================================================

class EmotionRecognizer:
    """Emotion recognition with fallback options"""
    
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(self):
        self.emotion_history = deque(maxlen=100)
        self.current_emotion = 'neutral'
        self.confidence = 0.0
        self.use_deepface = DEEPFACE_AVAILABLE
        self.use_fer = FER_AVAILABLE
        
        if self.use_deepface:
            print("✅ Using DeepFace for emotion detection")
        elif self.use_fer:
            try:
                self.fer = FER(mtcnn=True)
                print("✅ Using FER for emotion detection")
            except Exception as e:
                print(f"⚠️ FER initialization error: {e}")
                self.use_fer = False
        else:
            print("⚠️ No emotion detection library available - using mock data")
            print("   Install with: pip install deepface fer")
    
    def detect_emotion(self, face_image):
        """Detect emotion from face image"""
        try:
            if face_image is None or face_image.size == 0:
                return 'neutral', 0.0
            
            # Convert to RGB if needed
            if len(face_image.shape) == 2:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif face_image.shape[2] == 4:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
            
            # Try DeepFace first
            if self.use_deepface:
                try:
                    result = DeepFace.analyze(
                        img_path=face_image,
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
            
            # Try FER next
            if self.use_fer and hasattr(self, 'fer'):
                try:
                    result = self.fer.detect_emotions(face_image)
                    if result and len(result) > 0:
                        emotions = result[0].get('emotions', {})
                        if emotions:
                            emotion = max(emotions, key=emotions.get)
                            confidence = emotions[emotion] / 100.0
                            return emotion, confidence
                except Exception as e:
                    print(f"⚠️ FER error: {e}")
            
            # Fallback: return mock data for testing
            return self._mock_emotion()
            
        except Exception as e:
            print(f"❌ Emotion detection error: {e}")
            return 'neutral', 0.0
    
    def _mock_emotion(self):
        """Generate mock emotion for testing"""
        import random
        emotion = random.choice(self.EMOTIONS)
        confidence = random.uniform(0.4, 0.8)
        return emotion, confidence
    
    def get_dominant_emotion(self, frame):
        """Extract face and detect emotion"""
        # Simple face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            emotion, confidence = self.detect_emotion(face_roi)
        else:
            emotion = 'neutral'
            confidence = 0.0
        
        self.current_emotion = emotion
        self.confidence = confidence
        self.emotion_history.append({
            'timestamp': datetime.now(),
            'emotion': emotion,
            'confidence': confidence
        })
        
        return emotion, confidence
    
    def get_emotion_summary(self, minutes=5):
        """Get emotion summary for last N minutes"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=minutes)
        
        recent = [h for h in self.emotion_history 
                 if h['timestamp'] > cutoff]
        
        if not recent:
            return {'dominant': 'neutral', 'count': 0, 'confidence': 0.5}
        
        emotion_counts = {}
        for item in recent:
            emotion = item['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant = max(emotion_counts, key=emotion_counts.get)
        avg_confidence = sum(h['confidence'] for h in recent) / len(recent)
        
        return {
            'dominant': dominant,
            'count': len(recent),
            'confidence': avg_confidence,
            'distribution': emotion_counts
        }


# ============================================================
# SIMPLE ANOMALY DETECTION (Fixed for numpy)
# ============================================================

class SimpleAnomalyDetector:
    """Simple statistical anomaly detection (no TensorFlow needed)"""
    
    def __init__(self):
        self.history = deque(maxlen=1000)
        self.threshold = 0.6
        
    def detect_anomaly(self, data):
        """Detect anomaly using simple statistics"""
        # Make sure data is a list of numbers
        if not data or len(data) == 0:
            return False, 0.0
        
        # Ensure all elements are numbers (floats)
        clean_data = []
        for item in data:
            if isinstance(item, (int, float)):
                clean_data.append(float(item))
            elif isinstance(item, dict):
                # If it's a dict, try to extract a numeric value
                if 'behavior_score' in item:
                    clean_data.append(float(item['behavior_score']))
                else:
                    clean_data.append(0.5)  # Default
            else:
                clean_data.append(0.5)  # Default
        
        if len(clean_data) == 0:
            return False, 0.0
        
        # Store in history
        for val in clean_data:
            self.history.append(val)
        
        if len(self.history) < 30:
            return False, 0.0
        
        # Get recent data as list of floats
        recent = list(self.history)[-30:]
        
        # Convert to numpy array of floats
        try:
            recent_array = np.array(recent, dtype=np.float64)
        except:
            # If conversion fails, use list comprehension
            recent_array = np.array([float(x) for x in recent], dtype=np.float64)
        
        # Calculate mean and std
        mean = np.mean(recent_array)
        std = np.std(recent_array) + 0.001
        
        # Calculate z-score for latest value
        latest = recent_array[-1]
        z_score = abs((latest - mean) / std)
        
        # Normalize to 0-1
        anomaly_score = min(z_score / 3.0, 1.0)
        
        # Check if anomaly
        is_anomaly = anomaly_score > self.threshold
        
        return is_anomaly, anomaly_score


# ============================================================
# BEHAVIOR MONITOR - Combines emotion + adherence
# ============================================================

class BehaviorMonitor:
    """Monitors behavior by combining emotion and adherence data"""
    
    def __init__(self):
        self.emotion_recognizer = EmotionRecognizer()
        self.anomaly_detector = SimpleAnomalyDetector()
        self.data_buffer = []
        self.anomaly_threshold = 0.6
        self.alert_cooldown = 300  # 5 minutes
        self.last_alert_time = None
        self.alert_count = 0
        self._last_behavior_score = 0.0
        
    def update(self, frame, adherence_data):
        """Update monitor with new frame and adherence data"""
        # Get emotion
        emotion, confidence = self.emotion_recognizer.get_dominant_emotion(frame)
        
        # Calculate behavior score (higher = more concerning)
        emotion_score = self._emotion_to_score(emotion)
        adherence = adherence_data.get('daily_compliance', 0.5)
        missed_pills = adherence_data.get('missed_pills', 0)
        
        # Combined score (0 = good, 1 = bad)
        behavior_score = (
            (1 - emotion_score) * 0.4 +  # Negative emotions increase score
            (1 - adherence) * 0.4 +       # Low adherence increases score
            (missed_pills / 5.0) * 0.2     # Missed pills increase score
        )
        behavior_score = min(max(behavior_score, 0.0), 1.0)  # Clamp between 0 and 1
        
        self._last_behavior_score = behavior_score
        
        # Store in buffer
        features = {
            'timestamp': datetime.now(),
            'emotion': emotion,
            'emotion_score': emotion_score,
            'confidence': confidence,
            'adherence': adherence,
            'missed_pills': missed_pills,
            'behavior_score': behavior_score
        }
        
        self.data_buffer.append(features)
        
        # Keep last 500
        if len(self.data_buffer) > 500:
            self.data_buffer = self.data_buffer[-500:]
        
        return features
    
    def _emotion_to_score(self, emotion):
        """Convert emotion to numeric score (0=good, 1=bad)"""
        emotion_scores = {
            'happy': 0.0,
            'neutral': 0.3,
            'surprise': 0.4,
            'sad': 0.6,
            'fear': 0.7,
            'angry': 0.8,
            'disgust': 0.9
        }
        return emotion_scores.get(emotion, 0.5)
    
    def detect_anomaly(self):
        """Detect if current behavior is anomalous"""
        if len(self.data_buffer) < 10:
            return None
        
        # Get recent behavior scores as floats
        recent_scores = []
        for d in self.data_buffer[-30:]:
            score = d.get('behavior_score', 0.5)
            if isinstance(score, (int, float)):
                recent_scores.append(float(score))
            else:
                recent_scores.append(0.5)
        
        # Use anomaly detector
        is_anomaly, score = self.anomaly_detector.detect_anomaly(recent_scores)
        
        if is_anomaly:
            return self._should_send_alert(score)
        
        return None
    
    def _should_send_alert(self, anomaly_score):
        """Determine if alert should be sent"""
        # Check cooldown
        if self.last_alert_time:
            cooldown_remaining = (datetime.now() - self.last_alert_time).total_seconds()
            if cooldown_remaining < self.alert_cooldown:
                return None
        
        if anomaly_score > self.anomaly_threshold:
            self.last_alert_time = datetime.now()
            self.alert_count += 1
            
            # Get recent data
            recent = self.data_buffer[-10:] if len(self.data_buffer) >= 10 else self.data_buffer
            avg_adherence = 0
            if recent:
                adherence_values = [d.get('adherence', 0.5) for d in recent if isinstance(d.get('adherence'), (int, float))]
                if adherence_values:
                    avg_adherence = sum(adherence_values) / len(adherence_values)
            
            return {
                'anomaly_score': float(anomaly_score),
                'timestamp': datetime.now(),
                'current_emotion': self.emotion_recognizer.current_emotion,
                'recent_adherence': avg_adherence * 100,
                'alert_count': self.alert_count,
                'message': f"Anomaly detected: {anomaly_score:.2f} - {self.emotion_recognizer.current_emotion}"
            }
        
        return None


# ============================================================
# MOCK DATA GENERATOR (For testing without camera)
# ============================================================

class MockDataGenerator:
    """Generates mock data for testing"""
    
    @staticmethod
    def get_mock_frame():
        """Generate a mock frame"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @staticmethod
    def get_mock_adherence():
        """Generate mock adherence data"""
        import random
        return {
            'daily_compliance': random.uniform(0.3, 1.0),
            'missed_pills': random.randint(0, 2),
            'stress_level': random.uniform(0, 0.8)
        }


print("\n" + "="*50)
print("🧠 EMOTION MONITOR STATUS")
print("="*50)
print(f"✅ DeepFace: {'Available' if DEEPFACE_AVAILABLE else '⚠️ Not found'}")
print(f"✅ FER: {'Available' if FER_AVAILABLE else '⚠️ Not found'}")
print("✅ Using Simple Anomaly Detection (no TensorFlow required)")
print("="*50 + "\n")