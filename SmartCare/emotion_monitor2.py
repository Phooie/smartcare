# emotion_monitor.py - Updated with 5-minute negative emotion detection
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
    NEGATIVE_EMOTIONS = ['angry', 'disgust', 'fear', 'sad']
    
    def __init__(self):
        self.emotion_history = deque(maxlen=1000)  # Store more history
        self.current_emotion = 'neutral'
        self.confidence = 0.0
        self.use_deepface = DEEPFACE_AVAILABLE
        self.use_fer = FER_AVAILABLE
        
        # Track negative emotion streaks
        self.negative_start_time = None
        self.consecutive_negative_count = 0
        self.total_negative_count = 0
        
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
    
    def is_negative_emotion(self, emotion):
        """Check if emotion is negative"""
        return emotion in self.NEGATIVE_EMOTIONS
    
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
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
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
        
        # Track negative emotions
        is_negative = self.is_negative_emotion(emotion)
        
        self.emotion_history.append({
            'timestamp': datetime.now(),
            'emotion': emotion,
            'confidence': confidence,
            'is_negative': is_negative
        })
        
        # Update negative streak tracking
        if is_negative:
            self.consecutive_negative_count += 1
            self.total_negative_count += 1
            if self.negative_start_time is None:
                self.negative_start_time = datetime.now()
        else:
            # Reset streak on positive/neutral emotion
            self.consecutive_negative_count = 0
            self.negative_start_time = None
        
        return emotion, confidence
    
    def get_negative_emotion_duration(self):
        """Get how long negative emotions have been持续的"""
        if self.negative_start_time is None:
            return 0
        
        duration = (datetime.now() - self.negative_start_time).total_seconds()
        return duration
    
    def get_negative_emotion_percentage(self, minutes=5):
        """Get percentage of negative emotions in last N minutes"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=minutes)
        
        recent = [h for h in self.emotion_history if h['timestamp'] > cutoff]
        
        if not recent:
            return 0.0
        
        negative_count = sum(1 for h in recent if h['is_negative'])
        percentage = (negative_count / len(recent)) * 100
        
        return percentage
    
    def should_alert(self, minutes=5, threshold_percentage=70):
        """Check if alert should be sent based on negative emotion duration"""
        # Check if we have enough data
        if len(self.emotion_history) < 10:
            return False, "Not enough data"
        
        # Check if negative emotions have been持续的 for at least 'minutes' minutes
        negative_duration = self.get_negative_emotion_duration()
        if negative_duration < (minutes * 60):
            return False, f"Negative emotions only for {negative_duration/60:.1f} minutes"
        
        # Check if negative emotion percentage is high
        percentage = self.get_negative_emotion_percentage(minutes)
        if percentage < threshold_percentage:
            return False, f"Negative emotion percentage: {percentage:.1f}% (threshold: {threshold_percentage}%)"
        
        # Get the dominant negative emotion
        negative_emotions = [h['emotion'] for h in self.emotion_history if h['is_negative']]
        if negative_emotions:
            from collections import Counter
            counter = Counter(negative_emotions)
            dominant = counter.most_common(1)[0][0]
        else:
            dominant = 'unknown'
        
        return True, {
            'duration': negative_duration,
            'percentage': percentage,
            'dominant_emotion': dominant,
            'total_negative': self.total_negative_count
        }


# ============================================================
# SIMPLE ANOMALY DETECTION (No TensorFlow LSTM)
# ============================================================

class SimpleAnomalyDetector:
    """Simple statistical anomaly detection (no TensorFlow needed)"""
    
    def __init__(self):
        self.history = deque(maxlen=1000)
        self.threshold = 0.6
        
    def detect_anomaly(self, data):
        """Detect anomaly using simple statistics"""
        if not data or len(data) == 0:
            return False, 0.0
        
        # Ensure all elements are numbers
        clean_data = []
        for item in data:
            if isinstance(item, (int, float)):
                clean_data.append(float(item))
            else:
                clean_data.append(0.5)
        
        if len(clean_data) == 0:
            return False, 0.0
        
        # Store in history
        for val in clean_data:
            self.history.append(val)
        
        if len(self.history) < 30:
            return False, 0.0
        
        # Get recent data
        recent = list(self.history)[-30:]
        
        try:
            recent_array = np.array(recent, dtype=np.float64)
        except:
            recent_array = np.array([float(x) for x in recent], dtype=np.float64)
        
        mean = np.mean(recent_array)
        std = np.std(recent_array) + 0.001
        
        latest = recent_array[-1]
        z_score = abs((latest - mean) / std)
        anomaly_score = min(z_score / 3.0, 1.0)
        is_anomaly = anomaly_score > self.threshold
        
        return is_anomaly, anomaly_score


# ============================================================
# BEHAVIOR MONITOR - Updated with 5-minute negative emotion check
# ============================================================

class BehaviorMonitor:
    """Monitors behavior with 3-minute negative emotion detection"""
    
    def __init__(self):
        self.emotion_recognizer = EmotionRecognizer()
        self.anomaly_detector = SimpleAnomalyDetector()
        self.data_buffer = []
        self.anomaly_threshold = 0.6
        self.alert_cooldown = 600  # 10 minutes between alerts (increased)
        self.last_alert_time = None
        self.alert_count = 0
        self._last_behavior_score = 0.0
        
        # Configuration for negative emotion alerts
        self.negative_emotion_minutes = 3  # Alert after 5 minutes of negative emotions
        self.negative_emotion_threshold = 70  # 70% of emotions must be negative
        
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
            (1 - emotion_score) * 0.4 +
            (1 - adherence) * 0.4 +
            (missed_pills / 5.0) * 0.2
        )
        behavior_score = min(max(behavior_score, 0.0), 1.0)
        
        self._last_behavior_score = behavior_score
        
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
        """
        Detect if current behavior is anomalous.
        Only alerts if negative emotions persist for > 5 minutes.
        """
        if len(self.data_buffer) < 10:
            return None
        
        # ===== CHECK NEGATIVE EMOTION DURATION =====
        should_alert, alert_data = self.emotion_recognizer.should_alert(
            minutes=self.negative_emotion_minutes,
            threshold_percentage=self.negative_emotion_threshold
        )
        
        if should_alert:
            # Check cooldown
            if self.last_alert_time:
                cooldown_remaining = (datetime.now() - self.last_alert_time).total_seconds()
                if cooldown_remaining < self.alert_cooldown:
                    return None
            
            self.last_alert_time = datetime.now()
            self.alert_count += 1
            
            # Get recent adherence
            recent = self.data_buffer[-10:] if len(self.data_buffer) >= 10 else self.data_buffer
            avg_adherence = 0
            if recent:
                adherence_values = [d.get('adherence', 0.5) for d in recent if isinstance(d.get('adherence'), (int, float))]
                if adherence_values:
                    avg_adherence = sum(adherence_values) / len(adherence_values)
            
            # Calculate anomaly score based on negative emotion duration
            duration_minutes = alert_data['duration'] / 60
            percentage = alert_data['percentage']
            
            # Higher score for longer duration and higher percentage
            anomaly_score = min(
                (duration_minutes / 10) * 0.5 +  # Duration factor (max 0.5 at 10 minutes)
                (percentage / 100) * 0.5,        # Percentage factor (max 0.5 at 100%)
                1.0
            )
            
            return {
                'anomaly_score': anomaly_score,
                'timestamp': datetime.now(),
                'current_emotion': alert_data.get('dominant_emotion', 'unknown'),
                'recent_adherence': avg_adherence * 100,
                'alert_count': self.alert_count,
                'negative_duration_minutes': duration_minutes,
                'negative_percentage': percentage,
                'message': f"⚠️ {alert_data['dominant_emotion']} detected for {duration_minutes:.1f} minutes ({percentage:.0f}% negative emotions)"
            }
        
        # ===== ALSO CHECK STATISTICAL ANOMALY (backup) =====
        # This is a backup in case the negative emotion check misses something
        recent_scores = []
        for d in self.data_buffer[-30:]:
            score = d.get('behavior_score', 0.5)
            if isinstance(score, (int, float)):
                recent_scores.append(float(score))
            else:
                recent_scores.append(0.5)
        
        is_anomaly, score = self.anomaly_detector.detect_anomaly(recent_scores)
        
        if is_anomaly and score > 0.8:  # Higher threshold for statistical anomaly
            # Only trigger if we haven't sent an alert recently
            if self.last_alert_time:
                cooldown_remaining = (datetime.now() - self.last_alert_time).total_seconds()
                if cooldown_remaining < self.alert_cooldown:
                    return None
            
            self.last_alert_time = datetime.now()
            self.alert_count += 1
            
            recent = self.data_buffer[-10:] if len(self.data_buffer) >= 10 else self.data_buffer
            avg_adherence = 0
            if recent:
                adherence_values = [d.get('adherence', 0.5) for d in recent if isinstance(d.get('adherence'), (int, float))]
                if adherence_values:
                    avg_adherence = sum(adherence_values) / len(adherence_values)
            
            return {
                'anomaly_score': score,
                'timestamp': datetime.now(),
                'current_emotion': self.emotion_recognizer.current_emotion,
                'recent_adherence': avg_adherence * 100,
                'alert_count': self.alert_count,
                'message': f"Statistical anomaly detected (score: {score:.2f})"
            }
        
        return None


# ============================================================
# MOCK DATA GENERATOR
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
print("✅ Alert conditions:")
print(f"   - {3} minutes of negative emotions")
print(f"   - {70}% negative emotion threshold")
print(f"   - {10} minute cooldown between alerts")
print("="*50 + "\n")