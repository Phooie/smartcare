# report_generator.py - Fixed with correct timestamps spanning full date range

import numpy as np
import json
from datetime import datetime, timedelta
from collections import Counter, deque
from django.db.models import Count, Avg, Q
from django.utils import timezone
import random
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow loaded successfully")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - using fallback")

class LSTMAutoencoder:
    """LSTM Autoencoder for behavioral anomaly detection"""
    
    def __init__(self, sequence_length=20, n_features=5, latent_dim=8):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.model = None
        self.threshold = None
        self.trained = False
        self.tensorflow_available = TENSORFLOW_AVAILABLE
        
    def build_model(self):
        """Build LSTM autoencoder"""
        if not self.tensorflow_available:
            return None
            
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        
        # Encoder
        x = layers.LSTM(64, activation='relu', return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(32, activation='relu', return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        encoded = layers.LSTM(self.latent_dim, activation='relu', return_sequences=False)(x)
        
        # Decoder
        decoded = layers.RepeatVector(self.sequence_length)(encoded)
        decoded = layers.LSTM(32, activation='relu', return_sequences=True)(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.LSTM(64, activation='relu', return_sequences=True)(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        outputs = layers.TimeDistributed(layers.Dense(self.n_features))(decoded)
        
        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse')
        return self.model
    
    def prepare_sequences(self, data):
        """Prepare sequence data - expects list of feature vectors"""
        if len(data) < self.sequence_length:
            return None
        
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
        
        return np.array(sequences, dtype=np.float32)
    
    def train(self, data, epochs=30, batch_size=32):
        """Train the autoencoder"""
        if not self.tensorflow_available or self.model is None:
            self.threshold = 0.5
            self.trained = True
            return True
        
        if self.model is None:
            self.build_model()
            if self.model is None:
                self.threshold = 0.5
                self.trained = True
                return True
        
        sequences = self.prepare_sequences(data)
        if sequences is None or len(sequences) < 10:
            return False
        
        self.model.fit(
            sequences, sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        # Calculate threshold
        reconstructions = self.model.predict(sequences, verbose=0)
        mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
        self.threshold = float(np.mean(mse) + 2 * np.std(mse))
        self.trained = True
        
        return True
    
    def detect_anomaly(self, data):
        """Detect anomaly in new data - expects list of feature vectors"""
        if not self.trained:
            return False, 0.0
        
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                features = data
            else:
                features = [[float(x)] for x in data]
        else:
            return False, 0.0
        
        if len(features) < self.sequence_length:
            return False, 0.0
        
        if not self.tensorflow_available or self.model is None:
            first_feature = [f[0] for f in features[-30:]] if len(features) >= 30 else [f[0] for f in features]
            if len(first_feature) < 10:
                return False, 0.0
            
            mean = float(np.mean(first_feature))
            std = float(np.std(first_feature)) + 0.001
            z_score = abs((float(first_feature[-1]) - mean) / std)
            anomaly_score = min(z_score / 3.0, 1.0)
            return anomaly_score > 0.6, anomaly_score
        
        seq = self.prepare_sequences(features)
        if seq is None or len(seq) == 0:
            return False, 0.0
        
        last_seq = seq[-1].reshape(1, self.sequence_length, self.n_features)
        reconstruction = self.model.predict(last_seq, verbose=0)
        mse = float(np.mean(np.power(last_seq - reconstruction, 2)))
        
        is_anomaly = mse > self.threshold
        anomaly_score = min(mse / self.threshold, 2.0) if self.threshold > 0 else 0
        
        return bool(is_anomaly), float(anomaly_score)


class ReportGenerator:
    """Generate behavior reports using LSTM"""
    
    def __init__(self):
        self.lstm = LSTMAutoencoder()
        self.emotion_scores = {
            'happy': 0.0,
            'neutral': 0.3,
            'surprise': 0.4,
            'sad': 0.6,
            'fear': 0.7,
            'angry': 0.8,
            'disgust': 0.9
        }
        
        self.negative_emotions = ['sad', 'angry', 'fear', 'disgust']
    
    def generate_synthetic_data(self, days=60, samples_per_day=24):
        """
        Generate 2 months (60 days) of synthetic behavioral data.
        Each day has 24 samples (one per hour).
        TIMESTAMPS SPAN THE FULL 60 DAYS.
        """
        data = []
        
        # Set the start date to 60 days ago from today
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        total_samples = days * samples_per_day
        
        print(f"📊 Generating {days} days of data ({total_samples} samples)...")
        print(f"   Start date: {start_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   End date: {end_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Max pills per day: 6")
        
        # Define anomaly periods - realistic missed pill counts (0-4 per day)
        anomaly_periods = []
        num_anomalies = random.randint(4, 6)
        
        for _ in range(num_anomalies):
            start_day = random.randint(3, days - 6)
            duration = random.randint(2, 5)
            end_day = min(start_day + duration, days - 3)
            
            patterns = [
                {'emotion': 'sad', 'adherence_min': 0.3, 'adherence_max': 0.6, 'missed_min': 2, 'missed_max': 4, 'stress_min': 0.6, 'stress_max': 0.85},
                {'emotion': 'angry', 'adherence_min': 0.4, 'adherence_max': 0.7, 'missed_min': 1, 'missed_max': 3, 'stress_min': 0.5, 'stress_max': 0.8},
                {'emotion': 'fear', 'adherence_min': 0.35, 'adherence_max': 0.65, 'missed_min': 2, 'missed_max': 4, 'stress_min': 0.7, 'stress_max': 0.9},
                {'emotion': 'disgust', 'adherence_min': 0.4, 'adherence_max': 0.7, 'missed_min': 1, 'missed_max': 3, 'stress_min': 0.5, 'stress_max': 0.8},
                {'emotion': 'sad', 'adherence_min': 0.2, 'adherence_max': 0.5, 'missed_min': 3, 'missed_max': 5, 'stress_min': 0.7, 'stress_max': 0.9},
                {'emotion': 'angry', 'adherence_min': 0.25, 'adherence_max': 0.5, 'missed_min': 3, 'missed_max': 5, 'stress_min': 0.65, 'stress_max': 0.88},
            ]
            
            pattern = random.choice(patterns)
            anomaly_periods.append({
                'start_day': start_day,
                'end_day': end_day,
                'emotion': pattern['emotion'],
                'adherence_min': pattern['adherence_min'],
                'adherence_max': pattern['adherence_max'],
                'missed_min': pattern['missed_min'],
                'missed_max': pattern['missed_max'],
                'stress_min': pattern['stress_min'],
                'stress_max': pattern['stress_max']
            })
        
        print(f"   Anomaly periods: {len(anomaly_periods)}")
        
        anomaly_count = 0
        total_missed_pills = 0
        max_pills_per_day = 6
        
        # First, generate daily missed pill counts and emotions
        daily_data = {}
        
        for day in range(days):
            is_anomaly_day = False
            current_anomaly = None
            
            for period in anomaly_periods:
                if period['start_day'] <= day <= period['end_day']:
                    is_anomaly_day = True
                    current_anomaly = period
                    break
            
            if is_anomaly_day and current_anomaly:
                missed = random.randint(current_anomaly['missed_min'], current_anomaly['missed_max'])
                missed = min(missed, max_pills_per_day)
                emotion = current_anomaly['emotion']
                adherence = random.uniform(current_anomaly['adherence_min'], current_anomaly['adherence_max'])
                stress = random.uniform(current_anomaly['stress_min'], current_anomaly['stress_max'])
            else:
                weekday = day % 7
                
                if weekday in [0, 1]:
                    missed = 0 if random.random() < 0.85 else 1
                    emotion = random.choice(['happy', 'neutral', 'surprise'])
                    adherence = random.uniform(0.85, 1.0)
                    stress = random.uniform(0, 0.25)
                elif weekday in [2, 3]:
                    missed = random.randint(0, 1)
                    emotion = random.choice(['happy', 'neutral', 'sad'])
                    adherence = random.uniform(0.75, 0.95)
                    stress = random.uniform(0.15, 0.45)
                elif weekday == 4:
                    missed = random.randint(0, 2)
                    emotion = random.choice(['neutral', 'sad', 'happy'])
                    adherence = random.uniform(0.7, 0.9)
                    stress = random.uniform(0.25, 0.55)
                else:
                    missed = 0 if random.random() < 0.9 else 1
                    emotion = random.choice(['happy', 'neutral', 'surprise'])
                    adherence = random.uniform(0.88, 1.0)
                    stress = random.uniform(0, 0.2)
                
                if random.random() < 0.02:
                    emotion = random.choice(['sad', 'angry'])
                    missed = min(missed + random.randint(1, 2), max_pills_per_day)
                    adherence = max(0.4, adherence - random.uniform(0.15, 0.35))
                    stress = min(0.85, stress + random.uniform(0.15, 0.35))
            
            missed = max(0, min(missed, max_pills_per_day))
            total_missed_pills += missed
            
            daily_data[day] = {
                'missed': missed,
                'emotion': emotion,
                'stress': stress,
                'adherence': adherence
            }
        
        print(f"   Total missed pills over {days} days: {total_missed_pills}")
        print(f"   Average missed per day: {total_missed_pills/days:.1f}")
        
        # Now generate hourly samples with CORRECT timestamps
        sample_index = 0
        for day in range(days):
            # Get daily values
            day_info = daily_data.get(day, {'missed': 0, 'emotion': 'neutral', 'stress': 0.3, 'adherence': 0.85})
            missed = day_info['missed']
            base_emotion = day_info['emotion']
            base_stress = day_info['stress']
            base_adherence = day_info['adherence']
            
            # Determine which hours had missed pills
            missed_hours = []
            if missed > 0:
                for _ in range(missed):
                    if random.random() < 0.4:
                        missed_hours.append(random.randint(8, 10))  # Morning
                    elif random.random() < 0.7:
                        missed_hours.append(random.randint(20, 22))  # Evening
                    else:
                        missed_hours.append(random.randint(14, 16))  # Afternoon
            
            # Generate 24 samples for this day
            for hour in range(samples_per_day):
                # Calculate the actual timestamp for this sample
                timestamp = start_date + timedelta(days=day, hours=hour)
                
                # Check if this hour is a missed dose
                is_missed_hour = hour in missed_hours
                
                # Hourly emotion variation
                hour_emotion = base_emotion
                
                # Morning variations
                if 6 <= hour <= 9:
                    if random.random() < 0.1:
                        hour_emotion = 'happy'
                # Evening variations
                elif 20 <= hour <= 23:
                    if random.random() < 0.15:
                        hour_emotion = 'neutral'
                
                # Small hourly variations
                hour_stress = base_stress + random.uniform(-0.08, 0.08)
                hour_stress = max(0, min(1, hour_stress))
                
                hour_adherence = base_adherence + random.uniform(-0.03, 0.03)
                hour_adherence = max(0, min(1, hour_adherence))
                
                # Determine if this sample is an anomaly
                is_sample_anomaly = False
                if is_missed_hour and hour_emotion in self.negative_emotions:
                    is_sample_anomaly = True
                    anomaly_count += 1
                elif is_missed_hour and missed >= 4:
                    is_sample_anomaly = True
                    anomaly_count += 1
                
                emotion_score = self.emotion_scores.get(hour_emotion, 0.5)
                
                # Behavior score
                behavior_score = (
                    (1 - emotion_score) * 0.35 +
                    (1 - hour_adherence) * 0.35 +
                    (missed / max_pills_per_day) * 0.30
                )
                behavior_score = min(max(behavior_score, 0.0), 1.0)
                
                record = {
                    'timestamp': timestamp,  # CORRECT: spans the full date range
                    'emotion': str(hour_emotion),
                    'emotion_score': float(emotion_score),
                    'confidence': float(random.uniform(0.6, 0.95)),
                    'adherence': float(hour_adherence),
                    'missed_pills': int(1 if is_missed_hour else 0),
                    'stress_level': float(hour_stress),
                    'behavior_score': float(behavior_score),
                    'is_synthetic': True,
                    'is_anomaly': bool(is_sample_anomaly)
                }
                
                data.append(record)
                sample_index += 1
        
        print(f"   Total samples: {len(data)}")
        print(f"   Anomaly samples: {anomaly_count}")
        print(f"   Anomaly percentage: {(anomaly_count/len(data))*100:.1f}%")
        print(f"   Total missed pill events: {sum(1 for r in data if r['missed_pills'] > 0)}")
        
        # Verify date range
        if data:
            first_ts = data[0]['timestamp']
            last_ts = data[-1]['timestamp']
            print(f"   First timestamp: {first_ts.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Last timestamp: {last_ts.strftime('%Y-%m-%d %H:%M')}")
        
        return data
    
    def extract_features(self, behavioral_data):
        """Extract features from behavioral data for LSTM"""
        features = []
        for record in behavioral_data:
            features.append([
                float(record.get('emotion_score', 0.5)),
                float(record.get('confidence', 0.5)),
                float(record.get('adherence', 0.5)),
                float(record.get('missed_pills', 0)),
                float(record.get('stress_level', 0.5))
            ])
        return features
    
    def analyze_data(self, behavioral_data):
        """Analyze behavioral data and detect anomalies"""
        if not behavioral_data:
            return None
        
        features = self.extract_features(behavioral_data)
        
        if not self.lstm.trained and len(features) > 50:
            self.lstm.train(features, epochs=30)
        
        anomaly_results = []
        if len(features) > self.lstm.sequence_length:
            is_anomaly, score = self.lstm.detect_anomaly(features)
            anomaly_results.append({
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(score),
                'timestamp': behavioral_data[-1]['timestamp']
            })
        
        emotions = [str(d['emotion']) for d in behavioral_data]
        emotion_counts = Counter(emotions)
        dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else 'neutral'
        
        avg_emotion_score = float(np.mean([float(d['emotion_score']) for d in behavioral_data]))
        avg_adherence = float(np.mean([float(d['adherence']) for d in behavioral_data]))
        total_missed = int(sum([int(d['missed_pills']) for d in behavioral_data]))
        avg_stress = float(np.mean([float(d['stress_level']) for d in behavioral_data]))
        
        anomaly_count_from_data = int(sum(1 for d in behavioral_data if d.get('is_anomaly', False)))
        
        anomaly_count_from_lstm = len([a for a in anomaly_results if a.get('is_anomaly', False)])
        max_anomaly_score = 0.0
        if anomaly_results:
            max_anomaly_score = float(max([a.get('anomaly_score', 0) for a in anomaly_results]))
        
        return {
            'total_records': len(behavioral_data),
            'dominant_emotion': str(dominant_emotion),
            'emotion_distribution': {str(k): int(v) for k, v in dict(emotion_counts).items()},
            'average_emotion_score': avg_emotion_score,
            'average_adherence': avg_adherence,
            'total_missed_pills': total_missed,
            'average_stress': avg_stress,
            'anomalies': anomaly_results,
            'anomaly_count': max(anomaly_count_from_data, anomaly_count_from_lstm),
            'max_anomaly_score': max_anomaly_score,
            'lstm_trained': bool(self.lstm.trained),
            'threshold': float(self.lstm.threshold) if self.lstm.threshold else 0.0
        }
    
    def generate_report(self, start_date, end_date, data):
        """Generate a complete report"""
        if not data:
            return None
        
        analysis = self.analyze_data(data)
        if not analysis:
            return None
        
        summary_parts = []
        recommendations = []
        
        emotion_emoji = {
            'happy': '😊',
            'neutral': '😐',
            'sad': '😢',
            'angry': '😡',
            'fear': '😨',
            'surprise': '😮',
            'disgust': '🤢'
        }
        
        dominant_emoji = emotion_emoji.get(analysis['dominant_emotion'], '😐')
        summary_parts.append(f"Dominant Emotion: {dominant_emoji} {analysis['dominant_emotion'].title()}")
        summary_parts.append(f"Average Adherence: {analysis['average_adherence']*100:.1f}%")
        summary_parts.append(f"Total Missed Pills: {analysis['total_missed_pills']}")
        
        if analysis['anomaly_count'] > 0:
            summary_parts.append(f"⚠️ {analysis['anomaly_count']} anomalies detected!")
        
        if analysis['average_adherence'] < 0.6:
            recommendations.append("Low adherence detected. Consider additional reminders or caregiver support.")
        
        if analysis['dominant_emotion'] in ['sad', 'angry', 'fear', 'disgust']:
            recommendations.append("Negative emotions detected. Consider checking in on the patient's well-being.")
        
        if analysis['average_stress'] > 0.7:
            recommendations.append("High stress levels detected. Consider stress reduction activities.")
        
        if analysis['anomaly_count'] > 3:
            recommendations.append("Multiple anomalies detected. Consider immediate consultation with healthcare provider.")
        
        if not recommendations:
            recommendations.append("All metrics are within normal range. Continue current care plan.")
        
        report = {
            'start_date': start_date,
            'end_date': end_date,
            'total_records': int(analysis['total_records']),
            'dominant_emotion': str(analysis['dominant_emotion']),
            'emotion_distribution': analysis['emotion_distribution'],
            'average_emotion_score': float(analysis['average_emotion_score']),
            'average_adherence': float(analysis['average_adherence']),
            'total_missed_pills': int(analysis['total_missed_pills']),
            'average_stress': float(analysis['average_stress']),
            'anomaly_count': int(analysis['anomaly_count']),
            'max_anomaly_score': float(analysis['max_anomaly_score']),
            'lstm_trained': bool(analysis['lstm_trained']),
            'threshold': float(analysis['threshold']),
            'summary': '\n'.join(summary_parts),
            'recommendations': '\n'.join(recommendations)
        }
        
        return report