# test_lstm.py - Test LSTM Autoencoder with Synthetic Data
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime, timedelta
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
    print("❌ TensorFlow not available. Install with: pip install tensorflow")
    exit(1)


# ============================================================
# LSTM AUTOENCODER
# ============================================================

class LSTMAutoencoder:
    """LSTM Autoencoder for anomaly detection in behavioral data"""
    
    def __init__(self, sequence_length=20, n_features=5, latent_dim=8):
        """
        sequence_length: Number of time steps to look back
        n_features: Number of features (emotion score, adherence, etc.)
        latent_dim: Dimension of the latent space
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.model = None
        self.threshold = None
        self.trained = False
        
    def build_model(self):
        """Build LSTM autoencoder architecture"""
        # ENCODER
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        
        # First LSTM layer
        x = layers.LSTM(64, activation='relu', return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        
        # Second LSTM layer
        x = layers.LSTM(32, activation='relu', return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        
        # Third LSTM layer (bottleneck)
        encoded = layers.LSTM(self.latent_dim, activation='relu', return_sequences=False)(x)
        
        # DECODER
        # Repeat the latent vector
        decoded = layers.RepeatVector(self.sequence_length)(encoded)
        
        # First LSTM layer in decoder
        decoded = layers.LSTM(32, activation='relu', return_sequences=True)(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        
        # Second LSTM layer in decoder
        decoded = layers.LSTM(64, activation='relu', return_sequences=True)(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        
        # Output layer
        outputs = layers.TimeDistributed(layers.Dense(self.n_features))(decoded)
        
        # Create model
        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse')
        
        print(f"✅ LSTM Autoencoder built")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Features: {self.n_features}")
        print(f"   Latent dimension: {self.latent_dim}")
        
        return self.model
    
    def prepare_sequences(self, data):
        """Prepare sequence data for LSTM"""
        if len(data) < self.sequence_length:
            return None
        
        data_array = np.array(data)
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        
        sequences = []
        for i in range(len(data_array) - self.sequence_length + 1):
            seq = data_array[i:i + self.sequence_length]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def train(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train the autoencoder on normal behavior data"""
        if self.model is None:
            self.build_model()
        
        sequences = self.prepare_sequences(data)
        if sequences is None or len(sequences) < 10:
            print("⚠️ Not enough data for training")
            return False
        
        print(f"\n📊 Training Data:")
        print(f"   Total sequences: {len(sequences)}")
        print(f"   Sequence shape: {sequences.shape}")
        
        # Train
        history = self.model.fit(
            sequences, sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        # Calculate reconstruction error threshold
        reconstructions = self.model.predict(sequences, verbose=0)
        mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
        
        # Use mean + 2*std as threshold (95% confidence)
        self.threshold = np.mean(mse) + 2 * np.std(mse)
        self.trained = True
        
        print(f"\n✅ Training Complete!")
        print(f"   Training Loss: {history.history['loss'][-1]:.4f}")
        print(f"   Validation Loss: {history.history['val_loss'][-1]:.4f}")
        print(f"   Threshold: {self.threshold:.4f}")
        
        return True
    
    def detect_anomaly(self, new_data):
        """Detect if new data is anomalous"""
        if not self.trained or self.model is None:
            return False, 0.0, "Model not trained"
        
        seq = self.prepare_sequences(new_data)
        if seq is None or len(seq) == 0:
            return False, 0.0, "Not enough data"
        
        # Get last sequence
        last_seq = seq[-1].reshape(1, self.sequence_length, self.n_features)
        
        # Reconstruct
        reconstruction = self.model.predict(last_seq, verbose=0)
        
        # Calculate MSE
        mse = np.mean(np.power(last_seq - reconstruction, 2))
        
        # Check if anomaly
        is_anomaly = mse > self.threshold
        anomaly_score = min(mse / self.threshold, 2.0) if self.threshold > 0 else 0
        
        return is_anomaly, anomaly_score, mse


# ============================================================
# SYNTHETIC DATA GENERATOR
# ============================================================

class SyntheticDataGenerator:
    """Generate synthetic behavioral data for testing"""
    
    def __init__(self):
        self.emotions = ['happy', 'neutral', 'sad', 'angry', 'fear', 'surprise', 'disgust']
        self.emotion_scores = {
            'happy': 0.0,
            'neutral': 0.3,
            'surprise': 0.4,
            'sad': 0.6,
            'fear': 0.7,
            'angry': 0.8,
            'disgust': 0.9
        }
    
    def generate_normal_pattern(self, days=7, samples_per_day=288):
        """
        Generate normal behavior pattern (happy/neutral with good adherence)
        days: number of days
        samples_per_day: number of samples per day (e.g., 288 = every 5 minutes)
        """
        data = []
        total_samples = days * samples_per_day
        
        for i in range(total_samples):
            # Time of day (0-1)
            time_of_day = (i % samples_per_day) / samples_per_day
            
            # Normal pattern: mostly happy/neutral
            if random.random() < 0.7:
                emotion = 'happy'
            elif random.random() < 0.85:
                emotion = 'neutral'
            else:
                emotion = random.choice(['surprise', 'sad'])
            
            # Good adherence (80-100%)
            adherence = random.uniform(0.8, 1.0)
            
            # Low missed pills
            missed_pills = random.randint(0, 1)
            
            # Low stress
            stress = random.uniform(0, 0.3)
            
            # Create features
            features = [
                self.emotion_scores[emotion],
                0.7 + random.uniform(0, 0.3),  # confidence
                adherence,
                missed_pills / 5.0,
                stress
            ]
            
            data.append(features)
        
        return np.array(data)
    
    def generate_anxiety_pattern(self, days=3, samples_per_day=288):
        """
        Generate anxiety pattern (fear/sad with poor adherence)
        """
        data = []
        total_samples = days * samples_per_day
        
        for i in range(total_samples):
            # Time of day
            time_of_day = (i % samples_per_day) / samples_per_day
            
            # Anxiety pattern: mostly sad/fear
            if random.random() < 0.5:
                emotion = 'sad'
            elif random.random() < 0.75:
                emotion = 'fear'
            elif random.random() < 0.9:
                emotion = 'angry'
            else:
                emotion = random.choice(['happy', 'neutral'])
            
            # Poor adherence (40-70%)
            adherence = random.uniform(0.4, 0.7)
            
            # More missed pills
            missed_pills = random.randint(1, 3)
            
            # Higher stress
            stress = random.uniform(0.5, 0.9)
            
            features = [
                self.emotion_scores[emotion],
                0.5 + random.uniform(0, 0.3),
                adherence,
                missed_pills / 5.0,
                stress
            ]
            
            data.append(features)
        
        return np.array(data)
    
    def generate_depression_pattern(self, days=3, samples_per_day=288):
        """
        Generate depression pattern (sad/neutral with low adherence)
        """
        data = []
        total_samples = days * samples_per_day
        
        for i in range(total_samples):
            # Depression pattern: mostly sad/neutral
            if random.random() < 0.4:
                emotion = 'sad'
            elif random.random() < 0.7:
                emotion = 'neutral'
            elif random.random() < 0.85:
                emotion = 'disgust'
            else:
                emotion = 'happy'
            
            # Very poor adherence (20-50%)
            adherence = random.uniform(0.2, 0.5)
            
            # Many missed pills
            missed_pills = random.randint(2, 5)
            
            # High stress
            stress = random.uniform(0.6, 0.95)
            
            features = [
                self.emotion_scores[emotion],
                0.4 + random.uniform(0, 0.3),
                adherence,
                missed_pills / 5.0,
                stress
            ]
            
            data.append(features)
        
        return np.array(data)
    
    def generate_manic_pattern(self, days=2, samples_per_day=288):
        """
        Generate manic pattern (happy/surprise with irregular adherence)
        """
        data = []
        total_samples = days * samples_per_day
        
        for i in range(total_samples):
            # Manic pattern: mostly happy/surprise
            if random.random() < 0.5:
                emotion = 'happy'
            elif random.random() < 0.75:
                emotion = 'surprise'
            elif random.random() < 0.9:
                emotion = 'angry'
            else:
                emotion = 'neutral'
            
            # Irregular adherence (0-100%)
            adherence = random.uniform(0, 1.0)
            
            # Variable missed pills
            missed_pills = random.randint(0, 4)
            
            # Variable stress
            stress = random.uniform(0, 0.8)
            
            features = [
                self.emotion_scores[emotion],
                0.6 + random.uniform(0, 0.3),
                adherence,
                missed_pills / 5.0,
                stress
            ]
            
            data.append(features)
        
        return np.array(data)


# ============================================================
# TESTING FUNCTIONS
# ============================================================

def test_lstm_autoencoder():
    """Test the LSTM autoencoder with synthetic data"""
    
    print("\n" + "="*60)
    print("🧠 LSTM AUTOENCODER TEST WITH SYNTHETIC DATA")
    print("="*60)
    
    # Initialize
    generator = SyntheticDataGenerator()
    lstm = LSTMAutoencoder(sequence_length=20, n_features=5, latent_dim=8)
    
    # Generate normal training data (7 days)
    print("\n📊 Generating normal behavior data (7 days)...")
    normal_data = generator.generate_normal_pattern(days=7, samples_per_day=288)
    print(f"   Generated {len(normal_data)} samples")
    
    # Train the LSTM on normal data
    print("\n🔄 Training LSTM Autoencoder on normal data...")
    success = lstm.train(normal_data, epochs=30, batch_size=64)
    
    if not success:
        print("❌ Training failed")
        return
    
    # Generate test data
    print("\n📊 Generating test patterns...")
    
    # Normal pattern (should NOT trigger anomaly)
    normal_test = generator.generate_normal_pattern(days=1, samples_per_day=288)
    print(f"   Normal pattern: {len(normal_test)} samples")
    
    # Anxiety pattern (should trigger anomaly)
    anxiety_test = generator.generate_anxiety_pattern(days=1, samples_per_day=288)
    print(f"   Anxiety pattern: {len(anxiety_test)} samples")
    
    # Depression pattern (should trigger anomaly)
    depression_test = generator.generate_depression_pattern(days=1, samples_per_day=288)
    print(f"   Depression pattern: {len(depression_test)} samples")
    
    # Manic pattern (should trigger anomaly)
    manic_test = generator.generate_manic_pattern(days=1, samples_per_day=288)
    print(f"   Manic pattern: {len(manic_test)} samples")
    
    # Test each pattern
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    
    test_results = []
    
    for name, data in [
        ("Normal", normal_test),
        ("Anxiety", anxiety_test),
        ("Depression", depression_test),
        ("Manic", manic_test)
    ]:
        # Prepare data
        data_list = data.tolist()
        
        # Detect anomalies
        is_anomaly, score, mse = lstm.detect_anomaly(data_list)
        
        result = {
            'pattern': name,
            'is_anomaly': is_anomaly,
            'anomaly_score': score,
            'mse': mse
        }
        test_results.append(result)
        
        # Print result
        status = "🔴 ANOMALY" if is_anomaly else "✅ Normal"
        print(f"\n{name} Pattern:")
        print(f"   Status: {status}")
        print(f"   Anomaly Score: {score:.4f} (Threshold: {lstm.threshold:.4f})")
        print(f"   MSE: {mse:.4f}")
        if is_anomaly:
            print(f"   ⚠️ This pattern deviates from normal behavior!")
    
    return lstm, test_results


def visualize_anomaly_detection(lstm, normal_data, anomalous_data, pattern_name):
    """Visualize the anomaly detection results"""
    try:
        import matplotlib.pyplot as plt
        
        # Prepare data
        normal_seq = lstm.prepare_sequences(normal_data.tolist())
        anomaly_seq = lstm.prepare_sequences(anomalous_data.tolist())
        
        if normal_seq is None or anomaly_seq is None:
            print("⚠️ Not enough data for visualization")
            return
        
        # Get reconstructions
        normal_recon = lstm.model.predict(normal_seq, verbose=0)
        anomaly_recon = lstm.model.predict(anomaly_seq, verbose=0)
        
        # Calculate MSE per sequence
        normal_mse = np.mean(np.power(normal_seq - normal_recon, 2), axis=(1, 2))
        anomaly_mse = np.mean(np.power(anomaly_seq - anomaly_recon, 2), axis=(1, 2))
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Normal vs Anomaly MSE
        ax1 = axes[0, 0]
        ax1.hist(normal_mse, bins=30, alpha=0.7, label='Normal', color='green')
        ax1.hist(anomaly_mse, bins=30, alpha=0.7, label=pattern_name, color='red')
        ax1.axvline(lstm.threshold, color='black', linestyle='--', label=f'Threshold: {lstm.threshold:.3f}')
        ax1.set_xlabel('Reconstruction Error (MSE)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Normal vs Anomaly Reconstruction Error')
        ax1.legend()
        
        # Plot 2: Anomaly scores over time
        ax2 = axes[0, 1]
        scores = []
        for i in range(len(anomaly_mse)):
            scores.append(min(anomaly_mse[i] / lstm.threshold, 2.0) if lstm.threshold > 0 else 0)
        ax2.plot(scores, color='red', linewidth=2)
        ax2.axhline(1.0, color='black', linestyle='--', label='Threshold')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Anomaly Score')
        ax2.set_title(f'{pattern_name} - Anomaly Scores Over Time')
        ax2.legend()
        ax2.set_ylim(0, 2.2)
        
        # Plot 3: Feature patterns (first 3 features)
        ax3 = axes[1, 0]
        if len(anomaly_seq) > 0:
            sample_seq = anomaly_seq[0]
            for i in range(min(3, sample_seq.shape[1])):
                ax3.plot(sample_seq[:, i], label=f'Feature {i+1}')
            ax3.set_xlabel('Time Steps')
            ax3.set_ylabel('Feature Value')
            ax3.set_title(f'{pattern_name} - Feature Pattern')
            ax3.legend()
        
        # Plot 4: Reconstruction comparison
        ax4 = axes[1, 1]
        if len(anomaly_seq) > 0 and len(anomaly_recon) > 0:
            idx = min(5, len(anomaly_seq) - 1)
            original = anomaly_seq[idx]
            reconstructed = anomaly_recon[idx]
            
            # Plot first feature
            ax4.plot(original[:, 0], 'b-', label='Original', linewidth=2)
            ax4.plot(reconstructed[:, 0], 'r--', label='Reconstructed', linewidth=2)
            ax4.set_xlabel('Time Steps')
            ax4.set_ylabel('Feature Value')
            ax4.set_title(f'{pattern_name} - Original vs Reconstructed (Feature 1)')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'lstm_anomaly_{pattern_name.lower()}.png', dpi=150)
        print(f"📊 Visualization saved as: lstm_anomaly_{pattern_name.lower()}.png")
        plt.show()
        
    except ImportError:
        print("⚠️ Matplotlib not available for visualization")
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")


# ============================================================
# MAIN TEST
# ============================================================

def main():
    """Run the complete test"""
    
    print("\n" + "="*60)
    print("🧠 LSTM AUTOENCODER WITH SYNTHETIC DATA")
    print("="*60)
    print("\nThis test will:")
    print("1. Generate synthetic normal behavior data (7 days)")
    print("2. Train LSTM autoencoder on normal data")
    print("3. Test on different behavior patterns:")
    print("   - ✅ Normal: Should NOT trigger alarm")
    print("   - 🔴 Anxiety: Should trigger alarm")
    print("   - 🔴 Depression: Should trigger alarm")
    print("   - 🔴 Manic: Should trigger alarm")
    print("="*60)
    
    # Run the test
    lstm, results = test_lstm_autoencoder()
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    for result in results:
        status = "🔴 ALERT" if result['is_anomaly'] else "✅ OK"
        print(f"{status} {result['pattern']:12s} - Score: {result['anomaly_score']:.3f}")
    
    # Count correct detections
    correct = sum(1 for r in results if 
                  (r['pattern'] == 'Normal' and not r['is_anomaly']) or
                  (r['pattern'] != 'Normal' and r['is_anomaly']))
    
    print(f"\n✅ {correct}/{len(results)} patterns correctly identified!")
    
    # Optionally visualize
    try:
        visualize = input("\n📊 Visualize results? (y/n): ").lower()
        if visualize == 'y':
            # Generate data for visualization
            generator = SyntheticDataGenerator()
            normal_data = generator.generate_normal_pattern(days=1, samples_per_day=288)
            anomalous_data = generator.generate_anxiety_pattern(days=1, samples_per_day=288)
            visualize_anomaly_detection(lstm, normal_data, anomalous_data, "Anxiety")
    except:
        pass
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()