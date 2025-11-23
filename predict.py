import numpy as np
import joblib
import tensorflow as tf
import os

# Paths
MODEL_PATH = "models/alzheimer_model.keras"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/encoder.pkl"

def predict_sample(sample_data):
    """
    Loads artifacts and predicts class for a single sample.
    Args:
        sample_data (list or np.array): Raw numeric features for one patient.
    """
    try:
        # 1. Load Artifacts
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)

        # 2. Preprocess Input
        # Reshape to (1, -1) because scaler expects 2D array
        sample_reshaped = np.array(sample_data).reshape(1, -1)
        sample_scaled = scaler.transform(sample_reshaped)

        # 3. Inference
        prediction_prob = model.predict(sample_scaled, verbose=0)[0][0]
        prediction_class = 1 if prediction_prob > 0.5 else 0
        
        # Decode label (e.g., 0 -> 'Healthy', 1 -> 'Alzheimers')
        # We handle cases where encoder might not have string labels
        try:
            label_str = encoder.inverse_transform([prediction_class])[0]
        except:
            label_str = "Alzheimer's" if prediction_class == 1 else "Healthy"

        return {
            "label": label_str,
            "probability": float(prediction_prob),
            "risk_level": "High" if prediction_prob > 0.5 else "Low"
        }

    except FileNotFoundError:
        return "Error: Model artifacts not found. Run train.py first."

if __name__ == "__main__":
    # Example: Create a dummy random sample matching the feature size
    # In a real app, this would come from a frontend or API
    print("Running dummy prediction...")
    
    # Note: We need to know the feature size. 
    # For this demo, we catch the shape mismatch error if it occurs, 
    # but ideally, we check the scaler's expected features.
    scaler = joblib.load(SCALER_PATH)
    n_features = scaler.n_features_in_
    
    dummy_input = np.random.rand(n_features)
    
    result = predict_sample(dummy_input)
    print(f"Prediction Result: {result}")
