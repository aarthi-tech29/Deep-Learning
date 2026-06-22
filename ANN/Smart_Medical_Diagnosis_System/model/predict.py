import numpy as np
import joblib

from tensorflow.keras.models import load_model

# Load Model
model = load_model(
    "model/disease_model.h5"
)

# Load Scaler
scaler = joblib.load(
    "model/scaler.pkl"
)

# Load Label Encoder
encoder = joblib.load(
    "model/label_encoder.pkl"
)

# Sample Patient Data
patient = np.array([[
    140,  # Blood Pressure
    180,  # Sugar Level
    95,   # Heart Rate
    96,   # Oxygen Level
    2     # Symptoms Encoded Value
]])

# Scale Input
patient = scaler.transform(patient)

# Predict
prediction = model.predict(patient)

predicted_class = np.argmax(prediction)

disease = encoder.inverse_transform(
    [predicted_class]
)

print("Predicted Disease:", disease[0])