import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

from preprocess import transform_features
from shap_explain import explain_prediction 

# === Load model ===
MODEL_DIR = "models"
model = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))

# === User Input ===
print("🌾 === Smart Irrigation CLI Prediction ===")
crop_type = input("Enter Crop Type (e.g., Wheat, Paddy, Maize): ").strip()
crop_days = int(input("Enter Days Since Planting: "))
raw_soil_moisture = float(input("Enter Soil Moisture Sensor Value (Raw): "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Relative Humidity (%): "))

# === Transform Features ===
input_scaled, intermediates = transform_features(
    crop_type, crop_days, raw_soil_moisture, temperature, humidity
)

# === Predict and Confidence Score ===
prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0]
confidence = max(proba)
confidence_percent = f"{confidence * 100:.2f}"

result = " Irrigation Required" if prediction == 1 else "💧 No Irrigation Required"

# === Output Result ===
print("\n===  Prediction Result ===")
print(f"Crop Type: {crop_type}")
print(f"Crop Stage: {intermediates['CropStage']}")
print(f"Soil Moisture (%): {intermediates['SoilMoisturePercent']:.2f}")
print(f"VPD: {intermediates['VPD']:.2f}")
print(f"Prediction: {result}")
print(f"🔒 Confidence Score: {confidence_percent}%")

# === Feature Names for SHAP ===
feature_names = [
    "Crop Type",
    "Days Since Planting",
    "Soil Moisture (%)",
    "Temperature (°C)",
    "Humidity (%)",
    "VPD",
    "Crop Stage"
]

# === SHAP Explainability ===
explain_prediction(input_scaled, feature_names, intermediates)
