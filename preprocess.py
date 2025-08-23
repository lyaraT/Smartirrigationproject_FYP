import numpy as np
import os
import joblib
import pandas as pd

# Load encoders and scalers 
LE_CROP_PATH = os.path.join("models", "le_crop.pkl")
LE_STAGE_PATH = os.path.join("models", "le_stage.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
SM_MIN_PATH = os.path.join("models", "sm_min.pkl")
SM_MAX_PATH = os.path.join("models", "sm_max.pkl")

le_crop = joblib.load(LE_CROP_PATH)
le_stage = joblib.load(LE_STAGE_PATH)
scaler = joblib.load(SCALER_PATH)
sm_min = joblib.load(SM_MIN_PATH)
sm_max = joblib.load(SM_MAX_PATH)

# Growth Stage Mapping 
stage_mapping = {
    'Wheat':        [(0, 20, 'Germination'), (21, 60, 'Vegetative'), (61, 90, 'Reproductive'), (91, float('inf'), 'Maturity')],
    'Groundnuts':   [(0, 20, 'Germination'), (21, 40, 'Vegetative'), (41, 90, 'Reproductive'), (91, float('inf'), 'Maturity')],
    'Garden Flowers': [(0, 15, 'Germination'), (16, 45, 'Vegetative'), (46, 75, 'Reproductive'), (76, float('inf'), 'Maturity')],
    'Maize':        [(0, 15, 'Germination'), (16, 45, 'Vegetative'), (46, 75, 'Reproductive'), (76, float('inf'), 'Maturity')],
    'Paddy':        [(0, 15, 'Germination'), (16, 45, 'Vegetative'), (46, 80, 'Reproductive'), (81, float('inf'), 'Maturity')],
    'Potato':       [(0, 20, 'Germination'), (21, 45, 'Vegetative'), (46, 90, 'Reproductive'), (91, float('inf'), 'Maturity')],
    'Pulse':        [(0, 15, 'Germination'), (16, 40, 'Vegetative'), (41, 70, 'Reproductive'), (71, float('inf'), 'Maturity')],
    'Sugarcane':    [(0, 30, 'Germination'), (31, 120, 'Vegetative'), (121, 240, 'Reproductive'), (241, float('inf'), 'Maturity')],
    'Coffee':       [(0, 90, 'Germination'), (91, 270, 'Vegetative'), (271, float('inf'), 'Reproductive')],
}

def get_crop_stage(crop_type, crop_days):
    for start, end, label in stage_mapping.get(crop_type, []):
        if start <= crop_days <= end:
            return label
    return 'Unknown'

def calculate_vpd(temperature, humidity):
    return temperature * (1 - humidity / 100)

def transform_features(crop_type, crop_days, soil_moisture, temperature, humidity):
    """Returns a scaled feature vector for inference (7 features only)."""
    crop_stage = get_crop_stage(crop_type, crop_days)
    vpd = calculate_vpd(temperature, humidity)
    soil_percent = 100 * (soil_moisture - sm_min) / (sm_max - sm_min)

    crop_code = le_crop.transform([crop_type])[0]
    stage_code = le_stage.transform([crop_stage])[0]

    
    features = np.array([[crop_code, crop_days, soil_percent, temperature,
                          humidity, vpd, stage_code]])

    scaled_features = scaler.transform(features)

    return scaled_features, {
        "CropStage": crop_stage,
        "VPD": round(vpd, 2),
        "SoilMoisturePercent": round(soil_percent, 2)
    }

