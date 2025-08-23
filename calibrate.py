# calibrate.py
import os
import joblib
import numpy as np
import pandas as pd
import warnings

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from preprocess import transform_features  # existing pipeline


warnings.filterwarnings(
    "ignore",
    message=".*does not have valid feature names, but StandardScaler was fitted with feature names.*",
)

BASE_MODEL_PATH = os.path.join("models", "xgboost_model.pkl")
CAL_MODEL_PATH  = os.path.join("models", "xgb_calibrated.pkl")
DATA_PATH       = os.path.join("data", "FinalCropIrrigationScheduling.csv")

def build_matrix_from_dataset(df: pd.DataFrame):
    """
    Use your existing transform_features() so the features match the app exactly.
    Returns X (scaled features) and y (labels).
    """
    X_rows = []
    y_rows = []

    
    for _, r in df.iterrows():
        crop_type     = str(r["CropType"])
        crop_days     = int(r["CropDays"])
        soil_moisture = float(r["SoilMoisture"])
        temperature   = float(r["temperature"])
        humidity      = float(r["Humidity"])
        label         = int(r["Irrigation"])  # 0/1

        x_scaled, _ = transform_features(crop_type, crop_days, soil_moisture, temperature, humidity)
        X_rows.append(x_scaled[0])
        y_rows.append(label)

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=int)
    return X, y

def make_calibrator_prefit(base_model, X_val, y_val):
    """
    Create a CalibratedClassifierCV in a version-safe way:
    - Newer sklearn: uses `estimator=`
    - Older sklearn: falls back to `base_estimator=`
    """
    try:
        # Newer API
        cal = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv="prefit")
    except TypeError:
        # Older API fallback
        cal = CalibratedClassifierCV(base_estimator=base_model, method="isotonic", cv="prefit")
    cal.fit(X_val, y_val)
    return cal

def main():
    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL_PATH}")

    base_model = joblib.load(BASE_MODEL_PATH)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).dropna()

    # Building with existing pipeline
    X, y = build_matrix_from_dataset(df)

    # Hold out a validation set for calibration
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # Fit only the calibrator; base model remains untouched
    cal = make_calibrator_prefit(base_model, X_val, y_val)

    os.makedirs(os.path.dirname(CAL_MODEL_PATH), exist_ok=True)
    joblib.dump(cal, CAL_MODEL_PATH)
    print(f"Saved calibrated probability model to: {CAL_MODEL_PATH}")

if __name__ == "__main__":
    main()
