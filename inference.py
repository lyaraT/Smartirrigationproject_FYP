# inference.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)
from xgboost import XGBClassifier

# ----------------------------
# Paths
# ----------------------------
LE_CROP_PATH  = os.path.join("models", "le_crop.pkl")
LE_STAGE_PATH = os.path.join("models", "le_stage.pkl")
SCALER_PATH   = os.path.join("models", "scaler.pkl")
SM_MIN_PATH   = os.path.join("models", "sm_min.pkl")
SM_MAX_PATH   = os.path.join("models", "sm_max.pkl")
MODEL_PATH    = os.path.join("models", "xgboost_model.pkl")
DATA_PATH     = os.path.join("data",   "FinalCropIrrigationScheduling.csv")
PLOTS_DIR     = os.path.join("static", "plots")

# ----------------------------
# Load artifacts
# ----------------------------
le_crop = joblib.load(LE_CROP_PATH)
le_stage = joblib.load(LE_STAGE_PATH)
scaler = joblib.load(SCALER_PATH)
sm_min = joblib.load(SM_MIN_PATH)
sm_max = joblib.load(SM_MAX_PATH)

# ----------------------------
# Helper functions (match training-time logic)
# ----------------------------
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
    """Single-sample transform (used by UI); not required for batch eval below."""
    crop_stage = get_crop_stage(crop_type, crop_days)
    vpd = calculate_vpd(temperature, humidity)
    soil_percent = 100 * (soil_moisture - sm_min) / (sm_max - sm_min)

    crop_code  = le_crop.transform([crop_type])[0]
    stage_code = le_stage.transform([crop_stage])[0]

    feats = np.array([[crop_code, crop_days, soil_percent, temperature,
                       humidity, vpd, stage_code]])
    feats_scaled = scaler.transform(feats)
    return feats_scaled, {"CropStage": crop_stage, "VPD": round(vpd, 2), "SoilMoisturePercent": round(soil_percent, 2)}

# ----------------------------
# Script entry
# ----------------------------
if __name__ == "__main__":
    # Checks & setup
    if not os.path.exists(DATA_PATH):  raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load data & model
    df = pd.read_csv(DATA_PATH)
    model: XGBClassifier = joblib.load(MODEL_PATH)

    # Reproduce training-time feature engineering (vectorized)
    df["CropStage"] = df.apply(lambda r: get_crop_stage(r["CropType"], r["CropDays"]), axis=1)
    df["VPD"] = df.apply(lambda r: calculate_vpd(r["temperature"], r["Humidity"]), axis=1)
    df["CropType_Code"]  = le_crop.transform(df["CropType"])
    df["CropStage_Code"] = le_stage.transform(df["CropStage"])
    df["SoilMoisturePercent"] = 100 * (df["SoilMoisture"] - sm_min) / (sm_max - sm_min)

    FEATURES = ["CropType_Code", "CropDays", "SoilMoisturePercent",
                "temperature", "Humidity", "VPD", "CropStage_Code"]
    TARGET = "Irrigation"

    X = df[FEATURES]
    y = df[TARGET]

    # ---------------- Hold-out split (stratified) ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Use the same fitted scaler from training
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ---- Metrics
    print("=== Overfitting Check ===")
    print("Train Accuracy:", accuracy_score(y_train, model.predict(X_train_scaled)))

    y_pred = model.predict(X_test_scaled)
    print("\n=== Hold-Out Test Set ===")
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Report:\n", classification_report(y_test, y_pred))

    # ---- Confusion Matrix (plot + save)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False,
                xticklabels=['No Irrigation', 'Irrigation'],
                yticklabels=['No Irrigation', 'Irrigation'])
    plt.title("XGBoost Confusion Matrix (Hold-out)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=200)
    plt.show()

    # ---- ROC Curve (plot + save)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Hold-out)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=200)
    plt.show()

    print(f"\nSaved plots to: {os.path.abspath(PLOTS_DIR)}")

    # ---------------- Stratified 5-Fold CV (leakage-safe via Pipeline) ----------------
    print("\n=== Stratified 5-Fold CV (confirmed params; leakage-safe) ===")
    tuned_params = dict(
        colsample_bytree=0.7,
        gamma=0.3,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=2,
        n_estimators=100,
        reg_lambda=1.0,
        subsample=0.8,
        scale_pos_weight=0.68,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

    cv_pipe = Pipeline([
        ("scaler", StandardScaler()),      # scaler is fit inside each fold (no leakage)
        ("clf", XGBClassifier(**tuned_params)),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(cv_pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    cv_auc = cross_val_score(cv_pipe, X, y, cv=skf, scoring="roc_auc",  n_jobs=-1)

    print("\n=== Stratified 5-Fold Cross-Validation  ===")
    print("Accuracy per fold:", np.round(cv_acc, 4))
    print("Accuracy (mean ± std): {:.4f} ± {:.4f}".format(cv_acc.mean(), cv_acc.std()))
    print("ROC AUC per fold:", np.round(cv_auc, 4))
    print("ROC AUC (mean ± std): {:.4f} ± {:.4f}".format(cv_auc.mean(), cv_auc.std()))
