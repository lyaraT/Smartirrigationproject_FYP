import os
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = os.path.join("models", "xgboost_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def _humanize_explanation(feature_name: str, direction: str, info: dict) -> str:
    
    crop = info.get("crop_type")
    stage = info.get("CropStage")
    sm = info.get("SoilMoisturePercent")
    vpd = info.get("VPD")
    temp = info.get("temperature")
    rh = info.get("humidity")

    if direction == "increased":
        phrase = "makes irrigation more likely"
    else:
        phrase = "reduces the need for irrigation"

    if feature_name == "Soil Moisture (%)":
        if sm is not None:
            return (
                f"Soil moisture is about {sm:.1f}%. Under the present conditions, "
                f"this {phrase}."
            )
        return f"Current soil moisture {phrase}."

    if feature_name == "Crop Type":
        if crop:
            return (
                f"Given the present conditions and the crop being {crop}, "
                f"it {phrase}."
            )
        return f"The crop type {phrase} under the current conditions."

    if feature_name == "Crop Stage":
        if stage:
            return (
                f"Because the crop is in the {stage} stage under today’s conditions, "
                f"it {phrase}."
            )
        return f"At this growth stage, the crop {phrase}."

    if feature_name == "Temperature (°C)":
        if temp is not None:
            return (
                f"Temperature is around {temp:.0f}°C. At this temperature, "
                f"the crop’s water demand {phrase}."
            )
        return f"The current temperature {phrase}."

    if feature_name == "Humidity (%)":
        if rh is not None:
            return (
                f"Relative humidity is about {rh:.0f}%. Under these conditions, "
                f"the crop’s water loss {phrase}."
            )
        return f"The current humidity {phrase}."

    if feature_name == "VPD":
        if vpd is not None:
            return (
                f"VPD is {vpd:.2f}. With this air dryness, crop water demand "
                f"{phrase}."
            )
        return f"The air dryness (VPD) {phrase}."

    if feature_name == "Days Since Planting":
        return (
            f"At this crop age, the growth stage water demand {phrase}."
        )

    return f"This factor {phrase} under the present conditions."


def explain_prediction(input_scaled, feature_names, intermediates=None, return_text=False):
  
    explainer = shap.Explainer(model, feature_names=feature_names)
    shap_values = explainer(input_scaled)
    shap_vals = shap_values.values[0]

    # Identify Top 3 features
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    labels = ["Most", "Second most", "Third most"]

    explanations = []

    
    info = intermediates or {}

    for rank, i in enumerate(top_idx):
        name = feature_names[i]
        val = shap_vals[i]
        direction = "increased" if val > 0 else "reduced"
        sign = "+" if val > 0 else "−"
        title = f"{labels[rank]} influential factor: {name}"

        explanation = _humanize_explanation(name, direction, info)

        impact = f"{direction} irrigation probability by {sign}{abs(val):.2f}"
        explanations.append((title, explanation, impact))

    # Determine overall conclusion
    if intermediates and "prediction" in intermediates:
        prediction = intermediates["prediction"]  # 1 = Irrigation Required, 0 = Not Required
    else:
        prediction = 1 if np.sum(shap_vals) > 0 else 0

    top_directions = ["increased" if shap_vals[i] > 0 else "reduced" for i in top_idx]

    if prediction == 1:  # Irrigation Required
        if top_directions.count("reduced") >= 2:
            conclusion = (
                f"Although {explanations[0][0].split(': ')[1]} was the top contributor suggesting irrigation, "
                f"the other two key features reduced irrigation probability. "
                f"However, their impact was not enough to offset the top factor, so irrigation is required."
            )
        else:
            conclusion = "Overall, the combination of crop and environmental factors strongly indicates irrigation is required."
    else:  # No Irrigation Required
        if top_directions[0] == "increased" and top_directions.count("reduced") >= 2:
            conclusion = (
                f"Although {explanations[0][0].split(': ')[1]} was the top contributor suggesting irrigation, "
                f"the second and third most influential factors reduced irrigation probability enough to confirm no irrigation is required."
            )
        else:
            conclusion = "Overall, the combination of crop and environmental conditions confirms irrigation is not required at this time."

    # Plotting ALL features
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
    sorted_vals = shap_vals[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    colors = ["red" if val > 0 else "green" for val in sorted_vals]

    plt.figure(figsize=(8, 4))
    plt.barh(sorted_names, sorted_vals, color=colors)
    plt.xlabel("Contribution to Irrigation Prediction")
    plt.title("Top Feature Contributions")
    plt.axvline(x=0, color='black', linestyle='--')
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()

    plot_path = "static/plots/shap_bar.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

    return explanations, plot_path, conclusion
