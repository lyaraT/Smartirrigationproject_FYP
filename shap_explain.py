import os
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = os.path.join("models", "xgboost_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def explain_prediction(input_scaled, feature_names, intermediates=None, return_text=False):
    """
    Returns:
    - explanation_list: [(title, explanation, impact_str), ...] (top 3 only)
    - plot_path: path to SHAP feature contribution bar plot
    - conclusion: overall reasoning summary
    """
    explainer = shap.Explainer(model, feature_names=feature_names)
    shap_values = explainer(input_scaled)
    shap_vals = shap_values.values[0]

    # Identify Top 3 features
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    labels = ["Most", "Second most", "Third most"]

    explanations = []

    def interpret_feature(name, direction):
        explanation_map = {
            "Crop Type": {
                "increased": "This crop type generally requires more frequent irrigation for healthy growth.",
                "reduced": "This crop type is drought-tolerant, meaning it can thrive with less frequent irrigation."
            },
            "VPD": {
                "increased": "There is high vapour pressure deficit, meaning the air is dry and plants lose water faster.",
                "reduced": "There is low vapour pressure deficit, meaning the air is humid and plants lose water slowly."
            },
            "Soil Moisture (%)": {
                "increased": "There is low soil moisture, meaning roots may not be getting enough water to sustain growth.",
                "reduced": "There is adequate soil moisture, meaning plants have enough available water in the root zone."
            },
            "Temperature (°C)": {
                "increased": "There is high temperature, which increases evapotranspiration and water demand.",
                "reduced": "There is cool temperature, which slows down evapotranspiration and reduces water demand."
            },
            "Humidity (%)": {
                "increased": "There is low humidity, which increases water loss through plant transpiration.",
                "reduced": "There is high humidity, which helps plants retain moisture and reduces water loss."
            },
            "Crop Stage": {
                "increased": "The crop is in a stage such as flowering or fruiting, which generally demands more water.",
                "reduced": "The crop is in a stage that typically requires less water compared to peak growth phases."
            },
            "Days Since Planting": {
                "increased": "There are many days since planting, meaning the crop is mature and may require more water.",
                "reduced": "There are few days since planting, meaning the crop is young and its water needs are lower."
            }
        }
        return explanation_map.get(name, {}).get(direction, "This factor influenced the irrigation recommendation.")

    for rank, i in enumerate(top_idx):
        name = feature_names[i]
        val = shap_vals[i]
        direction = "increased" if val > 0 else "reduced"
        sign = "+" if val > 0 else "−"
        title = f"{labels[rank]} influential factor: {name}"
        explanation = interpret_feature(name, direction)
        impact = f"{direction} irrigation probability by {sign}{abs(val):.2f}"
        explanations.append((title, explanation, impact))

    #  Determine overall conclusion 
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
