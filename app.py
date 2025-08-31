# app.py
import gradio as gr
import os
import joblib
import numpy as np
from PIL import Image
from preprocess import transform_features
from shap_explain import explain_prediction


BASE_MODEL_PATH = os.path.join("models", "xgboost_model.pkl")
CAL_MODEL_PATH  = os.path.join("models", "xgb_calibrated.pkl")

model = joblib.load(BASE_MODEL_PATH)

calibrated_model = None
if os.path.exists(CAL_MODEL_PATH):
    try:
        calibrated_model = joblib.load(CAL_MODEL_PATH)
        print("Loaded calibrated probability model:", CAL_MODEL_PATH)
    except Exception as e:
        print("Warning: could not load calibrated model. Falling back to base proba. Error:", e)


CROP_MAX_DAYS = {
    "Wheat": 150, "Groundnuts": 150, "Garden Flowers": 120, "Maize": 120,
    "Paddy": 150, "Potato": 120, "Pulse": 120, "Sugarcane": 365, "Coffee": 365,
}

def get_image_path(crop_type, crop_stage):
    base_name = f"{crop_type}_{crop_stage}".replace(" ", "_")
    for ext in [".jpg", ".png", ".jpeg", ".webp"]:
        path = os.path.join("static", "images", base_name + ext)
        if os.path.exists(path):
            return path
    return None


def _format_confidence(proba_arr):
    if proba_arr is None or len(proba_arr) == 0:
        return "—"
    confidence = float(np.max(proba_arr))
    return f"{confidence * 100:.2f}%"

def predict_and_explain(crop_type, crop_days, soil_moisture, temperature, humidity):
    input_scaled, intermediates = transform_features(
        crop_type, crop_days, soil_moisture, temperature, humidity
    )

    prediction = model.predict(input_scaled)[0]

    try:
        if calibrated_model is not None:
            proba = calibrated_model.predict_proba(input_scaled)[0]
        else:
            proba = model.predict_proba(input_scaled)[0]
    except Exception as e:
        print("Probability computation failed:", e)
        proba = np.array([0.5, 0.5], dtype=float)

    confidence_percent = _format_confidence(proba)

    result_text  = "Irrigation Required" if prediction == 1 else "No Irrigation Required"
    result_color = "#E53935" if prediction == 1 else "#2E7D32"
    icon         = "💧" if prediction == 1 else "✅"

    crop_stage = intermediates["CropStage"]
    image_path = get_image_path(crop_type, crop_stage)
    image = Image.open(image_path) if image_path else None

    shap_factors, plot_path, conclusion = explain_prediction(
        input_scaled=input_scaled,
        feature_names=[
            "Crop Type","Days Since Planting","Soil Moisture (%)",
            "Temperature (°C)","Humidity (%)","VPD","Crop Stage",
        ],
        intermediates={"CropStage": crop_stage, "prediction": prediction},
        return_text=False,
    )

    caption_html = f"""
    <div class="media-header">
      <span class="media-chip">Crop: <b>{crop_type}</b></span>
      <span class="media-dot">•</span>
      <span class="media-chip neutral">Stage: <b>{crop_stage}</b></span>
    </div>
    """

    prediction_html = f"""
    <div style="
        background:#fff; border-radius:14px; padding:18px 20px;
        box-shadow:0 6px 16px rgba(0,0,0,.08); border:1px solid #E7F0E9;
        border-top:5px solid {result_color};
        font-family:'Segoe UI', sans-serif;
    ">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="font-size:22px">{icon}</div>
          <div style="font-weight:800; font-size:20px; color:{result_color};">{result_text}</div>
        </div>
        <div style="background:{result_color}; color:#fff; padding:8px 14px; border-radius:20px; font-size:13px; font-weight:700;">
          {confidence_percent} confident
        </div>
      </div>

      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-top:6px;">
        <div style="background:#F8FAF8; padding:10px 12px; border-radius:8px;">
          <div style="font-size:12px; color:#667; margin-bottom:4px;">Crop Type</div>
          <div style="font-weight:600; font-size:15px; color:#223;">{crop_type}</div>
        </div>
        <div style="background:#F8FAF8; padding:10px 12px; border-radius:8px;">
          <div style="font-size:12px; color:#667; margin-bottom:4px;">Growth Stage</div>
          <div style="font-weight:600; font-size:15px; color:#223;">{crop_stage}</div>
        </div>
        <div style="background:#F8FAF8; padding:10px 12px; border-radius:8px;">
          <div style="font-size:12px; color:#667; margin-bottom:4px;">Soil Moisture (%)</div>
          <div style="font-weight:600; font-size:15px; color:#223;">{intermediates['SoilMoisturePercent']:.2f}</div>
        </div>
        <div style="background:#F8FAF8; padding:10px 12px; border-radius:8px;">
          <div style="font-size:12px; color:#667; margin-bottom:4px;">VPD</div>
          <div style="font-weight:600; font-size:15px; color:#223;">{intermediates['VPD']:.2f}</div>
        </div>
      </div>
    </div>
    """

    explanation_items = ""
    for i, (title, explanation, impact) in enumerate(shap_factors, 1):
        color = "#E53935" if "increased" in impact else "#2E7D32"
        explanation_items += f"""
        <div style="border-bottom: 1px solid #EEF3EF; padding: 12px 0;">
          <div style="display: flex; align-items: flex-start; gap: 12px;">
            <div style="background-color:{color}; color:white; font-weight:700; border-radius:50%; width:28px; height:28px; display:flex; align-items:center; justify-content:center;">{i}</div>
            <div>
              <div style="font-size: 16px; font-weight: 700; color:#2F4F4F;">{title}</div>
              <div style="font-size: 14px; color:#4C6250; margin-top:2px;">{explanation}</div>
              <div style="margin-top:6px;">
                <span style="background-color:{color}; color:white; padding: 4px 10px; border-radius: 6px; font-size: 12.5px; font-weight:700;">
                  SHAP Contribution: {impact}
                </span>
              </div>
            </div>
          </div>
        </div>
        """

    explanation_html = f"""
    <div class="explanation-card">
      <h4 style="font-size: 18px; font-weight: 800; margin: 0 0 10px 0; color:#2F4F4F;">
        Top 3 features that contributed to the prediction
      </h4>
      {explanation_items}
      <p style="margin-top: 14px; font-weight: 700; color:#2F4F4F;">Overall conclusion</p>
      <p style="font-size: 14px; color:#4C6250; margin: 0 0 8px 0;">{conclusion}</p>
      <p style="font-style: italic; margin: 6px 0 0 0; font-size: 12.5px; color:#6C7E70;">
        Note: Higher absolute SHAP values mean greater influence on the decision.
      </p>
    </div>
    """

    return caption_html, image, prediction_html, plot_path, explanation_html


def update_days_slider(selected_crop, current_days):
    max_days = CROP_MAX_DAYS.get(selected_crop, 150)
    try:
        cur = float(current_days) if current_days not in (None, "") else 0.0
    except Exception:
        cur = 0.0
    new_value = min(max(cur, 0.0), float(max_days))
    return gr.update(minimum=0, maximum=int(max_days), value=int(new_value))


with gr.Blocks(theme=gr.themes.Default(primary_hue="green", secondary_hue="gray")) as demo:
    # Global styling 
    gr.HTML("""
    <style>
      .gradio-container {
        /* AWS-like soft diagonal gradient */
        background: linear-gradient(135deg, #d4fdd6 0%, #f8fff9 60%, #ffffff 100%) !important;
      }

      .gradio-container .gr-block,
      .gradio-container .gr-panel,
      .gradio-container .gr-box,
      .gradio-container .form { background: transparent !important; box-shadow: none !important; border: none !important; }
      .gradio-container .gr-row { border: none !important; }
      .gradio-container h3, .gradio-container h4 { color: #2F4F4F; }
      button[role="tab"][aria-selected="true"] { color:#2E7D32 !important; }
      .tabs > div[style*="height: 2px"] { background:#A5D6A7 !important; }

      .explanation-card {
        background-color: #ffffff !important;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.06);
        border: 1px solid #E7F0E9;
      }

      /* hero */
      .hero { position: relative; padding: 22px 24px; border-radius: 18px; background: linear-gradient(135deg, #F1F8E9 0%, #FAFFFB 65%); border: 1px solid #DCEFE0; box-shadow: 0 10px 24px rgba(46,125,50,.10); margin-bottom: 16px; overflow: hidden; display:flex; align-items:center; justify-content:center; text-align:center; }
      .hero::after{ content:""; position:absolute; right:-80px; top:-80px; width:280px; height:280px; border-radius:50%; background: radial-gradient(closest-side, rgba(165,214,167,.32), rgba(165,214,167,0)); filter: blur(2px); }
      .hero-content{ position:relative; z-index:1; max-width:900px; }
      .hero-icon { width:52px; height:52px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-size:28px; background:#E8F5E9; border:1px solid #CDE7D0; box-shadow: inset 0 0 0 5px #F5FBF6; margin-bottom:10px; }
      .hero-title{ margin:2px 0 4px 0; font-size:28px; font-weight:900; letter-spacing:.2px; line-height:1.15; color:#2E7D32; }
      .hero-sub{ margin:0 auto; color:#3F5A45; font-size:14px; max-width:660px; }
      .hero-accent{ width:70px; height:3px; margin:10px auto 0 auto; background:#66BB6A; border-radius:999px; box-shadow:0 2px 6px rgba(102,187,106,.45); }

      /* media chips */
      .media-header{ display:flex; align-items:center; gap:10px; margin: 6px 0 12px 0; padding: 6px 10px; background: #F2F9F2; border: 1px solid #E2F0E4; border-radius: 10px; width: fit-content; color:#2F4F4F; }
      .media-chip{ font-size: 13px; color:#2F4F4F; padding:4px 10px; border-radius:999px; background:#FFFFFF; border:1px solid #E6EFE8; font-weight:600; }
      .media-chip.neutral{ border-color:#E6EFE8; color:#3D5845; }
      .media-dot{ color:#7FA88A; }

      /* image card */
      #crop-image{ padding: 12px; background: linear-gradient(180deg,#FFFFFF 0%, #F6FBF6 100%); border: 1px solid #E7F0E9; border-radius: 16px; box-shadow: 0 6px 16px rgba(0,0,0,.06); }
      #crop-image img{ width: 100%; height: 100%; object-fit: cover; border-radius: 12px; display:block; transition: transform .35s ease; }
      #crop-image:hover img{ transform: scale(1.02); }
      #crop-image .label{ display:none !important; }

      /* explicit spacing element to control the gap above the footer */
      .premium-spacer { height: 14px; }

      /* premium footer */
      .footer{
        margin-top: 0;
        padding: 18px 0;
        text-align: center;
        font-size: 13.5px;
        font-weight: 500;
        color: #4C6250;
        border-top: 1px solid #E2F0E4;
        background: linear-gradient(90deg, #F6FBF6 0%, #FAFFFB 100%);
      }

      /* floating toast: fixed to viewport so user sees it anywhere */
      .floating-toast{
        position: fixed;
        top: 14px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background: #F2FBF3;
        color: #235C2A;
        border: 1px solid #DCEFE0;
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 13.5px;
        font-weight: 600;
        box-shadow: 0 6px 14px rgba(0,0,0,.08);
        opacity: 0;
        transition: opacity .25s ease, transform .25s ease;
      }
      .floating-toast.neutral{
        background: #F7F9F8;
        color: #3F5A45;
        border-color: #E2F0E4;
      }
      .floating-toast.show{
        opacity: 1;
        transform: translateX(-50%) translateY(0);
      }
    </style>

    <section class="hero">
      <div class="hero-content">
        <div class="hero-icon">🌿</div>
        <h1 class="hero-title">Smart Irrigation Assistant</h1>
        <p class="hero-sub">Get intelligent irrigation recommendations based on your crop and environmental data.</p>
        <div class="hero-accent"></div>
      </div>
    </section>
    """)

    with gr.Tabs():
        with gr.TabItem("Recommendation", id=0):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown("### 🌱 Input Crop & Environmental Data")

                    crop_type = gr.Dropdown(
                        choices=list(CROP_MAX_DAYS.keys()),
                        label="Crop Type",
                        value="Wheat",
                    )
                    crop_days = gr.Slider(0, CROP_MAX_DAYS["Wheat"], step=1, label="Days Since Planting", value=0)
                    soil_moisture = gr.Slider(100, 800, step=1, label="Soil Moisture Sensor Value (Raw)")
                    temperature = gr.Slider(5, 45, step=1, label="Temperature (°C)")
                    humidity = gr.Slider(0, 100, step=1, label="Relative Humidity (%)")

                    analyze_btn = gr.Button("Analyze", elem_id="analyze-btn", elem_classes="submit-btn")
                    clear_btn = gr.Button("Clear", elem_classes="clear-btn")

                with gr.Column(scale=1):
                    image_caption = gr.HTML()
                    image_output = gr.Image(label="Crop Image", type="pil", show_label=False, height=300, elem_id="crop-image")
                    prediction_output = gr.HTML()

        with gr.TabItem("Explainability", id=1):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    waterfall_plot = gr.Image(label="Top Feature Contributions", type="filepath")
                with gr.Column(scale=1):
                    explanation_block = gr.HTML()

    gr.HTML('<div class="premium-spacer"></div>')
    gr.HTML("""
    <div class="footer">
      Powered by Smart Irrigation AI — Delivering Intelligent Irrigation Insights
    </div>
    """)

    
    crop_type.change(fn=update_days_slider, inputs=[crop_type, crop_days], outputs=[crop_days])

    
    analyze = analyze_btn.click(
        fn=predict_and_explain,
        inputs=[crop_type, crop_days, soil_moisture, temperature, humidity],
        outputs=[image_caption, image_output, prediction_output, waterfall_plot, explanation_block],
    )
    analyze.then(
        fn=None,
        js="""
        () => {
          const msg = "Analysis completed successfully.";
          const toast = document.createElement('div');
          toast.className = 'floating-toast show';
          toast.textContent = msg;
          document.body.appendChild(toast);
          setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => { toast.remove(); }, 300);
          }, 4000);
        }
        """
    )

    
    clear = clear_btn.click(
        fn=lambda: ("", None, "", None, ""),
        inputs=[],
        outputs=[image_caption, image_output, prediction_output, waterfall_plot, explanation_block],
    )
    clear.then(
        fn=None,
        js="""
        () => {
          const msg = "Cleared successfully.";
          const toast = document.createElement('div');
          toast.className = 'floating-toast neutral show';
          toast.textContent = msg;
          document.body.appendChild(toast);
          setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => { toast.remove(); }, 300);
          }, 4000);
        }
        """
    )

if __name__ == "__main__":
    demo.launch()
