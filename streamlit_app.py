"""
Phase 8 - Streamlit user interface.

Farmer-facing web app for multimodal crop disease prediction.
Matches the "Unit 2 - Web app" diagram in the project design:
  - Live sensor card (polls Flask /latest_sensor every 5s)
  - Leaf image upload
  - Result panel with disease, confidence, Grad-CAM, and treatment advice

IMPORTANT: uploaded image bytes AND the prediction result are stored in
st.session_state, so the 5-second sensor auto-refresh does NOT wipe the
result panel. Streamlit reruns the whole script on every refresh; session
state is how we keep things alive across reruns.

The UI is a pure HTTP client - it talks to the Flask server at
FLASK_URL for both sensor readings and predictions. Keep the Flask
server running in a separate terminal before launching this app:

    Terminal 1:  venv/Scripts/python.exe server/flask_app.py
    Terminal 2:  venv/Scripts/python.exe -m streamlit run ui/streamlit_app.py
"""

import io
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from PIL import Image
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FLASK_URL = "http://localhost:5000"
SENSOR_REFRESH_MS = 10000  # 10 seconds - slower to reduce page blinking

st.set_page_config(
    page_title="Multimodal Crop Disease Prediction",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state defaults (persist across reruns)
# ---------------------------------------------------------------------------
for key, default in [
    ("image_bytes", None),
    ("image_name", None),
    ("prediction", None),
    ("prediction_error", None),
    ("is_predicting", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def fetch_sensor():
    try:
        r = requests.get(f"{FLASK_URL}/latest_sensor", timeout=3)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def fetch_server_health():
    try:
        r = requests.get(f"{FLASK_URL}/", timeout=3)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def post_prediction(image_bytes: bytes, filename: str):
    files = {"image": (filename, image_bytes, "application/octet-stream")}
    try:
        r = requests.post(f"{FLASK_URL}/predict", files=files, timeout=60)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def pretty_class_name(name: str) -> str:
    return name.replace("___", " - ").replace("_", " ").strip()


def status_tag(value: float, good: tuple, warn: tuple) -> str:
    if good[0] <= value <= good[1]:
        color, tag = "#22c55e", "Optimal"
    elif warn[0] <= value <= warn[1]:
        color, tag = "#f59e0b", "Caution"
    else:
        color, tag = "#ef4444", "Alert"
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:12px'>{tag}</span>"
    )


def clear_prediction():
    st.session_state.prediction = None
    st.session_state.prediction_error = None
    st.session_state.image_bytes = None
    st.session_state.image_name = None
    st.session_state.is_predicting = False


def run_prediction_callback():
    """
    Button on_click callback. Runs the prediction synchronously and stores
    the result in session_state. Using on_click avoids the button-click-lost
    race condition with st_autorefresh.
    """
    if st.session_state.image_bytes is None:
        return
    st.session_state.is_predicting = True
    result, perr = post_prediction(
        st.session_state.image_bytes, st.session_state.image_name
    )
    st.session_state.is_predicting = False
    if perr:
        st.session_state.prediction_error = perr
        st.session_state.prediction = None
    else:
        st.session_state.prediction_error = None
        st.session_state.prediction = result


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;margin-bottom:0'>
        🌿 Multimodal Crop Disease Prediction
    </h1>
    <p style='text-align:center;color:gray;margin-top:4px'>
        Leaf image + live IoT sensor readings &nbsp;•&nbsp;
        EfficientNetB0 + Sensor Net + Late Fusion
    </p>
    """,
    unsafe_allow_html=True,
)

health, herr = fetch_server_health()
if herr:
    st.error(
        f"❌ Cannot reach Flask server at {FLASK_URL} - {herr}\n\n"
        "Start the server first: `venv/Scripts/python.exe server/flask_app.py`"
    )
    st.stop()
else:
    src = health.get("sensor_source", "?")
    st.success(
        f"Connected to Flask server • predictor loaded: "
        f"{health.get('predictor_loaded', False)} • sensor source: {src}"
    )

# ---------------------------------------------------------------------------
# Two-column layout
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 2], gap="large")

# ---- LEFT: Live sensor card (auto-refreshing) ----
with col_left:
    st.subheader("📡 Live sensor data")
    # Auto-refresh the whole app so the sensor card updates - but ONLY when
    # no prediction is currently displayed. Once a result is on screen we
    # pause the refresh to stop the page blinking and to avoid stealing
    # button-click reruns.
    if st.session_state.prediction is None and not st.session_state.is_predicting:
        st_autorefresh(interval=SENSOR_REFRESH_MS, key="sensor_autorefresh")

    sensor, err = fetch_sensor()
    if err:
        st.error(f"Sensor fetch failed: {err}")
    else:
        temp = sensor["temperature"]
        hum = sensor["humidity"]
        soil = sensor["soil_moisture"]

        st.metric("Temperature", f"{temp:.1f} °C")
        st.markdown(status_tag(temp, (18, 30), (10, 35)), unsafe_allow_html=True)
        st.metric("Humidity", f"{hum:.1f} %")
        st.markdown(status_tag(hum, (50, 80), (30, 95)), unsafe_allow_html=True)
        st.metric("Soil moisture", f"{soil:.1f} %")
        st.markdown(status_tag(soil, (40, 70), (25, 85)), unsafe_allow_html=True)

        ts = sensor.get("timestamp", "")
        src = sensor.get("source", "?")
        st.caption(f"Source: **{src}** • updated: {ts}")

    with st.expander("ℹ️ About the sensor card"):
        st.write(
            "Readings come from an ESP32 + DHT22 + soil moisture sensor "
            "posted to the Flask server over Wi-Fi. While hardware is in "
            "transit, the server generates realistic mock values that drift "
            "every 5 seconds so the UI remains demonstrable."
        )

# ---- RIGHT: Upload + result ----
with col_right:
    st.subheader("🍃 Upload a leaf photo")

    uploaded = st.file_uploader(
        "Choose an image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="uploader",
    )

    # If user picks a NEW file, store its bytes and clear any old prediction
    if uploaded is not None:
        new_bytes = uploaded.getvalue()
        if new_bytes != st.session_state.image_bytes:
            st.session_state.image_bytes = new_bytes
            st.session_state.image_name = uploaded.name
            st.session_state.prediction = None  # reset old result for new image

    # Display current uploaded image (from session_state so it survives reruns)
    if st.session_state.image_bytes is not None:
        pil_preview = Image.open(io.BytesIO(st.session_state.image_bytes))

        preview_col, btn_col = st.columns([2, 1])
        with preview_col:
            st.image(
                pil_preview,
                caption=f"Uploaded: {st.session_state.image_name}",
                use_column_width=True,
            )
        with btn_col:
            st.write("")
            st.write("")
            st.button(
                "🔬 Predict disease",
                type="primary",
                use_container_width=True,
                on_click=run_prediction_callback,
                key="predict_btn",
            )
            st.button(
                "🧹 Clear",
                use_container_width=True,
                on_click=clear_prediction,
                key="clear_btn",
            )

    # Show any prediction error from the most recent callback run
    if st.session_state.prediction_error:
        st.error(f"Prediction request failed: {st.session_state.prediction_error}")
    else:
        st.info(
            "⬆️ Upload a leaf image to run the multimodal pipeline.\n\n"
            "Tip: grab any image from `data/test/` to try the app."
        )

    # ---- Result panel (reads from session_state so auto-refresh can't wipe it) ----
    result = st.session_state.prediction
    if result is not None:
        st.divider()
        disease = pretty_class_name(result["disease"])
        confidence = result["confidence"]
        top_k = result["top_k"]
        advice = result["advice"]
        sensor_used = result["sensor_used"]
        gradcam_url = FLASK_URL + result["gradcam_url"]
        img_only = pretty_class_name(result.get("image_only_top", ""))
        sen_only = pretty_class_name(result.get("sensor_only_top", ""))

        headline_color = "#16a34a" if "healthy" in result["disease"].lower() else "#dc2626"
        st.markdown(
            f"<h2 style='color:{headline_color};margin-bottom:0'>{disease}</h2>"
            f"<p style='color:gray;margin-top:2px'>"
            f"Confidence: <b>{confidence*100:.2f}%</b></p>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(confidence, 0.0), 1.0))

        img1, img2 = st.columns(2)
        with img1:
            pil_preview = Image.open(io.BytesIO(st.session_state.image_bytes))
            st.image(pil_preview, caption="Original", use_column_width=True)
        with img2:
            st.image(gradcam_url, caption="Grad-CAM explanation", use_column_width=True)

        st.markdown("##### Top 3 predictions")
        top_df = pd.DataFrame(
            {
                "class": [pretty_class_name(n) for n, _ in top_k],
                "probability": [p for _, p in top_k],
            }
        ).set_index("class")
        st.bar_chart(top_df, height=180)

        st.markdown("##### Modality breakdown")
        mod_col1, mod_col2, mod_col3 = st.columns(3)
        mod_col1.metric("Image-only", img_only)
        mod_col2.metric("Sensor-only", sen_only)
        mod_col3.metric("Fusion (final)", disease)
        st.caption(
            f"Sensor used at prediction time: "
            f"T={sensor_used['temperature']:.1f}°C • "
            f"H={sensor_used['humidity']:.1f}% • "
            f"Soil={sensor_used['soil_moisture']:.1f}%"
        )

        st.markdown("##### Treatment guidance")
        with st.expander("Symptoms", expanded=True):
            st.write(advice.get("symptoms", "-"))
        with st.expander("Cause"):
            st.write(advice.get("cause", "-"))
        with st.expander("Treatment"):
            st.write(advice.get("treatment", "-"))
        with st.expander("Prevention"):
            st.write(advice.get("prevention", "-"))

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "MCA Final Year Major Project • Multimodal Crop Disease Prediction • "
    f"Last rendered {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
