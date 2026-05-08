<<<<<<< HEAD
# Multimodal Crop Disease Prediction

MCA final-year major project — predicts crop leaf diseases by combining a leaf image (CNN) with live environmental sensor readings (temperature, humidity, soil moisture) from an ESP32 IoT device.

## Architecture

Three models combined via late fusion:

| Model | Type | Input | Output |
|---|---|---|---|
| Model 1 — Image | EfficientNetB0 (transfer learning) | Leaf photo (224x224) | 38 class probabilities |
| Model 2 — Sensor | Small dense network | temp, humidity, soil moisture | 38 class probabilities |
| Model 3 — Fusion | Dense(128) -> Dense(38) -> Softmax | Concatenated outputs of M1 + M2 | Final disease prediction |

Trained on the PlantVillage dataset (38 classes). Sensor dataset is synthesized using class-conditional agronomic ranges.

## System

- **Unit 1 — IoT device:** ESP32 + DHT22 + soil moisture sensor, posts JSON over WiFi/HTTP to the Flask server.
- **Unit 2 — Web app:** Streamlit UI showing live sensor readings, leaf upload, predicted disease, Grad-CAM heatmap, and treatment advice.

## Project layout

```
crop-disease-multimodal/
├── data/         raw + split datasets
├── models/       trained model artifacts (image / sensor / fusion / tflite)
├── src/          training + inference code
├── server/       Flask backend (talks to ESP32, serves predictions)
├── ui/           Streamlit frontend
├── firmware/     ESP32 Arduino sketch
├── notebooks/    experiments
├── reports/      figures, metrics for the project report
└── tests/        sanity tests
```

## Quick start

```bash
# 1. create virtual env
python -m venv venv
venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. (later) train models
python src/data_prep.py
python src/train_image.py
python src/sensor_data_gen.py
python src/train_sensor.py
python src/train_fusion.py

# 4. run the app
python server/flask_app.py
streamlit run ui/streamlit_app.py
```

## Status

Project bootstrapped — folder structure and config in place. Training scripts to be added in upcoming phases.
=======
# Final-Year-Project
>>>>>>> 9b59d56a734eac8d674e73e381b0cab7f4b89595
