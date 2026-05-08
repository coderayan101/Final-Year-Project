"""
Version-independent loader for the trained sensor model.

Rebuilds the dense-network architecture from code and loads only the
trained weights, avoiding cross-version Keras serialization issues.

Usage:
    from src.sensor_model_loader import load_sensor_model, load_scaler
    model = load_sensor_model()
    scaler = load_scaler()
    # scaled = scaler.transform([[temp, humidity, soil]])
    # probs  = model.predict(scaled)
"""

from pathlib import Path

import joblib
from tensorflow.keras import layers, models

from . import config

WEIGHTS_PATH = config.MODELS_DIR / "sensor" / "sensor_model.weights.h5"
SCALER_PATH = config.SENSOR_SCALER_PATH


def build_sensor_model(num_features: int = 3, num_classes: int = config.NUM_CLASSES):
    """
    Must match the architecture used in train_sensor.py exactly,
    or weight loading will fail.
    """
    inputs = layers.Input(shape=(num_features,), name="sensor_input")
    x = layers.Dense(config.SENSOR_HIDDEN[0], activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(config.SENSOR_HIDDEN[1], activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="sensor_dense_net")


def load_sensor_model(weights_path: Path = WEIGHTS_PATH):
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Sensor weights not found at {weights_path}. "
            f"Run: venv/Scripts/python.exe -m src.train_sensor"
        )
    model = build_sensor_model()
    model.load_weights(str(weights_path))
    return model


def load_scaler(scaler_path: Path = SCALER_PATH):
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            f"Run: venv/Scripts/python.exe -m src.train_sensor"
        )
    return joblib.load(scaler_path)


if __name__ == "__main__":
    import numpy as np

    model = load_sensor_model()
    scaler = load_scaler()
    print(f"Sensor model loaded.")
    print(f"  params: {model.count_params():,}")
    print(f"  input:  {model.input_shape}")
    print(f"  output: {model.output_shape}")

    sample = np.array([[22.0, 90.0, 70.0]])  # cool + humid = late blight territory
    scaled = scaler.transform(sample)
    probs = model.predict(scaled, verbose=0)
    print(f"  sample input: {sample[0]}")
    print(f"  top class idx: {probs.argmax()}  (prob {probs.max():.3f})")
