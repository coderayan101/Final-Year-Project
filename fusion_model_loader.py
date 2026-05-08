"""
Version-independent loader for the trained fusion model.
"""

from pathlib import Path

from tensorflow.keras import layers, models

from . import config

WEIGHTS_PATH = config.MODELS_DIR / "fusion" / "fusion_model.weights.h5"


def build_fusion_model(num_classes: int = config.NUM_CLASSES):
    """
    Concatenates image-model and sensor-model probability vectors and learns
    a class-aware combination. Must match train_fusion.py exactly.
    """
    img_in = layers.Input(shape=(num_classes,), name="image_probs")
    sen_in = layers.Input(shape=(num_classes,), name="sensor_probs")
    x = layers.Concatenate()([img_in, sen_in])
    x = layers.Dense(config.FUSION_HIDDEN, activation="relu")(x)
    x = layers.Dropout(config.FUSION_DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model([img_in, sen_in], outputs, name="late_fusion")


def load_fusion_model(weights_path: Path = WEIGHTS_PATH):
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Fusion weights not found at {weights_path}. "
            f"Run: venv/Scripts/python.exe -m src.train_fusion"
        )
    model = build_fusion_model()
    model.load_weights(str(weights_path))
    return model


if __name__ == "__main__":
    import numpy as np

    m = load_fusion_model()
    print(f"Fusion model loaded.")
    print(f"  params: {m.count_params():,}")
    print(f"  inputs: {[i.shape for i in m.inputs]}")
    print(f"  output: {m.output_shape}")

    # Smoke test
    img_probs = np.random.dirichlet(np.ones(38), 1).astype("float32")
    sen_probs = np.random.dirichlet(np.ones(38), 1).astype("float32")
    out = m.predict([img_probs, sen_probs], verbose=0)
    print(f"  dummy output sum: {out.sum():.4f}  top idx: {out.argmax()}")
