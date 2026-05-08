"""
Version-independent loader for the trained EfficientNetB0 image model.

Instead of loading a serialized .keras/.h5 file (which breaks across Keras
versions because of things like `quantization_config`), we rebuild the
architecture from code and load only the numerical weights.

Usage:
    from src.image_model_loader import load_image_model
    model = load_image_model()
    # model is a compiled Keras model, input (None, 224, 224, 3),
    # output (None, 38) softmax probabilities.
"""

from pathlib import Path

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

from . import config

WEIGHTS_PATH = config.MODELS_DIR / "image" / "image_model.weights.h5"


def build_image_model(num_classes: int = config.NUM_CLASSES):
    """
    Recreate the exact architecture used in train_image.py / the Colab notebook.
    Must match cell-for-cell or weights will not load.
    """
    base = EfficientNetB0(
        include_top=False,
        weights=None,  # ImageNet weights not needed; fine-tuned weights loaded below
        input_shape=(*config.IMG_SIZE, config.IMG_CHANNELS),
    )

    inputs = layers.Input(shape=(*config.IMG_SIZE, config.IMG_CHANNELS))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="efficientnetb0_plantvillage")
    return model


def load_image_model(weights_path: Path = WEIGHTS_PATH):
    """Build architecture + load trained weights. Returns ready-to-predict model."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained weights not found at {weights_path}. "
            f"Train in Colab then download image_model.weights.h5 into models/image/."
        )
    model = build_image_model()
    model.load_weights(str(weights_path))
    return model


if __name__ == "__main__":
    import numpy as np
    from tensorflow.keras.applications.efficientnet import preprocess_input

    m = load_image_model()
    print(f"Model loaded successfully.")
    print(f"  params: {m.count_params():,}")
    print(f"  input:  {m.input_shape}")
    print(f"  output: {m.output_shape}")

    # Smoke test: run a dummy input through the model
    dummy = np.random.rand(1, *config.IMG_SIZE, 3).astype("float32") * 255
    dummy = preprocess_input(dummy)
    probs = m.predict(dummy, verbose=0)
    print(f"  dummy prediction shape: {probs.shape}")
    print(f"  dummy prediction sum:   {probs.sum():.4f} (should be ~1.0)")
    print(f"  dummy top class idx:    {probs.argmax()}")
