"""
Phase 5 - Grad-CAM explanation for the image model.

Grad-CAM (Gradient-weighted Class Activation Mapping) produces a heatmap
showing which regions of a leaf image influenced the predicted class.
Implementation from scratch, no extra dependencies required.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (ICCV 2017).

Usage as a module:
    from src.gradcam import generate_gradcam
    overlay_bgr, heatmap, pred_idx, pred_prob = generate_gradcam(model, pil_image)

Usage as a script (runs a smoke test on a sample leaf):
    venv/Scripts/python.exe -m src.gradcam
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image  # noqa: E402
from tensorflow.keras.applications.efficientnet import preprocess_input  # noqa: E402

from . import config  # noqa: E402
from .image_model_loader import load_image_model  # noqa: E402
from .utils import load_class_names  # noqa: E402


LAST_CONV_LAYER_NAME = "top_conv"  # last conv layer of EfficientNetB0


def _find_base_model(model):
    """Find the nested EfficientNetB0 functional sub-model inside our wrapper."""
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            return layer
    raise ValueError("Could not find EfficientNetB0 base inside model")


def _build_grad_components(model):
    """
    Keras 3 does not allow referencing a nested functional sub-model's
    internal layer output in the outer graph, so instead of building a
    single grad_model we return the reusable building blocks and run them
    manually inside a GradientTape.

    Returns a dict with:
        conv_model : sub-model from image input -> top_conv feature maps
        head_layers: list of layers to apply AFTER top_conv to reach final
                     softmax predictions, in order
    """
    base = _find_base_model(model)

    # Sub-model: image input -> last conv layer feature maps
    conv_model = tf.keras.models.Model(
        inputs=base.input,
        outputs=base.get_layer(LAST_CONV_LAYER_NAME).output,
    )

    # Layers of the base model AFTER top_conv (usually top_bn + top_activation)
    base_layer_names = [l.name for l in base.layers]
    top_conv_idx = base_layer_names.index(LAST_CONV_LAYER_NAME)
    post_conv_base_layers = base.layers[top_conv_idx + 1 :]

    # Layers of the outer wrapper model after the base (GAP, Dropout, Dense)
    outer_head_layers = []
    seen_base = False
    for layer in model.layers:
        if layer is base:
            seen_base = True
            continue
        if seen_base and not isinstance(layer, tf.keras.layers.InputLayer):
            outer_head_layers.append(layer)

    head_layers = post_conv_base_layers + outer_head_layers
    return {"conv_model": conv_model, "head_layers": head_layers}


def preprocess_pil(pil_img: Image.Image) -> np.ndarray:
    """Resize PIL image to model input size and apply EfficientNet preprocessing."""
    img = pil_img.convert("RGB").resize(config.IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def compute_heatmap(grad_components, image_array, pred_index=None):
    """
    Compute a normalized (H, W) heatmap in [0, 1] for the given image.
    If pred_index is None, uses the top predicted class.
    Returns (heatmap, pred_index, pred_prob).
    """
    conv_model = grad_components["conv_model"]
    head_layers = grad_components["head_layers"]

    image_tensor = tf.convert_to_tensor(image_array)
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(image_tensor, training=False)
        tape.watch(conv_outputs)
        x = conv_outputs
        for layer in head_layers:
            x = layer(x, training=False)
        predictions = x
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    # Global average pool the gradients over spatial dims -> (channels,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # drop batch dim -> (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)  # (H, W)

    # ReLU + normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy()

    pred_prob = float(predictions[0, pred_index].numpy())
    return heatmap, int(pred_index), pred_prob


def overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4):
    """
    Overlay the heatmap onto the original image using JET colormap.
    Returns a BGR numpy array ready for cv2.imwrite, and an RGB array ready
    for Streamlit / matplotlib display.
    """
    base = np.array(pil_img.convert("RGB").resize(config.IMG_SIZE))  # RGB
    heatmap_resized = cv2.resize(heatmap, config.IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay_rgb = cv2.addWeighted(base, 1 - alpha, heatmap_rgb, alpha, 0)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    return overlay_bgr, overlay_rgb


def generate_gradcam(model, pil_img: Image.Image, pred_index: int = None):
    """
    High-level helper used by the Flask server / Streamlit UI.
    Returns (overlay_bgr, overlay_rgb, heatmap, pred_index, pred_prob).
    """
    grad_components = _build_grad_components(model)
    image_array = preprocess_pil(pil_img)
    heatmap, pred_idx, pred_prob = compute_heatmap(
        grad_components, image_array, pred_index=pred_index
    )
    overlay_bgr, overlay_rgb = overlay_heatmap(pil_img, heatmap)
    return overlay_bgr, overlay_rgb, heatmap, pred_idx, pred_prob


# ---------------------------------------------------------------------------
# Smoke test: pick a test image and save a Grad-CAM overlay
# ---------------------------------------------------------------------------
def _pick_sample_image() -> tuple:
    """Find any image in data/test to use as a demo. Returns (path, class_name)."""
    if not config.TEST_DIR.exists():
        raise FileNotFoundError(f"{config.TEST_DIR} missing")
    preferred = ["Tomato___Late_blight", "Potato___Late_blight", "Apple___Apple_scab"]
    for cls in preferred:
        d = config.TEST_DIR / cls
        if d.exists():
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if files:
                return files[0], cls
    # fallback: first non-empty class
    for d in sorted(config.TEST_DIR.iterdir()):
        if d.is_dir():
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if files:
                return files[0], d.name
    raise FileNotFoundError("No images found in data/test")


def main():
    print("Loading image model...")
    model = load_image_model()
    class_names = load_class_names()

    sample_path, true_class = _pick_sample_image()
    print(f"Sample: {sample_path.name}  (true class: {true_class})")

    pil_img = Image.open(sample_path)
    overlay_bgr, overlay_rgb, heatmap, pred_idx, pred_prob = generate_gradcam(model, pil_img)
    pred_class = class_names[pred_idx]

    out_dir = config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gradcam_sample.png"
    cv2.imwrite(str(out_path), overlay_bgr)

    print(f"\nPrediction: {pred_class}  (confidence {pred_prob*100:.2f}%)")
    print(f"Heatmap shape: {heatmap.shape}, max={heatmap.max():.3f}")
    print(f"Saved Grad-CAM overlay -> {out_path}")


if __name__ == "__main__":
    main()
