"""
End-to-end inference pipeline that combines image, sensor, and fusion models.

Loads all models once (via MultimodalPredictor) and exposes a single
`predict(pil_img, sensor_reading)` method used by the Flask server and
for standalone testing.

Usage (standalone smoke test):
    venv/Scripts/python.exe -m src.predict
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from typing import Dict, List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tensorflow.keras.applications.efficientnet import preprocess_input  # noqa: E402

from . import config  # noqa: E402
from .disease_dict import get_disease_info  # noqa: E402
from .fusion_model_loader import load_fusion_model  # noqa: E402
from .gradcam import generate_gradcam  # noqa: E402
from .image_model_loader import load_image_model  # noqa: E402
from .sensor_model_loader import load_scaler, load_sensor_model  # noqa: E402
from .utils import load_class_names  # noqa: E402


class MultimodalPredictor:
    """Holds all loaded models and runs the full image+sensor+fusion pipeline."""

    def __init__(self):
        print("[predictor] Loading class names...")
        self.class_names = load_class_names()
        print("[predictor] Loading image model...")
        self.image_model = load_image_model()
        print("[predictor] Loading sensor model + scaler...")
        self.sensor_model = load_sensor_model()
        self.scaler = load_scaler()
        print("[predictor] Loading fusion model...")
        self.fusion_model = load_fusion_model()
        print("[predictor] Ready.")

    # ---- Helpers ----
    def _preprocess_image(self, pil_img: Image.Image) -> np.ndarray:
        img = pil_img.convert("RGB").resize(config.IMG_SIZE)
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)
        return np.expand_dims(arr, axis=0)

    def _preprocess_sensor(self, reading: Dict[str, float]) -> np.ndarray:
        row = np.array(
            [[reading["temperature"], reading["humidity"], reading["soil_moisture"]]],
            dtype=np.float32,
        )
        return self.scaler.transform(row).astype("float32")

    # ---- Main API ----
    def predict(
        self,
        pil_img: Image.Image,
        sensor_reading: Dict[str, float],
        top_k: int = 3,
        do_gradcam: bool = True,
    ) -> dict:
        """
        Run the full image + sensor + fusion pipeline.

        Args:
            pil_img: PIL Image (RGB or convertible).
            sensor_reading: dict with keys 'temperature', 'humidity', 'soil_moisture'.
            top_k: number of top predictions to return.
            do_gradcam: whether to compute a Grad-CAM overlay.

        Returns:
            dict with prediction, confidence, top_k list, advice, and optional
            Grad-CAM arrays.
        """
        img_array = self._preprocess_image(pil_img)
        sen_array = self._preprocess_sensor(sensor_reading)

        img_probs = self.image_model.predict(img_array, verbose=0)  # (1, 38)
        sen_probs = self.sensor_model.predict(sen_array, verbose=0)  # (1, 38)
        fused_probs = self.fusion_model.predict([img_probs, sen_probs], verbose=0)  # (1, 38)

        fused = fused_probs[0]
        top_idx = int(fused.argmax())
        top_class = self.class_names[top_idx]
        confidence = float(fused[top_idx])

        top_sorted = fused.argsort()[::-1][:top_k]
        top_list: List[Tuple[str, float]] = [
            (self.class_names[int(i)], float(fused[int(i)])) for i in top_sorted
        ]

        result = {
            "disease": top_class,
            "confidence": confidence,
            "top_k": top_list,
            "image_only_top": self.class_names[int(img_probs[0].argmax())],
            "sensor_only_top": self.class_names[int(sen_probs[0].argmax())],
            "sensor_used": sensor_reading,
            "advice": get_disease_info(top_class),
        }

        if do_gradcam:
            overlay_bgr, overlay_rgb, heatmap, _, _ = generate_gradcam(
                self.image_model, pil_img, pred_index=int(img_probs[0].argmax())
            )
            result["gradcam_bgr"] = overlay_bgr
            result["gradcam_rgb"] = overlay_rgb
            result["heatmap"] = heatmap

        return result


def _smoke_test():
    """Run the pipeline on one sample test image."""
    from pathlib import Path

    predictor = MultimodalPredictor()

    # Pick a sample test image
    candidate_classes = ["Tomato___Late_blight", "Potato___Late_blight", "Apple___Apple_scab"]
    sample_path = None
    for cls in candidate_classes:
        d = config.TEST_DIR / cls
        if d.exists():
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if files:
                sample_path = files[0]
                break
    if sample_path is None:
        raise FileNotFoundError("No sample image found in data/test")

    print(f"\nSample: {sample_path.relative_to(Path.cwd())}")
    pil_img = Image.open(sample_path)
    sensor_reading = {"temperature": 20.0, "humidity": 90.0, "soil_moisture": 70.0}
    print(f"Mock sensor: {sensor_reading}")

    result = predictor.predict(pil_img, sensor_reading, do_gradcam=False)

    print("\n--- Fusion prediction ---")
    print(f"  disease:       {result['disease']}")
    print(f"  confidence:    {result['confidence']*100:.2f}%")
    print(f"  image-only:    {result['image_only_top']}")
    print(f"  sensor-only:   {result['sensor_only_top']}")
    print(f"  top-3:")
    for name, prob in result["top_k"]:
        print(f"    {prob*100:5.2f}%  {name}")
    print(f"  advice symptoms: {result['advice']['symptoms'][:80]}...")


if __name__ == "__main__":
    _smoke_test()
