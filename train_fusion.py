"""
Phase 4 - Fusion model training.

Builds a paired dataset of (image_probs, sensor_probs, label) by:
  1. sampling N images per class from data/train and data/val
  2. running them through the trained image model (EfficientNetB0)
  3. synthesizing a realistic sensor reading per sample using the same
     class-conditional climate profiles as src/sensor_data_gen.py, then
     running those through the trained sensor model
  4. caching everything to .npz so retraining the fusion head is instant

The fusion head is a small dense network: Dense(128) -> Dropout -> Dense(38).

Also runs an ablation on the held-out validation set and writes the
comparison table to reports/results/fusion_ablation.txt.

Usage:
    venv/Scripts/python.exe -m src.train_fusion

Outputs:
    models/fusion/fusion_model.weights.h5
    models/fusion/history.json
    models/fusion/paired_cache.npz   (cached extracted probs)
    reports/figures/fusion_training_curves.png
    reports/figures/fusion_confusion_matrix.png
    reports/results/fusion_classification_report.txt
    reports/results/fusion_ablation.txt
"""

# OpenMP workaround + TF first (see train_sensor.py for rationale)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
from tensorflow.keras import callbacks, optimizers  # noqa: E402
from tensorflow.keras.applications.efficientnet import preprocess_input  # noqa: E402

import json  # noqa: E402
import random  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from . import config  # noqa: E402
from .fusion_model_loader import build_fusion_model  # noqa: E402
from .image_model_loader import load_image_model  # noqa: E402
from .sensor_model_loader import load_sensor_model, load_scaler  # noqa: E402
from .sensor_data_gen import CLIMATE_PROFILES, DEFAULT_HEALTHY  # noqa: E402
from .utils import ensure_dir, load_class_names  # noqa: E402

# ---- Sampling limits (Option B - CPU-friendly) ----
N_TRAIN_PER_CLASS = 80   # 80 * 38 = 3,040 training pairs
N_VAL_PER_CLASS = 20     # 20 * 38 =   760 validation pairs
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def sample_sensor_reading(class_name: str) -> np.ndarray:
    """Draw one synthetic (temp, humidity, soil) reading for a given class."""
    profile = CLIMATE_PROFILES.get(class_name, DEFAULT_HEALTHY)
    t_m, t_s, h_m, h_s, s_m, s_s = profile
    temp = np.clip(np.random.normal(t_m, t_s), *config.SENSOR_RANGES["temperature"])
    hum = np.clip(np.random.normal(h_m, h_s), *config.SENSOR_RANGES["humidity"])
    soil = np.clip(np.random.normal(s_m, s_s), *config.SENSOR_RANGES["soil_moisture"])
    return np.array([temp, hum, soil], dtype="float32")


def load_and_preprocess_image(path: Path) -> np.ndarray:
    img = tf.keras.utils.load_img(str(path), target_size=config.IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = preprocess_input(arr)
    return arr


def extract_probs_for_split(
    split_dir: Path,
    class_names,
    per_class: int,
    image_model,
    sensor_model,
    scaler,
):
    """Return (img_probs, sen_probs, labels) numpy arrays for a given split."""
    img_batch, sen_batch, labels = [], [], []

    for idx, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        files = [p for p in class_dir.iterdir() if p.suffix in IMAGE_EXTS]
        random.shuffle(files)
        files = files[:per_class]

        for f in files:
            img = load_and_preprocess_image(f)
            sen = sample_sensor_reading(class_name)
            img_batch.append(img)
            sen_batch.append(sen)
            labels.append(idx)

    img_arr = np.stack(img_batch, axis=0)
    sen_arr = np.stack(sen_batch, axis=0)
    labels_arr = np.array(labels, dtype=np.int32)

    print(f"  running image model on {len(img_arr)} images...")
    img_probs = image_model.predict(img_arr, batch_size=32, verbose=1)

    print(f"  running sensor model on {len(sen_arr)} readings...")
    sen_scaled = scaler.transform(sen_arr)
    sen_probs = sensor_model.predict(sen_scaled, batch_size=128, verbose=0)

    return img_probs.astype("float32"), sen_probs.astype("float32"), labels_arr


def build_paired_dataset(cache_path: Path, class_names):
    if cache_path.exists():
        print(f"Loading cached paired dataset from {cache_path}")
        d = np.load(cache_path)
        return (
            d["train_img"], d["train_sen"], d["train_y"],
            d["val_img"], d["val_sen"], d["val_y"],
        )

    print("Loading image model...")
    image_model = load_image_model()
    print("Loading sensor model + scaler...")
    sensor_model = load_sensor_model()
    scaler = load_scaler()

    print(f"\nExtracting TRAIN probs ({N_TRAIN_PER_CLASS} per class)...")
    train_img, train_sen, train_y = extract_probs_for_split(
        config.TRAIN_DIR, class_names, N_TRAIN_PER_CLASS,
        image_model, sensor_model, scaler,
    )

    print(f"\nExtracting VAL probs ({N_VAL_PER_CLASS} per class)...")
    val_img, val_sen, val_y = extract_probs_for_split(
        config.VAL_DIR, class_names, N_VAL_PER_CLASS,
        image_model, sensor_model, scaler,
    )

    ensure_dir(cache_path.parent)
    np.savez(
        cache_path,
        train_img=train_img, train_sen=train_sen, train_y=train_y,
        val_img=val_img, val_sen=val_sen, val_y=val_y,
    )
    print(f"\nCached paired dataset -> {cache_path}")
    return train_img, train_sen, train_y, val_img, val_sen, val_y


def ablation_report(train_img, train_sen, train_y,
                    val_img, val_sen, val_y,
                    fusion_model, class_names):
    """Compare image-only vs sensor-only vs fusion on validation set."""
    img_only = val_img.argmax(axis=1)
    sen_only = val_sen.argmax(axis=1)
    fus_probs = fusion_model.predict([val_img, val_sen], verbose=0)
    fus_pred = fus_probs.argmax(axis=1)

    def acc(pred):
        return float((pred == val_y).mean())

    lines = [
        "=" * 60,
        "Multimodal Ablation Report (validation set)",
        "=" * 60,
        f"Number of validation samples: {len(val_y)}",
        f"Number of classes:            {len(class_names)}",
        "",
        f"Image-only  accuracy:  {acc(img_only):.4f}",
        f"Sensor-only accuracy:  {acc(sen_only):.4f}",
        f"Fusion      accuracy:  {acc(fus_pred):.4f}",
        "",
        f"Lift of fusion over image-only: "
        f"{(acc(fus_pred) - acc(img_only)) * 100:+.2f} percentage points",
        f"Lift of fusion over sensor-only: "
        f"{(acc(fus_pred) - acc(sen_only)) * 100:+.2f} percentage points",
        "",
    ]
    return "\n".join(lines), fus_pred


def main():
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    ensure_dir(config.MODELS_DIR / "fusion")
    ensure_dir(config.FIGURES_DIR)
    ensure_dir(config.RESULTS_DIR)

    class_names = load_class_names()
    print(f"Loaded {len(class_names)} classes")

    cache_path = config.MODELS_DIR / "fusion" / "paired_cache.npz"
    train_img, train_sen, train_y, val_img, val_sen, val_y = build_paired_dataset(
        cache_path, class_names
    )
    print(f"\nTrain: {len(train_y)}  Val: {len(val_y)}")

    # ---- Build + train fusion head ----
    model = build_fusion_model(num_classes=len(class_names))
    model.compile(
        optimizer=optimizers.Adam(config.FUSION_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    es = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )
    history = model.fit(
        [train_img, train_sen], train_y,
        validation_data=([val_img, val_sen], val_y),
        epochs=config.FUSION_EPOCHS,
        batch_size=config.FUSION_BATCH_SIZE,
        callbacks=[es],
        verbose=2,
    )

    # ---- Save weights + history ----
    weights_path = config.MODELS_DIR / "fusion" / "fusion_model.weights.h5"
    model.save_weights(str(weights_path))
    with open(config.MODELS_DIR / "fusion" / "history.json", "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"\nSaved weights -> {weights_path}")

    # ---- Training curves ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Fusion model accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Fusion model loss"); axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    curves_path = config.FIGURES_DIR / "fusion_training_curves.png"
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"Saved curves -> {curves_path}")

    # ---- Ablation + classification report ----
    report_text, fus_pred = ablation_report(
        train_img, train_sen, train_y,
        val_img, val_sen, val_y,
        model, class_names,
    )
    ablation_path = config.RESULTS_DIR / "fusion_ablation.txt"
    with open(ablation_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print("\n" + report_text)
    print(f"Saved ablation report -> {ablation_path}")

    cls_report = classification_report(
        val_y, fus_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=3,
    )
    cls_path = config.RESULTS_DIR / "fusion_classification_report.txt"
    with open(cls_path, "w", encoding="utf-8") as f:
        f.write(cls_report)
    print(f"Saved classification report -> {cls_path}")

    # ---- Confusion matrix ----
    cm = confusion_matrix(val_y, fus_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, ax=ax, cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names, square=True,
    )
    ax.set_title("Fusion model confusion matrix (validation)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    cm_path = config.FIGURES_DIR / "fusion_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {cm_path}")

    print("\n=== Fusion model training complete ===")


if __name__ == "__main__":
    main()
