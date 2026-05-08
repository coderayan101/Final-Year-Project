"""
Phase 8 - Evaluate the full multimodal fusion pipeline on the test set.

For each test image:
    1. Run the image model -> 38 probs
    2. Pick a matching-class sensor sample -> run sensor model -> 38 probs
    3. Run fusion model on concatenated probs -> final prediction

Compares image-only, sensor-only, and fusion accuracy side-by-side.

Outputs:
    reports/metrics/fusion_report.txt      - classification report
    reports/metrics/summary.json           - all three accuracies in one file
    reports/figures/cm_fusion.png           - confusion matrix heatmap
    reports/figures/fusion_comparison.png   - bar chart comparing 3 models

Usage:
    venv/Scripts/python.exe -m src.evaluate_fusion
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config
from .image_model_loader import load_image_model
from .sensor_model_loader import load_sensor_model, load_scaler
from .fusion_model_loader import load_fusion_model
from .utils import load_class_names


def main():
    # ------------------------------------------------------------------
    # Load everything
    # ------------------------------------------------------------------
    print("[eval-fusion] Loading all models...")
    image_model = load_image_model()
    sensor_model = load_sensor_model()
    fusion_model = load_fusion_model()
    scaler = load_scaler()
    class_names = load_class_names()
    num_classes = len(class_names)

    from tensorflow.keras.applications.efficientnet import preprocess_input

    # ------------------------------------------------------------------
    # Prepare sensor test pool (one random sample per class for pairing)
    # ------------------------------------------------------------------
    print("[eval-fusion] Loading sensor CSV for pairing...")
    df_sensor = pd.read_csv(config.SENSOR_CSV)
    label_map = {name: i for i, name in enumerate(class_names)}
    df_sensor["label_idx"] = df_sensor["disease_class"].map(label_map)
    df_sensor = df_sensor.dropna(subset=["label_idx"])
    df_sensor["label_idx"] = df_sensor["label_idx"].astype(int)

    # Build a pool: for each class, keep all sensor rows for random pairing
    sensor_pool = {}
    for idx in range(num_classes):
        rows = df_sensor[df_sensor["label_idx"] == idx][config.SENSOR_FEATURES].values
        if len(rows) > 0:
            sensor_pool[idx] = rows.astype("float32")
        else:
            # fallback: use global mean
            sensor_pool[idx] = df_sensor[config.SENSOR_FEATURES].mean().values.reshape(1, -1).astype("float32")

    # ------------------------------------------------------------------
    # Load test images
    # ------------------------------------------------------------------
    print(f"[eval-fusion] Loading test images from {config.TEST_DIR}...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(config.TEST_DIR),
        image_size=config.IMG_SIZE,
        batch_size=config.IMG_BATCH_SIZE,
        shuffle=False,
        label_mode="int",
    )
    dir_class_names = test_ds.class_names

    # Mapping from dir index -> our index
    dir_to_ours = {}
    for i, name in enumerate(dir_class_names):
        if name in class_names:
            dir_to_ours[i] = class_names.index(name)
        else:
            dir_to_ours[i] = i

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    y_true_all = []
    y_pred_img_all = []
    y_pred_sen_all = []
    y_pred_fus_all = []

    rng = np.random.RandomState(config.RANDOM_SEED)
    total_batches = sum(1 for _ in test_ds)
    print(f"[eval-fusion] Running inference on {total_batches} batches...")

    for batch_i, (images, labels) in enumerate(test_ds):
        if (batch_i + 1) % 20 == 0 or batch_i == 0:
            print(f"  batch {batch_i + 1}/{total_batches}")

        # Image predictions
        images_pp = preprocess_input(tf.cast(images, tf.float32))
        img_probs = image_model.predict(images_pp, verbose=0)  # (B, 38)

        batch_size = len(labels)
        # For each sample, pair with a matching-class sensor row
        sen_inputs = np.zeros((batch_size, 3), dtype="float32")
        for j, lab in enumerate(labels.numpy()):
            true_idx = dir_to_ours.get(int(lab), int(lab))
            pool = sensor_pool[true_idx]
            row = pool[rng.randint(len(pool))]
            sen_inputs[j] = row

        sen_scaled = scaler.transform(sen_inputs).astype("float32")
        sen_probs = sensor_model.predict(sen_scaled, verbose=0)  # (B, 38)

        # Fusion
        fus_probs = fusion_model.predict([img_probs, sen_probs], verbose=0)  # (B, 38)

        for j, lab in enumerate(labels.numpy()):
            true_idx = dir_to_ours.get(int(lab), int(lab))
            y_true_all.append(true_idx)
            y_pred_img_all.append(int(np.argmax(img_probs[j])))
            y_pred_sen_all.append(int(np.argmax(sen_probs[j])))
            y_pred_fus_all.append(int(np.argmax(fus_probs[j])))

    y_true = np.array(y_true_all)
    y_pred_img = np.array(y_pred_img_all)
    y_pred_sen = np.array(y_pred_sen_all)
    y_pred_fus = np.array(y_pred_fus_all)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    acc_img = accuracy_score(y_true, y_pred_img)
    acc_sen = accuracy_score(y_true, y_pred_sen)
    acc_fus = accuracy_score(y_true, y_pred_fus)

    report_fus = classification_report(
        y_true, y_pred_fus, target_names=class_names, digits=4, zero_division=0
    )
    cm_fus = confusion_matrix(y_true, y_pred_fus, labels=range(num_classes))

    print(f"\n{'='*60}")
    print(f"FUSION MODEL - TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"Image-only test accuracy:  {acc_img*100:.2f}%")
    print(f"Sensor-only test accuracy: {acc_sen*100:.2f}%")
    print(f"Fusion test accuracy:      {acc_fus*100:.2f}%")
    print(f"Fusion lift over image:    +{(acc_fus - acc_img)*100:.2f}%")
    print(f"\n{report_fus}")

    # ------------------------------------------------------------------
    # Save reports
    # ------------------------------------------------------------------
    metrics_dir = config.REPORTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Classification report
    report_path = metrics_dir / "fusion_report.txt"
    with open(report_path, "w") as f:
        f.write(f"FUSION MODEL - TEST SET RESULTS\n")
        f.write(f"Image-only test accuracy:  {acc_img*100:.2f}%\n")
        f.write(f"Sensor-only test accuracy: {acc_sen*100:.2f}%\n")
        f.write(f"Fusion test accuracy:      {acc_fus*100:.2f}%\n")
        f.write(f"Fusion lift over image:    +{(acc_fus - acc_img)*100:.2f}%\n\n")
        f.write(report_fus)
    print(f"[eval-fusion] Saved report -> {report_path}")

    # Summary JSON (all three numbers in one place for the report)
    summary = {
        "image_test_accuracy": round(acc_img * 100, 2),
        "sensor_test_accuracy": round(acc_sen * 100, 2),
        "fusion_test_accuracy": round(acc_fus * 100, 2),
        "fusion_lift_over_image": round((acc_fus - acc_img) * 100, 2),
        "total_test_images": len(y_true),
        "num_classes": num_classes,
    }
    summary_path = metrics_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval-fusion] Saved summary -> {summary_path}")

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    figures_dir = config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    cm_path = figures_dir / "cm_fusion.png"
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm_fus, annot=False, fmt="d", cmap="Oranges",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Fusion Model Confusion Matrix (Test Acc: {acc_fus*100:.2f}%)", fontsize=14)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    fig.savefig(str(cm_path), dpi=150)
    plt.close(fig)
    print(f"[eval-fusion] Saved confusion matrix -> {cm_path}")

    # ------------------------------------------------------------------
    # Comparison bar chart
    # ------------------------------------------------------------------
    comp_path = figures_dir / "fusion_comparison.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    models_names = ["Image Only", "Sensor Only", "Fusion"]
    accs = [acc_img * 100, acc_sen * 100, acc_fus * 100]
    colors = ["#2C5F2D", "#0984E3", "#E17055"]
    bars = ax.bar(models_names, accs, color=colors, width=0.5, edgecolor="white")

    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=13, fontweight="bold",
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Multimodal Fusion: Test Set Accuracy Comparison", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(comp_path), dpi=150)
    plt.close(fig)
    print(f"[eval-fusion] Saved comparison chart -> {comp_path}")

    return summary


if __name__ == "__main__":
    main()
