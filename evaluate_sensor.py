"""
Phase 8 - Evaluate the sensor model on a held-out test split.

Splits the sensor CSV 85/15 (matching the random seed used in training),
evaluates on the 15% test portion.

Outputs:
    reports/metrics/sensor_report.txt  - classification report
    reports/figures/cm_sensor.png      - confusion matrix heatmap

Usage:
    venv/Scripts/python.exe -m src.evaluate_sensor
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402 - must import before seaborn to avoid OpenMP crash

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config
from .sensor_model_loader import load_sensor_model, load_scaler
from .utils import load_class_names


def main():
    print("[eval-sensor] Loading model, scaler, class names...")
    model = load_sensor_model()
    scaler = load_scaler()
    class_names = load_class_names()
    num_classes = len(class_names)

    print(f"[eval-sensor] Loading sensor CSV from {config.SENSOR_CSV}...")
    df = pd.read_csv(config.SENSOR_CSV)
    print(f"  Total rows: {len(df)}")

    # Encode labels
    label_map = {name: i for i, name in enumerate(class_names)}
    df["label_idx"] = df["disease_class"].map(label_map)
    df = df.dropna(subset=["label_idx"])
    df["label_idx"] = df["label_idx"].astype(int)

    X = df[config.SENSOR_FEATURES].values.astype("float32")
    y = df["label_idx"].values

    # Same split as training: 85% train, 15% test
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.15, random_state=config.RANDOM_SEED, stratify=y
    )
    print(f"  Test split: {len(X_test)} samples")

    # Scale
    X_test_scaled = scaler.transform(X_test).astype("float32")

    # Predict
    print("[eval-sensor] Running inference...")
    preds = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))

    print(f"\n{'='*60}")
    print(f"SENSOR MODEL - TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"Test accuracy: {acc*100:.2f}%")
    print(f"\n{report}")

    # Save report
    metrics_dir = config.REPORTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_path = metrics_dir / "sensor_report.txt"
    with open(report_path, "w") as f:
        f.write(f"SENSOR MODEL - TEST SET RESULTS\n")
        f.write(f"Test accuracy: {acc*100:.2f}%\n\n")
        f.write(report)
    print(f"[eval-sensor] Saved report -> {report_path}")

    # Save confusion matrix
    figures_dir = config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm_path = figures_dir / "cm_sensor.png"

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Sensor Model Confusion Matrix (Test Acc: {acc*100:.2f}%)", fontsize=14)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    fig.savefig(str(cm_path), dpi=150)
    plt.close(fig)
    print(f"[eval-sensor] Saved confusion matrix -> {cm_path}")

    return acc, report, cm


if __name__ == "__main__":
    main()
