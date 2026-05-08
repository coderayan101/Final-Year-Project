"""
Phase 3 - Sensor model training.

Trains a small dense neural network on data/sensor/sensor_dataset.csv to
predict disease class from (temperature, humidity, soil_moisture).

The sensor model outputs 38-class probabilities (matching the image model)
so the fusion model can learn how to combine both modalities class-by-class.

Usage:
    venv/Scripts/python.exe -m src.train_sensor

Outputs:
    models/sensor/sensor_model.weights.h5
    models/sensor/scaler.pkl
    models/sensor/history.json
    reports/figures/sensor_training_curves.png
    reports/figures/sensor_confusion_matrix.png
    reports/results/sensor_classification_report.txt
"""

# IMPORTANT: set OpenMP duplicate-lib workaround and import tensorflow FIRST,
# before seaborn/scipy, to avoid an OpenMP collision segfault on Windows.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
from tensorflow.keras import callbacks, optimizers  # noqa: E402

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from . import config  # noqa: E402
from .sensor_model_loader import build_sensor_model  # noqa: E402
from .utils import ensure_dir, load_class_names  # noqa: E402


def main():
    ensure_dir(config.MODELS_DIR / "sensor")
    ensure_dir(config.FIGURES_DIR)
    ensure_dir(config.RESULTS_DIR)

    # ---- Load class names for consistent label indexing ----
    class_names = load_class_names()
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"Loaded {len(class_names)} classes from class_names.json")

    # ---- Load sensor dataset ----
    if not config.SENSOR_CSV.exists():
        raise FileNotFoundError(
            f"{config.SENSOR_CSV} missing. Run: "
            f"venv/Scripts/python.exe -m src.sensor_data_gen"
        )
    df = pd.read_csv(config.SENSOR_CSV)
    print(f"Loaded {len(df)} rows from {config.SENSOR_CSV.name}")

    X = df[config.SENSOR_FEATURES].values.astype("float32")
    y = df["disease_class"].map(class_to_idx).values
    if np.isnan(y).any():
        missing = sorted(set(df["disease_class"]) - set(class_to_idx))
        raise ValueError(f"Unknown classes in sensor CSV: {missing}")

    # ---- Train/val split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    # ---- Fit scaler on train only ----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    scaler_path = config.SENSOR_SCALER_PATH
    ensure_dir(scaler_path.parent)
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler -> {scaler_path}")

    # ---- Build model ----
    model = build_sensor_model(num_features=X.shape[1], num_classes=len(class_names))
    model.compile(
        optimizer=optimizers.Adam(config.SENSOR_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ---- Train ----
    es = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    )
    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=config.SENSOR_EPOCHS,
        batch_size=config.SENSOR_BATCH_SIZE,
        callbacks=[es],
        verbose=2,
    )

    # ---- Save weights + history ----
    weights_path = config.MODELS_DIR / "sensor" / "sensor_model.weights.h5"
    model.save_weights(str(weights_path))
    with open(config.MODELS_DIR / "sensor" / "history.json", "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"Saved weights -> {weights_path}")

    # ---- Training curves ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Sensor model accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Sensor model loss")
    axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout()
    curves_path = config.FIGURES_DIR / "sensor_training_curves.png"
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"Saved curves -> {curves_path}")

    # ---- Evaluation on validation split ----
    val_probs = model.predict(X_val_s, verbose=0)
    val_preds = val_probs.argmax(axis=1)

    report = classification_report(
        y_val, val_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=3,
    )
    report_path = config.RESULTS_DIR / "sensor_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved classification report -> {report_path}")

    cm = confusion_matrix(y_val, val_preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, ax=ax, cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names,
        square=True,
    )
    ax.set_title("Sensor model confusion matrix (validation)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    cm_path = config.FIGURES_DIR / "sensor_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {cm_path}")

    # ---- Final summary ----
    final_acc = history.history["val_accuracy"][-1]
    best_acc = max(history.history["val_accuracy"])
    print("\n=== Sensor model training complete ===")
    print(f"  final val accuracy: {final_acc:.4f}")
    print(f"  best val accuracy:  {best_acc:.4f}")
    print(f"  classes: {len(class_names)}")
    print("\nNote: sensor-only accuracy is expected to be modest (~30-60%) "
          "because many disease classes share overlapping climate conditions. "
          "The fusion model will combine this signal with the image model "
          "for much stronger final predictions.")


if __name__ == "__main__":
    main()
