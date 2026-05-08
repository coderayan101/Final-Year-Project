"""
Phase 8 - Evaluate the image model on the held-out test set.

Outputs:
    reports/metrics/image_report.txt   - classification report
    reports/figures/cm_image.png       - confusion matrix heatmap
    Prints overall test accuracy and macro-F1.

Usage:
    venv/Scripts/python.exe -m src.evaluate_image
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config
from .image_model_loader import load_image_model
from .utils import load_class_names


def main():
    print("[eval-image] Loading model and class names...")
    model = load_image_model()
    class_names = load_class_names()
    num_classes = len(class_names)

    print(f"[eval-image] Loading test data from {config.TEST_DIR}...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(config.TEST_DIR),
        image_size=config.IMG_SIZE,
        batch_size=config.IMG_BATCH_SIZE,
        shuffle=False,
        label_mode="int",
    )
    # The directory order must match our class_names
    dir_class_names = test_ds.class_names
    print(f"[eval-image] Found {len(dir_class_names)} classes, "
          f"{sum(1 for _ in test_ds.unbatch())} images")

    # Build mapping from dir index -> our class_names index
    dir_to_ours = {}
    for i, name in enumerate(dir_class_names):
        if name in class_names:
            dir_to_ours[i] = class_names.index(name)
        else:
            print(f"  WARNING: dir class '{name}' not in class_names.json")
            dir_to_ours[i] = i

    # Apply EfficientNet preprocessing
    from tensorflow.keras.applications.efficientnet import preprocess_input

    y_true_all = []
    y_pred_all = []

    total_batches = sum(1 for _ in test_ds)
    print(f"[eval-image] Running inference on {total_batches} batches...")

    for batch_i, (images, labels) in enumerate(test_ds):
        if (batch_i + 1) % 20 == 0 or batch_i == 0:
            print(f"  batch {batch_i + 1}/{total_batches}")

        images_pp = preprocess_input(tf.cast(images, tf.float32))
        preds = model.predict(images_pp, verbose=0)
        pred_indices = np.argmax(preds, axis=1)

        for label, pred in zip(labels.numpy(), pred_indices):
            y_true_all.append(dir_to_ours.get(int(label), int(label)))
            y_pred_all.append(int(pred))

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    print(f"\n{'='*60}")
    print(f"IMAGE MODEL - TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"Test accuracy: {acc*100:.2f}%")
    print(f"\n{report}")

    # Save report
    metrics_dir = config.REPORTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_path = metrics_dir / "image_report.txt"
    with open(report_path, "w") as f:
        f.write(f"IMAGE MODEL - TEST SET RESULTS\n")
        f.write(f"Test accuracy: {acc*100:.2f}%\n\n")
        f.write(report)
    print(f"[eval-image] Saved report -> {report_path}")

    # Save confusion matrix
    figures_dir = config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm_path = figures_dir / "cm_image.png"

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Greens",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Image Model Confusion Matrix (Test Acc: {acc*100:.2f}%)", fontsize=14)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    fig.savefig(str(cm_path), dpi=150)
    plt.close(fig)
    print(f"[eval-image] Saved confusion matrix -> {cm_path}")

    return acc, report, cm


if __name__ == "__main__":
    main()
