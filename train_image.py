"""
Phase 2 - Image model training (EfficientNetB0).

This is the local-runnable version. Training 38k images on CPU takes many
hours, so in practice the Colab notebook at notebooks/train_image_colab.ipynb
is used to actually train the model. This file mirrors that logic so the
project codebase is self-contained and inspectable.

Usage (if you really want to run locally, CPU):
    venv/Scripts/python.exe -m src.train_image

Output:
    models/image/image_model.h5
    models/image/history.json
    reports/figures/image_training_curves.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from . import config
from .utils import ensure_dir, save_class_names


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        image_size=config.IMG_SIZE,
        batch_size=config.IMG_BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=config.RANDOM_SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        image_size=config.IMG_SIZE,
        batch_size=config.IMG_BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )
    class_names = train_ds.class_names
    save_class_names(class_names)

    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augment",
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (
        train_ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(num_classes: int):
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*config.IMG_SIZE, config.IMG_CHANNELS),
    )
    base.trainable = False

    inputs = layers.Input(shape=(*config.IMG_SIZE, config.IMG_CHANNELS))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="efficientnetb0_plantvillage")
    return model, base


def plot_history(history_dicts, out_path: Path):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in history_dicts:
        acc += h["accuracy"]
        val_acc += h["val_accuracy"]
        loss += h["loss"]
        val_loss += h["val_loss"]

    epochs = range(1, len(acc) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, acc, label="train")
    axes[0].plot(epochs, val_acc, label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, loss, label="train")
    axes[1].plot(epochs, val_loss, label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ensure_dir(config.IMAGE_MODEL_PATH.parent)
    ensure_dir(config.FIGURES_DIR)

    print("Loading datasets...")
    train_ds, val_ds, class_names = build_datasets()
    print(f"  classes: {len(class_names)}")

    print("Building model...")
    model, base = build_model(len(class_names))

    # ---- Stage 1: train classification head only ----
    model.compile(
        optimizer=optimizers.Adam(config.IMG_LR_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\nStage 1 - head training ({config.IMG_EPOCHS_HEAD} epochs)")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=config.IMG_EPOCHS_HEAD)

    # ---- Stage 2: unfreeze top N layers and fine-tune ----
    base.trainable = True
    for layer in base.layers[: -config.IMG_FINETUNE_LAYERS]:
        layer.trainable = False
    model.compile(
        optimizer=optimizers.Adam(config.IMG_LR_FINETUNE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\nStage 2 - fine-tuning top {config.IMG_FINETUNE_LAYERS} layers "
          f"({config.IMG_EPOCHS_FINETUNE} epochs)")
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=config.IMG_EPOCHS_FINETUNE)

    # ---- Save ----
    model.save(config.IMAGE_MODEL_PATH)
    history_path = config.IMAGE_MODEL_PATH.parent / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({"stage1": h1.history, "stage2": h2.history}, f, indent=2)
    plot_history([h1.history, h2.history], config.FIGURES_DIR / "image_training_curves.png")

    print(f"\nSaved model     -> {config.IMAGE_MODEL_PATH}")
    print(f"Saved history   -> {history_path}")
    print(f"Saved curves    -> {config.FIGURES_DIR / 'image_training_curves.png'}")


if __name__ == "__main__":
    main()
