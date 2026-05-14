"""
train_teacher.py
────────────────
Trains the CNN-BiLSTM teacher network to convergence.
Saves:
  saved_models/teacher_best.keras  — best val accuracy checkpoint
  saved_models/teacher_final.keras — end of training

Run:  python training/train_teacher.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

from config import (
    PROCESSED_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR,
    BATCH_SIZE, TEACHER_EPOCHS, LEARNING_RATE,
    CLASS_WEIGHTS, CLASSES, NUM_CLASSES, RANDOM_SEED
)
from models.teacher import build_teacher
from training.losses import WeightedCrossEntropy


tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_split(split: str):
    path = os.path.join(PROCESSED_DIR, f"{split}.npz")
    data = np.load(path)
    X = data["X_ecg"][..., np.newaxis]   # add channel dim: (N, 360, 1)
    y = data["y"]
    return X, y


def make_dataset(X, y, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=RANDOM_SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Teacher — Loss")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history.history["accuracy"],     label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Teacher — Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves → {save_path}")
    plt.close()


def train():
    print("=" * 60)
    print("  TEACHER TRAINING")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────
    print("Loading data …")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}")

    train_ds = make_dataset(X_train, y_train, shuffle=True)
    val_ds   = make_dataset(X_val,   y_val,   shuffle=False)

    # ── Build model ──────────────────────────────────────────
    teacher, _ = build_teacher()
    teacher.summary()

    teacher.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=WeightedCrossEntropy(CLASS_WEIGHTS),
        metrics=["accuracy"],
    )

    # ── Callbacks ────────────────────────────────────────────
    best_path  = os.path.join(MODELS_DIR, "teacher_best.keras")
    final_path = os.path.join(MODELS_DIR, "teacher_final.keras")
    log_path   = os.path.join(LOGS_DIR, "teacher")

    cb_list = [
        callbacks.ModelCheckpoint(
            best_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12,
            restore_best_weights=True, verbose=1
        ),
        callbacks.TensorBoard(log_dir=log_path, histogram_freq=0),
        callbacks.CSVLogger(os.path.join(LOGS_DIR, "teacher_history.csv")),
    ]

    # ── Train ────────────────────────────────────────────────
    history = teacher.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TEACHER_EPOCHS,
        callbacks=cb_list,
        verbose=1,
    )

    teacher.save(final_path)
    print(f"\nFinal model → {final_path}")
    print(f"Best  model → {best_path}")

    # ── Plot ─────────────────────────────────────────────────
    fig_path = os.path.join(FIGURES_DIR, "teacher_training.png")
    plot_history(history, fig_path)

    # ── Quick val metrics ─────────────────────────────────────
    best_teacher = tf.keras.models.load_model(best_path)
    val_loss, val_acc = best_teacher.evaluate(val_ds, verbose=0)
    print(f"\nBest val accuracy: {val_acc:.4f}  |  val loss: {val_loss:.4f}")


if __name__ == "__main__":
    train()
