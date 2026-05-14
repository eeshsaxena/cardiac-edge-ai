"""
teacher.py
──────────
CNN-BiLSTM teacher network for 5-class arrhythmia classification.
Trained on GPU; saved feature maps are used for L_spectral distillation.

Architecture:
  Input (360, 1)
  → Conv block × 4  (depthwise-standard, increasing filters)
  → Bidirectional LSTM (128 units)
  → Dense(128) + Dropout
  → Dense(5) + Softmax

Also exposes a 'feature_extractor' sub-model that returns the last
conv block's output — used to compute L_spectral during distillation.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import WINDOW_LEN, NUM_CLASSES, TEACHER_FILTERS, LSTM_UNITS


def conv_block(x, filters: int, kernel_size: int, name_prefix: str):
    """Conv1D + BatchNorm + ReLU."""
    x = layers.Conv1D(
        filters, kernel_size, padding="same",
        name=f"{name_prefix}_conv"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu")(x)
    return x


def build_teacher(input_length: int = WINDOW_LEN) -> tuple[Model, Model]:
    """
    Returns:
        teacher_model        — full classification model
        feature_extractor    — sub-model up to last conv block output
                               (used for L_spectral computation)
    """
    inp = layers.Input(shape=(input_length, 1), name="ecg_input")

    # ── Block 1 ──────────────────────────────────────────────
    x = conv_block(inp, TEACHER_FILTERS[0], 7, "blk1")
    x = conv_block(x,   TEACHER_FILTERS[0], 7, "blk1b")
    x = layers.MaxPooling1D(2, name="pool1")(x)
    x = layers.Dropout(0.1, name="drop1")(x)

    # ── Block 2 ──────────────────────────────────────────────
    x = conv_block(x, TEACHER_FILTERS[1], 5, "blk2")
    x = conv_block(x, TEACHER_FILTERS[1], 5, "blk2b")
    x = layers.MaxPooling1D(2, name="pool2")(x)
    x = layers.Dropout(0.1, name="drop2")(x)

    # ── Block 3 ──────────────────────────────────────────────
    x = conv_block(x, TEACHER_FILTERS[2], 3, "blk3")
    x = conv_block(x, TEACHER_FILTERS[2], 3, "blk3b")
    x = layers.MaxPooling1D(2, name="pool3")(x)
    x = layers.Dropout(0.15, name="drop3")(x)

    # ── Block 4 (feature layer for distillation) ─────────────
    feat = conv_block(x, TEACHER_FILTERS[3], 3, "blk4")
    feat = conv_block(feat, TEACHER_FILTERS[3], 3, "blk4b")
    # feat shape: (batch, T//8, 256) — this is used for L_spectral

    # ── BiLSTM ───────────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=False), name="bilstm"
    )(feat)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="drop_fc")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    teacher_model     = Model(inp, out,  name="teacher")
    feature_extractor = Model(inp, feat, name="teacher_features")

    return teacher_model, feature_extractor


if __name__ == "__main__":
    model, feat_model = build_teacher()
    model.summary()
    print(f"\nFeature extractor output shape: {feat_model.output_shape}")
    total = model.count_params()
    print(f"Total parameters: {total:,}")
