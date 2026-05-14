"""
student.py
──────────
TinyConv student network — MCU-deployable ECG branch.
Uses depthwise-separable convolutions for 40× param reduction vs teacher.

Architecture:
  Input (360, 1)
  → DWS-Conv block × 3
  → GlobalAveragePooling
  → Dense(32) + ReLU
  → Dense(5) + Softmax   [classification head]
  OR
  → 64-dim feature vector [for late fusion with PPG]

Parameter target: ~45K  (fits in Arduino Nano 33 BLE 1MB flash as INT8)
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import WINDOW_LEN, NUM_CLASSES, STUDENT_FILTERS, STUDENT_FC_UNITS


def dws_conv_block(x, filters: int, kernel_size: int, name_prefix: str):
    """
    Depthwise-Separable Conv1D:
      DepthwiseConv1D (spatial filtering per channel)
      → PointwiseConv1D (channel mixing)
      → BatchNorm → ReLU
    Saves ~8-9× FLOPs vs standard Conv for equivalent output.
    """
    # Depthwise
    x = layers.DepthwiseConv2D(
        kernel_size=(kernel_size, 1), padding="same",
        name=f"{name_prefix}_dw"
    )(tf.expand_dims(x, axis=2))
    x = tf.squeeze(x, axis=2)
    # Pointwise
    x = layers.Conv1D(filters, 1, padding="same", name=f"{name_prefix}_pw")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu")(x)
    return x


def build_student(
    input_length: int = WINDOW_LEN,
    return_features: bool = False,
) -> Model:
    """
    Args:
        return_features: If True, model outputs a 64-dim feature vector
                         (for PPG fusion). If False, outputs 5-class softmax.
    """
    inp = layers.Input(shape=(input_length, 1), name="ecg_input_student")

    # ── Block 1 ──────────────────────────────────────────────
    x = dws_conv_block(inp, STUDENT_FILTERS[0], 7, "s_blk1")
    x = layers.MaxPooling1D(2, name="s_pool1")(x)

    # ── Block 2 ──────────────────────────────────────────────
    x = dws_conv_block(x, STUDENT_FILTERS[1], 5, "s_blk2")
    x = layers.MaxPooling1D(2, name="s_pool2")(x)

    # ── Block 3 (feature layer — mirrors teacher's blk4 for L_spectral) ──
    feat = dws_conv_block(x, STUDENT_FILTERS[2], 3, "s_blk3")
    # feat shape: (batch, T//4, 64)

    # ── Head ─────────────────────────────────────────────────
    x = layers.GlobalAveragePooling1D(name="s_gap")(feat)
    x = layers.Dense(STUDENT_FC_UNITS, activation="relu", name="s_fc1")(x)

    if return_features:
        # Return 64-dim embedding for fusion module
        model = Model(inp, x, name="student_ecg_features")
    else:
        out = layers.Dense(NUM_CLASSES, activation="softmax", name="s_output")(x)
        model = Model(inp, out, name="student")

    return model


def build_student_with_intermediate() -> tuple[Model, Model]:
    """
    Returns:
        student_model      — full classification model
        feat_extractor     — outputs conv feature maps (for L_spectral)
    """
    inp = layers.Input(shape=(WINDOW_LEN, 1), name="ecg_input_student")

    x = dws_conv_block(inp, STUDENT_FILTERS[0], 7, "s_blk1")
    x = layers.MaxPooling1D(2, name="s_pool1")(x)

    x = dws_conv_block(x, STUDENT_FILTERS[1], 5, "s_blk2")
    x = layers.MaxPooling1D(2, name="s_pool2")(x)

    feat = dws_conv_block(x, STUDENT_FILTERS[2], 3, "s_blk3")

    gap = layers.GlobalAveragePooling1D(name="s_gap")(feat)
    fc  = layers.Dense(STUDENT_FC_UNITS, activation="relu", name="s_fc1")(gap)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="s_output")(fc)

    student_model  = Model(inp, out,  name="student")
    feat_extractor = Model(inp, feat, name="student_features")

    return student_model, feat_extractor


if __name__ == "__main__":
    model, feats = build_student_with_intermediate()
    model.summary()
    print(f"\nFeature extractor output shape: {feats.output_shape}")
    print(f"Total parameters: {model.count_params():,}")
