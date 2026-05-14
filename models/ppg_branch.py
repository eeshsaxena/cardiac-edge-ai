"""
ppg_branch.py
─────────────
Lightweight PPG sub-network. Takes a 360-sample PPG window
and produces a 32-dim feature vector for late fusion with ECG.

Kept lighter than ECG branch (PPG morphology is simpler).
Target: ~8K parameters.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import WINDOW_LEN, PPG_FILTERS


def build_ppg_branch(input_length: int = WINDOW_LEN) -> Model:
    """
    Returns a Model that maps (batch, 360, 1) PPG → (batch, 32) features.
    """
    inp = layers.Input(shape=(input_length, 1), name="ppg_input")

    # Block 1 — coarse waveform shape
    x = layers.Conv1D(PPG_FILTERS[0], 9, padding="same", name="ppg_conv1")(inp)
    x = layers.BatchNormalization(name="ppg_bn1")(x)
    x = layers.Activation("relu", name="ppg_relu1")(x)
    x = layers.MaxPooling1D(2, name="ppg_pool1")(x)

    # Block 2 — fine pulse morphology (dicrotic notch)
    x = layers.Conv1D(PPG_FILTERS[1], 5, padding="same", name="ppg_conv2")(x)
    x = layers.BatchNormalization(name="ppg_bn2")(x)
    x = layers.Activation("relu", name="ppg_relu2")(x)
    x = layers.MaxPooling1D(2, name="ppg_pool2")(x)

    # Global context
    x = layers.GlobalAveragePooling1D(name="ppg_gap")(x)
    x = layers.Dense(32, activation="relu", name="ppg_fc")(x)

    return Model(inp, x, name="ppg_branch")


if __name__ == "__main__":
    model = build_ppg_branch()
    model.summary()
    print(f"PPG branch parameters: {model.count_params():,}")
