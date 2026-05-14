"""
fusion.py
─────────
Late fusion model: combines ECG student (64-dim) + PPG branch (32-dim)
via a learned scalar α and a small MLP classification head.

Architecture:
  ECG features (64) ──┐
                       → Concat(96) → Dense(64) → Dense(5) + Softmax
  PPG features (32) ──┘

  α = sigmoid(Dense(1)(concat)) ∈ [0, 1]
  Interpretation: how much PPG contributes to the decision.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import WINDOW_LEN, NUM_CLASSES, FUSION_FC_UNITS
from models.student   import build_student
from models.ppg_branch import build_ppg_branch


def build_fusion_model() -> tuple[Model, Model]:
    """
    Returns:
        fusion_model  — full model: (ecg_input, ppg_input) → (probs, alpha)
        alpha_model   — sub-model to extract alpha values for analysis
    """
    # Sub-networks
    ecg_branch = build_student(return_features=True)  # output: 64-dim
    ppg_branch  = build_ppg_branch()                   # output: 32-dim

    # Inputs
    ecg_inp = layers.Input(shape=(WINDOW_LEN, 1), name="fusion_ecg_input")
    ppg_inp = layers.Input(shape=(WINDOW_LEN, 1), name="fusion_ppg_input")

    ecg_feat = ecg_branch(ecg_inp)   # (batch, 64)
    ppg_feat = ppg_branch(ppg_inp)   # (batch, 32)

    # Concatenate
    concat = layers.Concatenate(name="feature_concat")([ecg_feat, ppg_feat])  # (batch, 96)

    # Learned scalar α — how much ECG vs PPG contributes
    alpha = layers.Dense(1, activation="sigmoid", name="alpha")(concat)  # (batch, 1)

    # Weighted combination of branch logits (before softmax, for α)
    ecg_logits = layers.Dense(NUM_CLASSES, name="ecg_logits")(ecg_feat)  # (batch, 5)
    ppg_logits = layers.Dense(NUM_CLASSES, name="ppg_logits")(ppg_feat)  # (batch, 5)

    # α-blended logits
    fused_logits = layers.Lambda(
        lambda inputs: inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2],
        name="alpha_blend"
    )([alpha, ecg_logits, ppg_logits])

    # MLP fusion head on concatenated features
    x = layers.Dense(FUSION_FC_UNITS, activation="relu", name="fusion_fc1")(concat)
    x = layers.Dropout(0.2, name="fusion_drop")(x)
    mlp_logits = layers.Dense(NUM_CLASSES, name="mlp_logits")(x)

    # Final: average fused + MLP logits, then softmax
    final_logits = layers.Average(name="final_avg")([fused_logits, mlp_logits])
    probs = layers.Softmax(name="output")(final_logits)

    fusion_model = Model(
        inputs=[ecg_inp, ppg_inp],
        outputs=[probs, alpha],
        name="fusion_model"
    )

    return fusion_model, ecg_branch, ppg_branch


if __name__ == "__main__":
    model, ecg_b, ppg_b = build_fusion_model()
    model.summary()
    print(f"\nECG branch params : {ecg_b.count_params():,}")
    print(f"PPG branch params : {ppg_b.count_params():,}")
    print(f"Total fusion params: {model.count_params():,}")
