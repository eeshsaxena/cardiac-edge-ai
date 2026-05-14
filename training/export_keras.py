"""
export_keras.py  —  Properly transfer PyTorch student weights to Keras.
The naive structural approach failed because SeparableConv1D stores
depthwise+pointwise as one layer with different axis ordering vs PyTorch.

PyTorch DWSBlock(in, out, k):
  dw: Conv1d(in, in, k, groups=in)  → weight (in, 1, k)
  pw: Conv1d(in, out, 1)            → weight (out, in, 1)
  bn: BN(out)                        → gamma, beta, mean, var

Keras SeparableConv1D(out, k):
  depthwise_kernel:  (k, in, 1)      ← transpose dw: (in,1,k) → (k,in,1)
  pointwise_kernel:  (1, in, out)    ← transpose pw: (out,in,1) → (1,in,out)
  pointwise_bias:    (out,)          ← zeros (PyTorch has no bias in DWS)
  BN: gamma, beta, mean, var         ← same shapes, direct copy

Run: python training/export_keras.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

from config import (
    MODELS_DIR, NUM_CLASSES, STUDENT_FILTERS, STUDENT_FC_UNITS,
    WINDOW_LEN, PROCESSED_DIR
)
from models.student import build_student_with_intermediate


# ── PyTorch model (same as training script) ───────────────────────────────────

class DWSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel, padding=kernel//2,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        f = STUDENT_FILTERS
        self.blk1  = DWSBlock(1,    f[0], 7)
        self.pool1 = nn.MaxPool1d(2)
        self.blk2  = DWSBlock(f[0], f[1], 5)
        self.pool2 = nn.MaxPool1d(2)
        self.blk3  = DWSBlock(f[1], f[2], 3)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc1   = nn.Linear(f[2], STUDENT_FC_UNITS)
        self.fc2   = nn.Linear(STUDENT_FC_UNITS, NUM_CLASSES)
    def forward(self, x):
        x = self.pool1(self.blk1(x))
        x = self.pool2(self.blk2(x))
        x = self.gap(self.blk3(x)).squeeze(-1)
        return self.fc2(F.relu(self.fc1(x)))


def get_pytorch_weights(pt_path):
    """Load .pt checkpoint and return ordered list of (name, numpy_array) tuples."""
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    sd = ckpt["model_state"]
    return [(k, v.numpy()) for k, v in sd.items()]


def transfer_block_weights(keras_sep_layer, keras_bn_layer,
                            dw_w, pw_w,
                            bn_gamma, bn_beta, bn_mean, bn_var):
    """
    Transfer one DWSBlock from PyTorch to a Keras (SeparableConv1D, BN) pair.

    PyTorch shapes → Keras shapes:
      dw_w:  (in_ch, 1, kernel)    → depthwise_kernel: (kernel, in_ch, 1)
      pw_w:  (out_ch, in_ch, 1)    → pointwise_kernel: (1, in_ch, out_ch)
      bias:  zeros                  → pointwise_bias:   (out_ch,)
    """
    k_dw = dw_w.transpose(2, 1, 0)          # (kernel, 1, in_ch) ... wait
    # dw_w shape: (in_ch, 1, kernel) because groups=in_ch gives each channel its own filter
    # Keras depthwise: (kernel_size, in_channels, depth_multiplier)
    # So: dw_w (in_ch, 1, k) → we need (k, in_ch, 1)
    k_dw = dw_w.transpose(2, 0, 1)          # (k, in_ch, 1) ✓

    # pw_w shape: (out_ch, in_ch, 1) → Keras pointwise (1, in_ch, out_ch)
    k_pw = pw_w.transpose(2, 1, 0)          # (1, in_ch, out_ch) ✓

    pb = np.zeros(pw_w.shape[0], dtype=np.float32)   # pointwise bias = zeros

    keras_sep_layer.set_weights([k_dw, k_pw, pb])
    keras_bn_layer.set_weights([bn_gamma, bn_beta, bn_mean, bn_var])
    print(f"    {keras_sep_layer.name}: dw{dw_w.shape}->{k_dw.shape}  "
          f"pw{pw_w.shape}->{k_pw.shape}")


def transfer_all(pt_path, out_keras_path):
    print(f"\nLoading PyTorch checkpoint: {pt_path}")
    sd_list = get_pytorch_weights(pt_path)
    sd = dict(sd_list)
    print(f"PyTorch keys: {list(sd.keys())}")

    # Build Keras model
    keras_model, _ = build_student_with_intermediate()
    print(f"Keras layers with weights: "
          f"{[l.name for l in keras_model.layers if l.get_weights()]}")

    # ── Block 1: DWSBlock(1, 32, 7) ──────────────────────────────────────────
    sep1 = keras_model.get_layer("s_blk1_dws")
    bn1  = keras_model.get_layer("s_blk1_bn")
    transfer_block_weights(
        sep1, bn1,
        sd["blk1.dw.weight"],                    # (1, 1, 7)
        sd["blk1.pw.weight"],                    # (32, 1, 1)
        sd["blk1.bn.weight"], sd["blk1.bn.bias"],
        sd["blk1.bn.running_mean"], sd["blk1.bn.running_var"],
    )

    # ── Block 2: DWSBlock(32, 64, 5) ─────────────────────────────────────────
    sep2 = keras_model.get_layer("s_blk2_dws")
    bn2  = keras_model.get_layer("s_blk2_bn")
    transfer_block_weights(
        sep2, bn2,
        sd["blk2.dw.weight"],                    # (32, 1, 5)
        sd["blk2.pw.weight"],                    # (64, 32, 1)
        sd["blk2.bn.weight"], sd["blk2.bn.bias"],
        sd["blk2.bn.running_mean"], sd["blk2.bn.running_var"],
    )

    # ── Block 3: DWSBlock(64, 64, 3) ─────────────────────────────────────────
    sep3 = keras_model.get_layer("s_blk3_dws")
    bn3  = keras_model.get_layer("s_blk3_bn")
    transfer_block_weights(
        sep3, bn3,
        sd["blk3.dw.weight"],                    # (64, 1, 3)
        sd["blk3.pw.weight"],                    # (64, 64, 1)
        sd["blk3.bn.weight"], sd["blk3.bn.bias"],
        sd["blk3.bn.running_mean"], sd["blk3.bn.running_var"],
    )

    # ── FC layers ─────────────────────────────────────────────────────────────
    # PyTorch Linear(in, out): weight (out, in), bias (out,)
    # Keras Dense(out):        kernel (in, out), bias (out,)
    fc1 = keras_model.get_layer("s_fc1")
    fc1.set_weights([
        sd["fc1.weight"].T,       # (64, 32) ← transpose of (32, 64)
        sd["fc1.bias"],           # (32,)
    ])
    print(f"    s_fc1: {sd['fc1.weight'].shape} -> {sd['fc1.weight'].T.shape}")

    fc2 = keras_model.get_layer("s_output")
    fc2.set_weights([
        sd["fc2.weight"].T,       # (32, 5) ← transpose of (5, 32)
        sd["fc2.bias"],           # (5,)
    ])
    print(f"    s_output: {sd['fc2.weight'].shape} -> {sd['fc2.weight'].T.shape}")

    # ── Quick accuracy check ──────────────────────────────────────────────────
    print("\nVerifying accuracy on 500 test samples ...")
    data   = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
    X_samp = data["X_ecg"][:500][..., np.newaxis].astype("float32")
    y_samp = data["y"][:500]
    preds  = keras_model.predict(X_samp, verbose=0).argmax(1)
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_samp, preds)
    f1  = f1_score(y_samp, preds, average="macro", zero_division=0)
    print(f"  Sample Acc: {acc*100:.2f}%  Macro-F1: {f1:.4f}")
    if f1 < 0.70:
        print("  [WARN] F1 still low — check layer name mapping.")
    else:
        print("  [OK] Weight transfer successful!")

    # ── Save ─────────────────────────────────────────────────────────────────
    keras_model.save(out_keras_path)
    print(f"\nSaved -> {out_keras_path}")
    return keras_model, f1


def main():
    variants = [
        ("student_ce_only_torch.pt",  "student_ce_only.keras"),
        ("student_ce_kl_torch.pt",    "student_ce_kl.keras"),
        ("student_full_kd_torch.pt",  "student_full_kd.keras"),
    ]
    for pt_file, keras_file in variants:
        pt_path    = os.path.join(MODELS_DIR, pt_file)
        keras_path = os.path.join(MODELS_DIR, keras_file)
        if not os.path.exists(pt_path):
            print(f"[SKIP] {pt_file} not found"); continue
        print(f"\n{'='*60}")
        print(f"  {pt_file}  →  {keras_file}")
        print(f"{'='*60}")
        _, f1 = transfer_all(pt_path, keras_path)
        print(f"Final F1: {f1:.4f}")


if __name__ == "__main__":
    main()
