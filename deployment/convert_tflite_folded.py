"""
convert_tflite_folded.py
────────────────────────
Convert PyTorch student models to TFLite by:
  1. Folding BatchNorm into preceding Conv weights (standard MCU practice)
  2. Building a pure TF graph using only TFLite-native ops
  3. Converting to INT8 with calibration data

BN folding eliminates the Keras 3 Cast issue completely.
The folded model has the IDENTICAL numerical output but fewer ops.

Run: python deployment/convert_tflite_folded.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from config import (
    PROCESSED_DIR, MODELS_DIR, TFLITE_DIR,
    NUM_CLASSES, STUDENT_FILTERS, STUDENT_FC_UNITS, WINDOW_LEN
)


# ── BatchNorm Folding ─────────────────────────────────────────────────────────

def fold_bn_into_conv(w, gamma, beta, mean, var, eps=1e-5):
    """
    Fold BatchNorm params into preceding conv weight+bias.

    Args:
        w:     conv weight numpy (out_ch, in_ch, k) — PyTorch layout
        gamma: BN scale  (out_ch,)
        beta:  BN shift  (out_ch,)
        mean:  BN running mean  (out_ch,)
        var:   BN running var   (out_ch,)
    Returns:
        w_new: folded weight (out_ch, in_ch, k)
        b_new: folded bias   (out_ch,)
    """
    std   = np.sqrt(var + eps)
    scale = gamma / std                              # (out_ch,)
    # broadcast scale over (in_ch, k) dims
    w_new = w * scale[:, np.newaxis, np.newaxis]
    b_new = -mean * scale + beta
    return w_new.astype(np.float32), b_new.astype(np.float32)


def extract_folded_weights(pt_path):
    """
    Load PyTorch checkpoint and return folded weights dict.
    Returns dict with keys: blk1, blk2, blk3, fc1, fc2
    Each block: {dw_w, dw_b, pw_w, pw_b}
    """
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    sd   = {k: v.numpy() for k, v in ckpt["model_state"].items()}

    blocks = {}
    for name, (in_ch, out_ch, k) in [
        ("blk1", (1,  STUDENT_FILTERS[0], 7)),
        ("blk2", (STUDENT_FILTERS[0], STUDENT_FILTERS[1], 5)),
        ("blk3", (STUDENT_FILTERS[1], STUDENT_FILTERS[2], 3)),
    ]:
        # Depthwise: dw.weight (in_ch, 1, k) → fold BN (out_ch == in_ch for dw)
        dw_w = sd[f"{name}.dw.weight"]            # (in_ch, 1, k)
        pw_w = sd[f"{name}.pw.weight"]            # (out_ch, in_ch, 1)
        bn_g = sd[f"{name}.bn.weight"]            # (out_ch,)
        bn_b = sd[f"{name}.bn.bias"]
        bn_m = sd[f"{name}.bn.running_mean"]
        bn_v = sd[f"{name}.bn.running_var"]

        # BN is after pointwise conv, so fold into pw_w
        pw_f, pw_b = fold_bn_into_conv(pw_w, bn_g, bn_b, bn_m, bn_v)

        blocks[name] = {
            "dw_w": dw_w,    # (in_ch, 1, k)  — no bias, no BN after dw
            "pw_w": pw_f,    # (out_ch, in_ch, 1)  — BN folded in
            "pw_b": pw_b,    # (out_ch,)
        }

    return blocks, {
        "fc1_w": sd["fc1.weight"],  # (32, 64)
        "fc1_b": sd["fc1.bias"],
        "fc2_w": sd["fc2.weight"],  # (5, 32)
        "fc2_b": sd["fc2.bias"],
    }


# ── Build TF concrete function (BatchNorm-free) ───────────────────────────────

def build_tf_model(blocks, fcs):
    """
    Manually implement the student network using tf.nn ops.
    All ops are TFLite-native: conv, depthwise_conv, max_pool, dense, relu, softmax.
    """
    f = STUDENT_FILTERS

    # Convert PyTorch weights to TF layout:
    #   PyTorch Conv1d: (out, in, k) → TF Conv1D: (k, in, out)
    #   Depthwise PyTorch: (in, 1, k) → TF DepthwiseConv1D: (k, in, 1)
    def pt2tf_conv(w):    return w.transpose(2, 1, 0)   # (out,in,k) → (k,in,out)
    def pt2tf_dw(w):                                       # (in,1,k) → (k,1,in,1) for depthwise_conv2d
        return w.transpose(2, 1, 0)[:, :, :, np.newaxis]  # ✓

    blk_tf = {}
    for name in ["blk1", "blk2", "blk3"]:
        blk_tf[name] = {
            "dw": tf.constant(pt2tf_dw(blocks[name]["dw_w"])),   # (k, in, 1)
            "pw": tf.constant(pt2tf_conv(blocks[name]["pw_w"])),  # (1, in, out)
            "pb": tf.constant(blocks[name]["pw_b"]),
        }

    # FC: PyTorch (out, in) → TF (in, out)
    fc1_w = tf.constant(fcs["fc1_w"].T)
    fc1_b = tf.constant(fcs["fc1_b"])
    fc2_w = tf.constant(fcs["fc2_w"].T)
    fc2_b = tf.constant(fcs["fc2_b"])

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, WINDOW_LEN, 1], dtype=tf.float32)
    ])
    def infer(x):
        # Block 1: DWS + ReLU + MaxPool
        x = tf.nn.depthwise_conv2d(
            tf.expand_dims(x, 2),
            blk_tf["blk1"]["dw"],        # (7, 1, 1, 1) — already 4D
            strides=[1,1,1,1], padding="SAME"
        )[:, :, 0, :]
        x = tf.nn.conv1d(x, blk_tf["blk1"]["pw"], stride=1, padding="SAME")
        x = tf.nn.bias_add(x, blk_tf["blk1"]["pb"])
        x = tf.nn.relu(x)
        x = tf.nn.max_pool1d(x, ksize=2, strides=2, padding="VALID")

        # Block 2: DWS + ReLU + MaxPool
        x = tf.nn.depthwise_conv2d(
            tf.expand_dims(x, 2),
            blk_tf["blk2"]["dw"],        # (5, 1, 32, 1) — already 4D
            strides=[1,1,1,1], padding="SAME"
        )[:, :, 0, :]
        x = tf.nn.conv1d(x, blk_tf["blk2"]["pw"], stride=1, padding="SAME")
        x = tf.nn.bias_add(x, blk_tf["blk2"]["pb"])
        x = tf.nn.relu(x)
        x = tf.nn.max_pool1d(x, ksize=2, strides=2, padding="VALID")

        # Block 3: DWS + ReLU
        x = tf.nn.depthwise_conv2d(
            tf.expand_dims(x, 2),
            blk_tf["blk3"]["dw"],        # (3, 1, 64, 1) — already 4D
            strides=[1,1,1,1], padding="SAME"
        )[:, :, 0, :]
        x = tf.nn.conv1d(x, blk_tf["blk3"]["pw"], stride=1, padding="SAME")
        x = tf.nn.bias_add(x, blk_tf["blk3"]["pb"])
        x = tf.nn.relu(x)

        # Global Average Pool → (1, C)
        x = tf.reduce_mean(x, axis=1)

        # FC1 + ReLU
        x = tf.matmul(x, fc1_w) + fc1_b
        x = tf.nn.relu(x)

        # FC2 + Softmax
        x = tf.matmul(x, fc2_w) + fc2_b
        return tf.nn.softmax(x)

    return infer


# ── Verify accuracy ───────────────────────────────────────────────────────────

def verify(infer_fn, X_test, y_test, max_samples=500):
    preds = []
    for i in range(min(max_samples, len(X_test))):
        p = infer_fn(X_test[i:i+1]).numpy().argmax(1)
        preds.append(p[0])
    preds = np.array(preds)
    y = y_test[:max_samples]
    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds, average="macro", zero_division=0)
    return acc, f1


# ── Convert ───────────────────────────────────────────────────────────────────

def convert_variant(pt_path, slug, X_calib, X_test, y_test):
    print(f"\n{'='*55}")
    print(f"  {os.path.basename(pt_path)}  →  {slug}")
    print(f"{'='*55}")

    blocks, fcs = extract_folded_weights(pt_path)
    infer       = build_tf_model(blocks, fcs)
    concrete    = infer.get_concrete_function()

    # Verify BN-folded accuracy
    acc, f1 = verify(infer, X_test, y_test)
    print(f"  BN-folded accuracy check: Acc={acc*100:.2f}%  F1={f1:.4f}")
    if f1 < 0.80:
        print("  [WARN] Low F1 after folding — check weight mapping")

    # ── FP32 TFLite ────────────────────────────────────────────────────────
    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    tflite_fp32 = conv.convert()
    fp32_path = os.path.join(TFLITE_DIR, f"{slug}_fp32.tflite")
    with open(fp32_path, "wb") as f:
        f.write(tflite_fp32)
    print(f"  FP32 → {os.path.basename(fp32_path)}  ({len(tflite_fp32)/1024:.1f} KB)")

    # ── INT8 TFLite ────────────────────────────────────────────────────────
    idx = np.random.choice(len(X_calib), min(200, len(X_calib)), replace=False)
    def rep_ds():
        for i in idx:
            yield [X_calib[i:i+1]]

    conv2 = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    conv2.optimizations = [tf.lite.Optimize.DEFAULT]
    conv2.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,   # fallback for depthwise expand
    ]
    conv2.inference_input_type  = tf.float32  # easier for Arduino
    conv2.inference_output_type = tf.float32
    conv2.representative_dataset = rep_ds

    try:
        tflite_int8 = conv2.convert()
        int8_path = os.path.join(TFLITE_DIR, f"{slug}_int8.tflite")
        with open(int8_path, "wb") as f:
            f.write(tflite_int8)
        print(f"  INT8 → {os.path.basename(int8_path)}  ({len(tflite_int8)/1024:.1f} KB)")
        generate_header(tflite_int8, slug)
        return fp32_path, int8_path
    except Exception as e:
        print(f"  [WARN] INT8 failed: {e}")
        print(f"  Using FP32 for header.")
        generate_header(tflite_fp32, slug + "_fp32")
        return fp32_path, None


def generate_header(data, slug):
    var = slug.replace("-", "_")
    lines = [
        f"// CardioEdge TFLite model: {slug}",
        f"// Size: {len(data)} bytes ({len(data)/1024:.1f} KB)",
        "#pragma once",
        "#include <stdint.h>",
        f"const unsigned int {var}_model_len = {len(data)};",
        f"alignas(8) const uint8_t {var}_model[] = {{",
    ]
    hex_vals = [f"0x{b:02x}" for b in data]
    for i in range(0, len(hex_vals), 12):
        lines.append("  " + ", ".join(hex_vals[i:i+12]) + ",")
    lines.append("};")
    h_path = os.path.join(TFLITE_DIR, f"{slug}_model.h")
    with open(h_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Header → {os.path.basename(h_path)}")


def main():
    print("=" * 60)
    print("  TFLITE CONVERSION — BN-Folded PyTorch Student Models")
    print("=" * 60)

    data  = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
    X_ecg = data["X_ecg"][..., np.newaxis].astype(np.float32)  # (N, 360, 1)
    y_test = data["y"]
    print(f"Calibration/test samples: {len(X_ecg):,}")

    variants = [
        ("student_ce_only_torch.pt",  "student_ce"),
        ("student_ce_kl_torch.pt",    "student_ce_kl"),
        ("student_full_kd_torch.pt",  "student_kd"),
    ]

    results = []
    for pt_file, slug in variants:
        pt_path = os.path.join(MODELS_DIR, pt_file)
        if not os.path.exists(pt_path):
            print(f"[SKIP] {pt_file} not found"); continue
        fp32_p, int8_p = convert_variant(pt_path, slug, X_ecg, X_ecg, y_test)
        fp32_kb = os.path.getsize(fp32_p) / 1024
        int8_kb = os.path.getsize(int8_p) / 1024 if int8_p else 0
        results.append((slug, fp32_kb, int8_kb))

    print(f"\n{'='*60}")
    print("  DEPLOYMENT SIZE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'FP32 (KB)':>10} {'INT8 (KB)':>10} {'Compression':>12}")
    print(f"  {'-'*54}")
    for slug, fp32, int8 in results:
        ratio = f"{fp32/int8:.1f}×" if int8 > 0 else "N/A"
        print(f"  {slug:<20} {fp32:>10.1f} {int8:>10.1f} {ratio:>12}")

    # Nano 33 BLE constraints check
    print(f"\n  Arduino Nano 33 BLE constraints:")
    print(f"    Flash: 1024 KB  RAM: 256 KB")
    for slug, fp32, int8 in results:
        fits = "✅ FITS" if int8 < 800 else "❌ TOO LARGE"
        print(f"    {slug}: {int8:.1f} KB  {fits}")


if __name__ == "__main__":
    main()
