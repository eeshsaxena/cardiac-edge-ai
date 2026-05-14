"""
convert_tflite.py — INT8 quantization + TFLite conversion.
Run: python deployment/convert_tflite.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import tensorflow as tf
from config import PROCESSED_DIR, MODELS_DIR, TFLITE_DIR, BATCH_SIZE
import training.losses  # registers WeightedCrossEntropy with Keras serialization


def representative_dataset_gen(X_ecg, n_samples=200):
    """Yield sample inputs for INT8 calibration."""
    indices = np.random.choice(len(X_ecg), min(n_samples, len(X_ecg)), replace=False)
    for i in indices:
        yield [X_ecg[i:i+1]]


def convert_model(model_path: str, out_name: str, X_calib: np.ndarray,
                  input_key: str = None, X_calib_ppg: np.ndarray = None):
    if not os.path.exists(model_path):
        print(f"[SKIP] {model_path} not found.")
        return

    print(f"\nConverting: {os.path.basename(model_path)}")

    # Force float32 globally — prevents Keras 3 BatchNorm Cast ops breaking MLIR
    import keras
    keras.mixed_precision.set_global_policy("float32")

    model = tf.keras.models.load_model(model_path, compile=False)

    # Build concrete TF function (avoids Keras 3 IR / MLIR Cast issues)
    is_fusion = (X_calib_ppg is not None)
    if is_fusion:
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, X_calib.shape[1],  1], dtype=tf.float32),
            tf.TensorSpec(shape=[1, X_calib_ppg.shape[1], 1], dtype=tf.float32),
        ])
        def serving_fn(ecg, ppg):
            return model([ecg, ppg], training=False)[0]  # probs only
    else:
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, X_calib.shape[1], 1], dtype=tf.float32)
        ])
        def serving_fn(x):
            return model(x, training=False)

    concrete = serving_fn.get_concrete_function()

    # ── FP16 version ──────────────────────────────────────────────────────
    try:
        conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
        tflite_fp16 = conv.convert()
        fp16_path = os.path.join(TFLITE_DIR, f"{out_name}_fp16.tflite")
        with open(fp16_path, "wb") as f:
            f.write(tflite_fp16)
        print(f"  FP16 → {fp16_path}  ({len(tflite_fp16)/1024:.1f} KB)")
    except Exception as e:
        print(f"  [WARN] FP16 failed: {e}")
        tflite_fp16 = None

    # ── INT8 version ──────────────────────────────────────────────────────
    try:
        conv2 = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
        conv2.optimizations = [tf.lite.Optimize.DEFAULT]
        conv2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv2.inference_input_type  = tf.int8
        conv2.inference_output_type = tf.int8

        if is_fusion:
            def rep_ds_fusion():
                idx = np.random.choice(len(X_calib), min(200, len(X_calib)), replace=False)
                for i in idx:
                    yield [X_calib[i:i+1], X_calib_ppg[i:i+1]]
            conv2.representative_dataset = rep_ds_fusion
        else:
            def rep_ds():
                yield from representative_dataset_gen(X_calib)
            conv2.representative_dataset = rep_ds

        tflite_int8 = conv2.convert()
        int8_path = os.path.join(TFLITE_DIR, f"{out_name}_int8.tflite")
        with open(int8_path, "wb") as f:
            f.write(tflite_int8)
        print(f"  INT8 → {int8_path}  ({len(tflite_int8)/1024:.1f} KB)")
        generate_arduino_header(tflite_int8, out_name)
    except Exception as e:
        print(f"  [WARN] INT8 failed: {e}")
        if tflite_fp16:
            print("  Falling back to FP16 for Arduino header.")
            generate_arduino_header(tflite_fp16, out_name)



def generate_arduino_header(tflite_bytes: bytes, model_name: str):
    """Convert .tflite binary to C array header for Arduino."""
    var_name = model_name.replace("-", "_").replace(" ", "_")
    lines = [
        f"// Auto-generated TFLite model header: {model_name}",
        f"// Size: {len(tflite_bytes)} bytes",
        f"#pragma once",
        f"#include <stdint.h>",
        f"const unsigned int {var_name}_len = {len(tflite_bytes)};",
        f"alignas(8) const uint8_t {var_name}[] = {{",
    ]
    hex_vals = [f"0x{b:02x}" for b in tflite_bytes]
    for i in range(0, len(hex_vals), 12):
        lines.append("  " + ", ".join(hex_vals[i:i+12]) + ",")
    lines.append("};")

    h_path = os.path.join(TFLITE_DIR, f"{model_name}_model.h")
    with open(h_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Arduino header → {h_path}")


def main():
    print("Loading calibration data …")
    data   = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
    X_ecg  = data["X_ecg"][..., np.newaxis].astype(np.float32)
    X_ppg  = data["X_ppg"][..., np.newaxis].astype(np.float32)

    # Teacher uses BiLSTM which fails TFLite/MLIR conversion in Keras 3
    # — teacher is never deployed to MCU, only used for KD. Skip it.
    conversions = [
        ("student_ce_only.keras",  "student_ce",     X_ecg, None),
        ("student_ce_kl.keras",    "student_ce_kl",  X_ecg, None),
        ("student_full_kd.keras",  "student_kd",     X_ecg, None),
    ]
    # Add fusion only if checkpoint exists
    fusion_path = os.path.join(MODELS_DIR, "fusion_best.keras")
    if os.path.exists(fusion_path):
        conversions.append(("fusion_best.keras", "fusion", X_ecg, X_ppg))
    else:
        print("  [SKIP] fusion_best.keras not yet available — run train_fusion.py first")

    for ckpt, name, X_c, X_p in conversions:
        convert_model(os.path.join(MODELS_DIR, ckpt), name, X_c,
                      X_calib_ppg=X_p)

    print("\n✓ All conversions done. Files in deployment/tflite/")


if __name__ == "__main__":
    main()
