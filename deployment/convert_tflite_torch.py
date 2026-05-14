"""
convert_tflite_torch.py
────────────────────────
Convert PyTorch student models directly to TFLite using ai_edge_torch.
This bypasses the Keras 3 BatchNorm Cast issue entirely.

Produces for each variant:
  - student_*_fp32.tflite   (float32 — for validation)
  - student_*_int8.tflite   (INT8 quantized — for Arduino)
  - student_*_model.h       (C array header for Arduino sketch)

Run: python deployment/convert_tflite_torch.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    PROCESSED_DIR, MODELS_DIR, TFLITE_DIR,
    NUM_CLASSES, STUDENT_FILTERS, STUDENT_FC_UNITS
)


# ── PyTorch Student Model (same as training) ──────────────────────────────────

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
        return torch.softmax(self.fc2(F.relu(self.fc1(x))), dim=1)


# ── Conversion ────────────────────────────────────────────────────────────────

def generate_arduino_header(tflite_path: str, model_name: str):
    """Convert .tflite file to Arduino C header."""
    with open(tflite_path, "rb") as f:
        data = f.read()
    var = model_name.replace("-", "_")
    lines = [
        f"// Auto-generated TFLite model: {model_name}",
        f"// Size: {len(data)} bytes  ({len(data)/1024:.1f} KB)",
        f"#pragma once",
        f"#include <stdint.h>",
        f"const unsigned int {var}_len = {len(data)};",
        f"alignas(8) const uint8_t {var}[] = {{",
    ]
    hex_vals = [f"0x{b:02x}" for b in data]
    for i in range(0, len(hex_vals), 12):
        lines.append("  " + ", ".join(hex_vals[i:i+12]) + ",")
    lines.append("};")
    h_path = tflite_path.replace(".tflite", "_model.h")
    with open(h_path, "w") as f:
        f.write("\n".join(lines))
    print(f"    Arduino header → {os.path.basename(h_path)}")


def convert_variant(pt_path: str, out_slug: str, X_calib: np.ndarray):
    import ai_edge_torch

    print(f"\nConverting: {os.path.basename(pt_path)}")

    ckpt  = torch.load(pt_path, map_location="cpu", weights_only=True)
    model = StudentNet()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    size_kb = sum(p.numel() * p.element_size() for p in model.parameters()) // 1024
    print(f"  Params: {params:,}  |  FP32 size: {size_kb} KB")

    # Sample input: (1, 1, 360) — batch=1, channels=1, length=360
    sample = (torch.from_numpy(X_calib[:1]).float(),)  # (1, 1, 360)

    # ── FP32 TFLite ────────────────────────────────────────────────────────
    try:
        edge_fp32 = ai_edge_torch.convert(model, sample)
        fp32_path = os.path.join(TFLITE_DIR, f"{out_slug}_fp32.tflite")
        edge_fp32.export(fp32_path)
        size = os.path.getsize(fp32_path)
        print(f"  FP32 → {os.path.basename(fp32_path)}  ({size/1024:.1f} KB)")
    except Exception as e:
        print(f"  [WARN] FP32 failed: {e}")
        fp32_path = None

    # ── INT8 TFLite (post-training quantization) ───────────────────────────
    try:
        # Build calibration dataset
        X_torch = torch.from_numpy(X_calib).float()  # (N, 1, 360)
        idx     = np.random.choice(len(X_calib), min(200, len(X_calib)), replace=False)
        calib_samples = [(X_torch[i:i+1],) for i in idx]

        edge_int8 = ai_edge_torch.convert(
            model, sample,
            quant_config=ai_edge_torch.quantize.pt2e_quantize.PT2EQuantConfig(
                global_config=ai_edge_torch.quantize.quant_config.QuantConfig(
                    fake_quant=ai_edge_torch.quantize.fq_name_lib.Int8ActPerTensorMinMaxConfig(),
                )
            )
        )
        int8_path = os.path.join(TFLITE_DIR, f"{out_slug}_int8.tflite")
        edge_int8.export(int8_path)
        size = os.path.getsize(int8_path)
        print(f"  INT8 → {os.path.basename(int8_path)}  ({size/1024:.1f} KB)")
        generate_arduino_header(int8_path, out_slug)

    except Exception as e:
        print(f"  [WARN] INT8 quantization failed: {e}")
        print("        Using FP32 for Arduino header.")
        if fp32_path and os.path.exists(fp32_path):
            generate_arduino_header(fp32_path, out_slug + "_fp32")


def main():
    print("=" * 60)
    print("  PyTorch → TFLite Conversion (via ai_edge_torch)")
    print("=" * 60)

    # Load calibration ECG data in PyTorch format (N, 1, 360)
    data  = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
    X_ecg = data["X_ecg"][:, np.newaxis, :].astype(np.float32)  # (N, 1, 360)
    print(f"Calibration samples: {len(X_ecg):,}\n")

    variants = [
        ("student_ce_only_torch.pt",  "student_ce"),
        ("student_ce_kl_torch.pt",    "student_ce_kl"),
        ("student_full_kd_torch.pt",  "student_kd"),
    ]

    for pt_file, slug in variants:
        pt_path = os.path.join(MODELS_DIR, pt_file)
        if not os.path.exists(pt_path):
            print(f"[SKIP] {pt_file} not found")
            continue
        convert_variant(pt_path, slug, X_ecg)

    # Summary
    print(f"\n{'='*60}")
    print("  TFLite files:")
    for f in sorted(os.listdir(TFLITE_DIR)):
        if f.endswith(".tflite"):
            sz = os.path.getsize(os.path.join(TFLITE_DIR, f)) / 1024
            print(f"  {f:<40} {sz:6.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
