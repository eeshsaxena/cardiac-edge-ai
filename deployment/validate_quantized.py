"""
validate_quantized.py — Compare BN-folded FP32 vs INT8 TFLite accuracy.
Produces Table 3 (quantization accuracy degradation) for the paper.
Run: python deployment/validate_quantized.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, classification_report
from config import (
    PROCESSED_DIR, MODELS_DIR, TFLITE_DIR,
    NUM_CLASSES, CLASSES, STUDENT_FILTERS, STUDENT_FC_UNITS
)


# ── PyTorch reference model ───────────────────────────────────────────────────

class DWSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, k, padding=k//2, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
    def forward(self, x): return F.relu(self.bn(self.pw(self.dw(x))))

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        f = STUDENT_FILTERS
        self.blk1  = DWSBlock(1,    f[0], 7); self.pool1 = nn.MaxPool1d(2)
        self.blk2  = DWSBlock(f[0], f[1], 5); self.pool2 = nn.MaxPool1d(2)
        self.blk3  = DWSBlock(f[1], f[2], 3)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc1   = nn.Linear(f[2], STUDENT_FC_UNITS)
        self.fc2   = nn.Linear(STUDENT_FC_UNITS, NUM_CLASSES)
    def forward(self, x):
        x = self.pool1(self.blk1(x)); x = self.pool2(self.blk2(x))
        x = self.gap(self.blk3(x)).squeeze(-1)
        return self.fc2(F.relu(self.fc1(x)))


def pytorch_eval(pt_path, X_np, y):
    """Evaluate full-precision PyTorch model — gold standard."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(pt_path, map_location=device, weights_only=True)
    model  = StudentNet().to(device); model.load_state_dict(ckpt["model_state"])
    model.eval()
    X_t = torch.from_numpy(X_np).to(device)   # (N, 1, 360)
    with torch.no_grad():
        preds = model(X_t).argmax(1).cpu().numpy()
    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds, average="macro", zero_division=0)
    return acc, f1, preds


def tflite_eval(tflite_path, X_np, y):
    """
    Evaluate TFLite model.
    Our models use float32 I/O (int8 ops internally via dynamic-range quant).
    Input shape expected by model: (1, 360, 1).
    """
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    preds = []
    for i in range(len(X_np)):
        # X_np: (N, 1, 360) → model expects (1, 360, 1)
        x = X_np[i].transpose(1, 0)[np.newaxis, :, :]  # (1, 360, 1)
        interp.set_tensor(inp_d["index"], x)
        interp.invoke()
        out = interp.get_tensor(out_d["index"])         # (1, 5)
        preds.append(int(np.argmax(out)))

    preds = np.array(preds)
    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds, average="macro", zero_division=0)
    return acc, f1, preds


def main():
    print("Loading test data …")
    data  = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
    X_ecg = data["X_ecg"][:, np.newaxis, :].astype(np.float32)  # (N, 1, 360)
    y     = data["y"]
    print(f"Test samples: {len(y):,}  |  Classes: {CLASSES}\n")

    variants = [
        ("student_ce_only_torch.pt",  "student_ce",    "CE Only"),
        ("student_ce_kl_torch.pt",    "student_ce_kl", "CE + KL"),
        ("student_full_kd_torch.pt",  "student_kd",    "Full KD (CE+KL+Spec)"),
    ]

    print("=" * 75)
    print(f"  {'Model':<24} {'Format':<18} {'Acc':>8} {'Macro-F1':>10} {'F1 Drop':>9} {'Size':>8}")
    print("-" * 75)

    table_rows = []
    for pt_file, slug, label in variants:
        pt_path   = os.path.join(MODELS_DIR, pt_file)
        fp32_path = os.path.join(TFLITE_DIR,  f"{slug}_fp32.tflite")
        int8_path = os.path.join(TFLITE_DIR,  f"{slug}_int8.tflite")

        if not os.path.exists(pt_path):
            print(f"  [SKIP] {pt_file}"); continue

        # PyTorch (reference)
        pt_acc, pt_f1, pt_preds = pytorch_eval(pt_path, X_ecg, y)
        pt_size = sum(p.numel()*p.element_size()
                      for p in torch.load(pt_path, map_location="cpu",
                                          weights_only=True)["model_state"].values()) / 1024
        print(f"  {label:<24} {'PyTorch FP32':<18} {pt_acc*100:>7.2f}% {pt_f1:>10.4f} "
              f"{'—':>9} {pt_size:>6.0f} KB")
        table_rows.append([label, "PyTorch FP32", f"{pt_acc*100:.2f}%",
                           f"{pt_f1:.4f}", "—", f"{pt_size:.0f} KB"])

        # BN-folded FP32 TFLite
        if os.path.exists(fp32_path):
            fp32_acc, fp32_f1, _ = tflite_eval(fp32_path, X_ecg, y)
            fp32_size = os.path.getsize(fp32_path) / 1024
            fp32_drop = pt_f1 - fp32_f1
            print(f"  {'':24} {'TFLite FP32':<18} {fp32_acc*100:>7.2f}% {fp32_f1:>10.4f} "
                  f"{fp32_drop:>+9.4f} {fp32_size:>6.1f} KB")
            table_rows.append(["", "TFLite FP32", f"{fp32_acc*100:.2f}%",
                               f"{fp32_f1:.4f}", f"{fp32_drop:+.4f}", f"{fp32_size:.1f} KB"])

        # INT8 TFLite
        if os.path.exists(int8_path):
            int8_acc, int8_f1, int8_preds = tflite_eval(int8_path, X_ecg, y)
            int8_size = os.path.getsize(int8_path) / 1024
            int8_drop = pt_f1 - int8_f1
            print(f"  {'':24} {'TFLite INT8':<18} {int8_acc*100:>7.2f}% {int8_f1:>10.4f} "
                  f"{int8_drop:>+9.4f} {int8_size:>6.1f} KB")
            table_rows.append(["", "TFLite INT8", f"{int8_acc*100:.2f}%",
                               f"{int8_f1:.4f}", f"{int8_drop:+.4f}", f"{int8_size:.1f} KB"])

            # Per-class breakdown for full KD
            if slug == "student_kd":
                print(f"\n  -- Per-class (INT8 Full KD) --")
                print(classification_report(y, int8_preds, target_names=CLASSES,
                                            digits=4, zero_division=0))

        print()

    print("=" * 75)
    print("  ✓ Table 3 — Quantization accuracy drop analysis complete.")

    # Arduino Nano 33 BLE fitness check
    print(f"\n  Arduino Nano 33 BLE (1024 KB Flash, 256 KB RAM):")
    int8_path = os.path.join(TFLITE_DIR, "student_kd_int8.tflite")
    if os.path.exists(int8_path):
        kb = os.path.getsize(int8_path) / 1024
        pct_flash = kb / 1024 * 100
        print(f"    student_kd INT8: {kb:.1f} KB → uses {pct_flash:.1f}% of flash  ✅")
        print(f"    Remaining flash for sketch + runtime: {1024-kb:.0f} KB")

    # Save CSV
    import csv
    csv_path = os.path.join(os.path.dirname(TFLITE_DIR), "..", "experiments",
                            "logs", "table3_quantization.csv")
    csv_path = os.path.normpath(csv_path)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Format", "Acc", "Macro-F1", "F1 Drop", "Size"])
        w.writerows(table_rows)
    print(f"\n  CSV saved → {csv_path}")


if __name__ == "__main__":
    main()
