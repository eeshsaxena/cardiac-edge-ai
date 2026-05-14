"""
eval_torch.py  —  Evaluate all PyTorch student checkpoints on the test set.
Generates Table 1 metrics + confusion matrix for the paper.
Run: python training/eval_torch.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    PROCESSED_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR,
    NUM_CLASSES, CLASSES, STUDENT_FILTERS, STUDENT_FC_UNITS
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DWSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel, padding=kernel//2, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


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
        x = self.pool1(self.blk1(x))
        x = self.pool2(self.blk2(x))
        x = self.gap(self.blk3(x)).squeeze(-1)
        return self.fc2(F.relu(self.fc1(x)))


def load_test():
    data = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
    X = torch.from_numpy(data["X_ecg"][:, np.newaxis, :].astype("float32"))
    y = data["y"]
    return X, y


def predict(model, X, batch=512):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logits = model(X[i:i+batch].to(DEVICE))
            all_preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(all_preds)


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Confusion matrix -> {path}")


def main():
    print("=" * 60)
    print("  STUDENT TEST SET EVALUATION")
    print("=" * 60)

    X_test, y_test = load_test()
    print(f"Test samples: {len(y_test):,}  |  Device: {DEVICE}\n")

    variants = [
        ("student_ce_only_torch.pt",  "CE only"),
        ("student_ce_kl_torch.pt",    "CE + KL"),
        ("student_full_kd_torch.pt",  "CE + KL + Spectral (Full KD)"),
    ]

    table_rows = []
    for pt_file, label in variants:
        pt_path = os.path.join(MODELS_DIR, pt_file)
        if not os.path.exists(pt_path):
            print(f"[SKIP] {pt_file} not found")
            table_rows.append([label, "N/A", "N/A", "N/A"])
            continue

        ckpt  = torch.load(pt_path, map_location=DEVICE, weights_only=True)
        model = StudentNet().to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        params = sum(p.numel() for p in model.parameters())
        model_kb = sum(p.numel() * p.element_size() for p in model.parameters()) // 1024

        preds = predict(model, X_test)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average="macro", zero_division=0)
        table_rows.append([label, f"{acc*100:.2f}%", f"{f1:.4f}",
                           f"{params:,}", f"{model_kb} KB"])

        print(f"-- {label} --")
        print(f"   Params: {params:,}  |  Size: {model_kb} KB")
        print(f"   Test Acc: {acc*100:.2f}%  |  Macro-F1: {f1:.4f}")

        if "full_kd" in pt_file:
            print(classification_report(y_test, preds,
                                        target_names=CLASSES, digits=4,
                                        zero_division=0))
            slug = pt_file.replace("_torch.pt", "")
            plot_confusion(
                y_test, preds,
                f"Student Full KD — Test Set Confusion Matrix",
                os.path.join(FIGURES_DIR, f"confusion_{slug}.png")
            )
        print()

    # Print paper table
    print("=" * 70)
    print("  TABLE 1 — ABLATION RESULTS (Test Set)")
    print("=" * 70)
    hdr = f"{'Model':<30} {'Acc':>8} {'Macro-F1':>10} {'Params':>10} {'Size':>8}"
    print(hdr)
    print("-" * 70)
    for row in table_rows:
        if len(row) == 5:
            print(f"{row[0]:<30} {row[1]:>8} {row[2]:>10} {row[3]:>10} {row[4]:>8}")
    print("=" * 70)

    # Save CSV
    import csv
    csv_path = os.path.join(LOGS_DIR, "table1_ablation_test.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Test Acc", "Macro-F1", "Params", "Size KB"])
        w.writerows(table_rows)
    print(f"CSV saved -> {csv_path}")


if __name__ == "__main__":
    main()
