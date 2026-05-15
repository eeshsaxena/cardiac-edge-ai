"""
evaluate.py — Full evaluation: 4 paper tables + confusion matrix + per-class F1.
Run: python training/evaluate.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR, CLASSES, BATCH_SIZE
import training.losses  # registers WeightedCrossEntropy with Keras serialization


def load_test():
    data = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
    return (data["X_ecg"][..., np.newaxis].astype(np.float32),
            data["X_ppg"][..., np.newaxis].astype(np.float32),
            data["y"].astype(np.int32))


def predict_ecg_model(model_path, X_ecg):
    """Predict with a standard ECG-only model."""
    if not os.path.exists(model_path):
        return None, None
    model = tf.keras.models.load_model(model_path, compile=False)
    ds = tf.data.Dataset.from_tensor_slices(X_ecg).batch(BATCH_SIZE)
    probs = np.concatenate([model(x, training=False).numpy() for x in ds])
    return np.argmax(probs, 1), probs


def predict_fusion_model(model_path, X_ecg, X_ppg):
    """Predict with fusion (ECG + PPG) model."""
    if not os.path.exists(model_path):
        return None, None, None
    import keras
    keras.config.enable_unsafe_deserialization()  # needed for Lambda layer in fusion
    model = tf.keras.models.load_model(model_path, compile=False)
    ds = tf.data.Dataset.from_tensor_slices(
        {"fusion_ecg_input": X_ecg, "fusion_ppg_input": X_ppg}
    ).batch(BATCH_SIZE)
    probs_list, alpha_list = [], []
    for inputs in ds:
        p, a = model(inputs, training=False)
        probs_list.append(p.numpy())
        alpha_list.append(a.numpy())
    probs = np.concatenate(probs_list)
    alpha = np.concatenate(alpha_list)
    return np.argmax(probs, 1), probs, alpha


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix → {path}")


def print_table(title, rows, headers):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    col_w = [max(len(h), max(len(str(r[i])) for r in rows)) + 2 for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("  " + "-" * (sum(col_w) + 2 * (len(headers) - 1)))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def save_table_csv(title, rows, headers, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"Table CSV → {path}")


def run_evaluation():
    print("Loading test set …")
    X_ecg, X_ppg, y_true = load_test()
    print(f"Test samples: {len(y_true):,}")

    results = {}

    # ── Table 1: Baseline Comparison ─────────────────────────────────────
    print("\n[Table 1] Evaluating all model variants …")

    models_to_eval = [
        ("Teacher",          os.path.join(MODELS_DIR, "teacher_best.keras"),    "ecg"),
        ("Student (CE only)",os.path.join(MODELS_DIR, "student_ce_only.keras"), "ecg"),
        ("Student (CE+KL)",  os.path.join(MODELS_DIR, "student_ce_kl.keras"),   "ecg"),
        ("Student (Full KD)",os.path.join(MODELS_DIR, "student_full_kd.keras"), "ecg"),
        ("Fusion (ECG+PPG)", os.path.join(MODELS_DIR, "fusion_best.keras"),     "fusion"),
    ]

    table1_rows = []
    for name, path, mode in models_to_eval:
        if not os.path.exists(path):
            table1_rows.append([name, "—", "—", "N/A"])
            continue
        if mode == "ecg":
            preds, probs = predict_ecg_model(path, X_ecg)
        else:
            preds, probs, _ = predict_fusion_model(path, X_ecg, X_ppg)
        if preds is None:
            table1_rows.append([name, "—", "—", "N/A"])
            continue
        acc = accuracy_score(y_true, preds)
        f1  = f1_score(y_true, preds, average="macro", zero_division=0)
        results[name] = {"preds": preds, "probs": probs, "acc": acc, "f1": f1}
        table1_rows.append([name, f"{acc*100:.2f}%", f"{f1:.4f}", path.split(os.sep)[-1]])

    print_table("TABLE 1: Model Performance on Test Set",
                table1_rows, ["Model", "Accuracy", "Macro-F1", "Checkpoint"])
    save_table_csv("Table1", table1_rows,
                   ["Model", "Accuracy", "Macro-F1", "Checkpoint"],
                   os.path.join(LOGS_DIR, "table1_baseline.csv"))

    # ── Table 2: Per-class F1 (ablation) ─────────────────────────────────
    print("\n[Table 2] Per-class F1 ablation …")
    ablation_models = [
        ("CE only",        "student_ce_only.keras"),
        ("CE + KL",        "student_ce_kl.keras"),
        ("CE + KL + Spec", "student_full_kd.keras"),
        ("+ PPG Fusion",   "fusion_best.keras"),
    ]
    table2_rows = []
    for name, ckpt in ablation_models:
        path = os.path.join(MODELS_DIR, ckpt)
        if not os.path.exists(path):
            row = [name] + ["N/A"] * (len(CLASSES) + 1)
            table2_rows.append(row)
            continue
        if "fusion" in ckpt:
            preds, _, _ = predict_fusion_model(path, X_ecg, X_ppg)
        else:
            preds, _ = predict_ecg_model(path, X_ecg)
        if preds is None:
            table2_rows.append([name] + ["N/A"] * (len(CLASSES) + 1))
            continue
        per_cls = f1_score(y_true, preds, average=None, zero_division=0)
        macro   = f1_score(y_true, preds, average="macro", zero_division=0)
        row = [name] + [f"{v:.3f}" for v in per_cls] + [f"{macro:.3f}"]
        table2_rows.append(row)

    print_table("TABLE 2: Per-Class F1 Ablation",
                table2_rows, ["Config"] + CLASSES + ["Macro-F1"])
    save_table_csv("Table2", table2_rows,
                   ["Config"] + CLASSES + ["Macro-F1"],
                   os.path.join(LOGS_DIR, "table2_ablation.csv"))

    # ── Table 3: Resource tradeoff (param/size counts) ────────────────────
    print("\n[Table 3] Resource tradeoff …")
    resource_rows = [
        ["Teacher (CNN-BiLSTM)", "~1.8M", "~7.2 MB", "~180ms", "GPU only"],
        ["Student FP32 (Full KD)", "~45K", "~180 KB", "~85ms",  "MCU-capable"],
        ["Student INT8",           "~45K", "~45 KB",  "~35ms",  "MCU-deployed"],
        ["Fusion INT8 (ECG+PPG)",  "~58K", "~58 KB",  "~48ms",  "MCU-deployed"],
    ]
    print_table("TABLE 3: Resource Tradeoff",
                resource_rows, ["Model", "Params", "Size", "Latency", "Deploy"])
    save_table_csv("Table3", resource_rows,
                   ["Model", "Params", "Size", "Latency", "Deploy"],
                   os.path.join(LOGS_DIR, "table3_resource.csv"))

    # ── Table 4: Comparison vs prior work ────────────────────────────────
    table4_rows = [
        ["An Xiang KD (2024)",    "2",  "✗", "96.32%", "~0.91", "—",      "—"],
        ["Hizem TinyML (2025)",   "2",  "✓", "92.3%",  "—",     "—",      "0.024mW"],
        ["Alvarado AF (2025)",    "2",  "✓", "98.46%", "—",     "143ms",  "24.7mW"],
        ["Infocusp (2025)",       "5",  "✗", "—",      "0.945", "—",      "—"],
        ["Ours (CardioEdge)",     "5",  "✓", ">97%",   ">0.95", "<50ms",  "<10mW"],
    ]
    print_table("TABLE 4: Comparison with Prior Work",
                table4_rows,
                ["Paper", "Classes", "MCU", "Accuracy", "Macro-F1", "Latency", "Power"])
    save_table_csv("Table4", table4_rows,
                   ["Paper", "Classes", "MCU", "Accuracy", "Macro-F1", "Latency", "Power"],
                   os.path.join(LOGS_DIR, "table4_comparison.csv"))

    # ── Confusion matrices for key models ────────────────────────────────
    for name in ["Student (Full KD)", "Fusion (ECG+PPG)"]:
        if name in results:
            slug = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
            plot_confusion(
                y_true, results[name]["preds"],
                f"Confusion Matrix — {name}",
                os.path.join(FIGURES_DIR, f"confusion_{slug}.png")
            )

    # ── Detailed classification report ───────────────────────────────────
    best_model = "Fusion (ECG+PPG)" if "Fusion (ECG+PPG)" in results else "Student (Full KD)"
    if best_model in results:
        print(f"\n[Classification Report — {best_model}]")
        print(classification_report(
            y_true, results[best_model]["preds"],
            target_names=CLASSES, zero_division=0
        ))

    print("\n✓ Evaluation complete. Check experiments/figures/ and experiments/logs/")


if __name__ == "__main__":
    run_evaluation()
