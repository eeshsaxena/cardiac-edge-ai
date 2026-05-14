"""
train_fusion.py — ECG + PPG late fusion training (two-phase).
Run: python training/train_fusion.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from config import (
    PROCESSED_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR,
    BATCH_SIZE, FUSION_EPOCHS, LEARNING_RATE,
    CLASS_WEIGHTS, CLASSES, RANDOM_SEED
)
from models.fusion import build_fusion_model
from training.losses import WeightedCrossEntropy

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_split(split):
    data = np.load(os.path.join(PROCESSED_DIR, f"{split}.npz"))
    return (data["X_ecg"][..., np.newaxis].astype(np.float32),
            data["X_ppg"][..., np.newaxis].astype(np.float32),
            data["y"].astype(np.int32))


def make_ds(X_ecg, X_ppg, y, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(
        ({"fusion_ecg_input": X_ecg, "fusion_ppg_input": X_ppg}, y)
    )
    if shuffle:
        ds = ds.shuffle(len(y), seed=RANDOM_SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def evaluate(model, ds):
    probs_list, y_list, alpha_list = [], [], []
    for inputs, y_b in ds:
        p, a = model(inputs, training=False)
        probs_list.append(p.numpy())
        y_list.append(y_b.numpy())
        alpha_list.append(a.numpy())
    probs = np.concatenate(probs_list)
    ys    = np.concatenate(y_list)
    alpha = np.concatenate(alpha_list)
    preds = np.argmax(probs, 1)
    acc   = np.mean(preds == ys)
    f1    = f1_score(ys, preds, average="macro", zero_division=0)
    return acc, f1, alpha, ys


def plot_alpha(alpha, y_true, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    a_flat = alpha.flatten()
    for i, cls in enumerate(CLASSES):
        mask = y_true == i
        if mask.sum():
            ax.scatter([cls]*mask.sum(), a_flat[mask], alpha=0.3, s=8)
    ax.axhline(0.5, color="red", ls="--", lw=1, label="α=0.5 (equal weight)")
    ax.set(xlabel="Class", ylabel="α (higher = more ECG weight)",
           title="Learned fusion α per class", ylim=(0, 1))
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Alpha plot → {path}")


def train():
    print("="*60 + "\n  FUSION TRAINING\n" + "="*60)

    X_tr_e, X_tr_p, y_tr = load_split("train")
    X_vl_e, X_vl_p, y_vl = load_split("val")
    train_ds = make_ds(X_tr_e, X_tr_p, y_tr)
    val_ds   = make_ds(X_vl_e, X_vl_p, y_vl, shuffle=False)

    model, ecg_b, _ = build_fusion_model()

    # Load pretrained student ECG weights if available
    for ckpt in ["student_full_kd.keras", "student_ce_kl.keras", "student_ce_only.keras"]:
        p = os.path.join(MODELS_DIR, ckpt)
        if os.path.exists(p):
            pre = tf.keras.models.load_model(p)
            n = 0
            for layer in model.layers:
                try:
                    w = pre.get_layer(layer.name).get_weights()
                    if w:
                        layer.set_weights(w)
                        n += 1
                except (ValueError, AttributeError):
                    pass
            print(f"Loaded {n} ECG layers from {ckpt}")
            break

    ce_loss  = WeightedCrossEntropy(CLASS_WEIGHTS)
    opt      = tf.keras.optimizers.Adam(LEARNING_RATE)
    best_f1, best_w, patience = 0.0, None, 0

    for epoch in range(1, FUSION_EPOCHS + 1):
        # Phase 1 freeze
        if epoch == 1:
            for l in model.layers:
                if "student" in l.name.lower():
                    l.trainable = False
            print("Phase 1 (ECG frozen) …")
        if epoch == 11:
            for l in model.layers:
                l.trainable = True
            opt.learning_rate.assign(LEARNING_RATE * 0.3)
            print("Phase 2 (all unfrozen, LR × 0.3) …")

        t0, total_loss, steps = time.time(), 0.0, 0
        for inputs, y_b in train_ds:
            with tf.GradientTape() as tape:
                p, _ = model(inputs, training=True)
                loss  = ce_loss(y_b, p)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += loss.numpy(); steps += 1

        acc, f1, alpha, _ = evaluate(model, val_ds)
        print(f"Ep {epoch:3d}/{FUSION_EPOCHS} | loss={total_loss/steps:.4f} | "
              f"acc={acc:.4f} | F1={f1:.4f} | mean_α={alpha.mean():.3f} | "
              f"{time.time()-t0:.1f}s")

        if f1 > best_f1:
            best_f1, best_w, patience = f1, model.get_weights(), 0
        else:
            patience += 1
            if patience == 5:
                opt.learning_rate.assign(float(opt.learning_rate) * 0.5)
            if patience >= 10:
                print(f"Early stop at epoch {epoch}"); break

    model.set_weights(best_w)
    save_path = os.path.join(MODELS_DIR, "fusion_best.keras")
    model.save(save_path)
    print(f"\nSaved → {save_path}  |  Best Val F1 = {best_f1:.4f}")

    # Alpha analysis plot
    _, _, alpha_val, y_val_true = evaluate(model, val_ds)
    plot_alpha(alpha_val, y_val_true, os.path.join(FIGURES_DIR, "alpha_per_class.png"))


if __name__ == "__main__":
    train()
