"""
train_student_torch.py
──────────────────────
PyTorch GPU training for the student KD pipeline.
Uses pre-computed teacher outputs (no TF in the GPU loop).

Trains 3 ablation variants:
  1. CE only          → student_ce_only_torch.pt  + .keras
  2. CE + KL          → student_ce_kl_torch.pt    + .keras
  3. CE + KL + Spec   → student_full_kd_torch.pt  + .keras

After training, weights are transferred back to the Keras student
model and saved as .keras for TFLite conversion.

Run: python training/train_student_torch.py
"""
import os, sys, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pywt

from config import (
    PROCESSED_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR,
    BATCH_SIZE, STUDENT_EPOCHS, LEARNING_RATE,
    LAMBDA_CE, LAMBDA_KL, LAMBDA_SPECTRAL, KD_TEMPERATURE,
    CLASS_WEIGHTS, CLASSES, NUM_CLASSES, WINDOW_LEN,
    STUDENT_FILTERS, STUDENT_FC_UNITS, RANDOM_SEED,
    WAVELET, WAVELET_LEVEL, SPECTRAL_BAND_HZ, ECG_FS,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_BATCH = 256 if torch.cuda.is_available() else BATCH_SIZE  # larger batch for GPU
USE_AMP   = torch.cuda.is_available()  # mixed precision on GPU
print(f"\n{'='*60}")
print(f"  DEVICE: {DEVICE}  |  Batch: {GPU_BATCH}  |  AMP: {USE_AMP}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'='*60}\n")

CACHE_DIR = os.path.join(LOGS_DIR, "teacher_cache")


# ── PyTorch Student Model ─────────────────────────────────────────────────────

class DWSBlock(nn.Module):
    """Depthwise-Separable Conv1D + BN + ReLU."""
    def __init__(self, in_ch, out_ch, kernel, name=""):
        super().__init__()
        # Depthwise (groups=in_ch)
        self.dw = nn.Conv1d(in_ch, in_ch, kernel, padding=kernel//2, groups=in_ch, bias=False)
        # Pointwise
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


class StudentNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, filters=STUDENT_FILTERS, fc=STUDENT_FC_UNITS):
        super().__init__()
        self.blk1  = DWSBlock(1,         filters[0], 7)
        self.pool1 = nn.MaxPool1d(2)
        self.blk2  = DWSBlock(filters[0], filters[1], 5)
        self.pool2 = nn.MaxPool1d(2)
        self.blk3  = DWSBlock(filters[1], filters[2], 3)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc1   = nn.Linear(filters[2], fc)
        self.fc2   = nn.Linear(fc, num_classes)

    def forward(self, x, return_features=False):
        x = self.pool1(self.blk1(x))
        x = self.pool2(self.blk2(x))
        feat = self.blk3(x)                  # (B, C, T) — for L_spectral
        x = self.gap(feat).squeeze(-1)       # (B, C)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        if return_features:
            return logits, feat
        return logits


# ── Loss Functions ────────────────────────────────────────────────────────────

def weighted_ce(logits, labels, weights_dict=CLASS_WEIGHTS):
    w = torch.tensor([weights_dict[i] for i in range(NUM_CLASSES)],
                     dtype=torch.float32, device=DEVICE)
    return F.cross_entropy(logits, labels, weight=w)


def kl_loss(student_logits, teacher_logits, T=KD_TEMPERATURE):
    p_t = F.softmax(teacher_logits / T, dim=1)
    p_s = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(p_s, p_t, reduction="batchmean") * (T ** 2)


def _compute_band_mask():
    f_low, f_high = SPECTRAL_BAND_HZ
    mask = []
    for lvl in range(1, WAVELET_LEVEL + 1):
        hi = ECG_FS / (2 ** lvl)
        lo = ECG_FS / (2 ** (lvl + 1))
        mask.append(lo <= f_high and hi >= f_low)
    mask.append(True)  # approx always in band
    return mask


BAND_MASK = _compute_band_mask()


def spectral_loss_np(t_feat_np, s_feat_np):
    """Vectorised batch wavelet spectral loss (B, C, T) numpy arrays."""
    # GAP over channels -> (B, T)
    t_sig = t_feat_np.mean(axis=1).astype(np.float64)
    s_sig = s_feat_np.mean(axis=1).astype(np.float64)
    min_len = min(t_sig.shape[1], s_sig.shape[1])
    t_sig = t_sig[:, :min_len]
    s_sig = s_sig[:, :min_len]

    total, count = 0.0, 0
    # Process all samples at once (vectorised per level)
    t_coeffs = pywt.wavedec(t_sig, WAVELET, level=WAVELET_LEVEL, axis=1)
    s_coeffs = pywt.wavedec(s_sig, WAVELET, level=WAVELET_LEVEL, axis=1)
    # t_coeffs[0]=approx, [1..]=details high->low
    for k, keep in enumerate(BAND_MASK[:-1]):
        if keep:
            tc = t_coeffs[WAVELET_LEVEL - k]  # (B, T_k)
            sc = s_coeffs[WAVELET_LEVEL - k]
            n = min(tc.shape[1], sc.shape[1])
            diff = tc[:, :n] - sc[:, :n]
            total += float(np.sum(diff ** 2))
            count += diff.size
    if BAND_MASK[-1]:
        tc, sc = t_coeffs[0], s_coeffs[0]
        n = min(tc.shape[1], sc.shape[1])
        diff = tc[:, :n] - sc[:, :n]
        total += float(np.sum(diff ** 2))
        count += diff.size
    return total / max(count, 1)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(split):
    """Returns (X_ecg, y, t_logits, t_features) as tensors on CPU."""
    # ECG data
    ecg = np.load(os.path.join(PROCESSED_DIR, f"{split}.npz"))
    X = ecg["X_ecg"].astype(np.float32)           # (N, 360)
    y = ecg["y"].astype(np.int64)

    # Teacher cache
    cache = np.load(os.path.join(CACHE_DIR, f"teacher_{split}.npz"))
    t_logits = cache["logits"].astype(np.float32)     # (N, 5)
    t_feats  = cache["features"].astype(np.float32)   # (N, T, C) — TF format

    # Convert TF feature shape (N, T, C) → PyTorch (N, C, T)
    t_feats = t_feats.transpose(0, 2, 1)

    # Reshape X: (N, 360) → (N, 1, 360) for Conv1d
    X = X[:, np.newaxis, :]

    return (
        torch.from_numpy(X),
        torch.from_numpy(y),
        torch.from_numpy(t_logits),
        torch.from_numpy(t_feats),
    )


def make_loader(X, y, t_logits, t_feats, shuffle=True):
    ds = TensorDataset(X, y, t_logits, t_feats)
    return DataLoader(ds, batch_size=GPU_BATCH, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))


# ── Training ──────────────────────────────────────────────────────────────────

def train_variant(name, use_kl, use_spectral, loaders):
    train_loader, val_loader = loaders

    print(f"\n{'='*60}")
    print(f"  VARIANT: {name}  |  KL={use_kl}  Spectral={use_spectral}")
    print(f"{'='*60}")

    model = StudentNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    best_f1, best_state, patience_cnt = 0.0, None, 0
    log_rows = []
    PATIENCE = 10

    for epoch in range(1, STUDENT_EPOCHS + 1):
        t0 = time.time()
        model.train()
        epoch_loss, steps = 0.0, 0

        for X_b, y_b, tl_b, tf_b in train_loader:
            X_b  = X_b.to(DEVICE)
            y_b  = y_b.to(DEVICE)
            tl_b = tl_b.to(DEVICE)
            tf_b = tf_b.to(DEVICE)

            optimizer.zero_grad()

            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                if use_spectral:
                    s_logits, s_feat = model(X_b, return_features=True)
                else:
                    s_logits = model(X_b)

                # CE loss
                loss = LAMBDA_CE * weighted_ce(s_logits, y_b)

                # KL loss
                if use_kl:
                    loss = loss + LAMBDA_KL * kl_loss(s_logits, tl_b)

            # Spectral loss (numpy CPU, outside autocast)
            if use_spectral:
                t_np = tf_b.cpu().numpy()
                s_np = s_feat.detach().cpu().numpy()
                l_spec = spectral_loss_np(t_np, s_np)
                loss = loss + LAMBDA_SPECTRAL * torch.tensor(l_spec, dtype=loss.dtype, device=DEVICE)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        # Validation
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_b, y_b, _, _ in val_loader:
                logits = model(X_b.to(DEVICE))
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_true.append(y_b.numpy())

        preds = np.concatenate(all_preds)
        trues = np.concatenate(all_true)
        val_acc = (preds == trues).mean()

        from sklearn.metrics import f1_score
        val_f1 = f1_score(trues, preds, average="macro", zero_division=0)
        elapsed = time.time() - t0

        print(f"Ep {epoch:3d}/{STUDENT_EPOCHS} | "
              f"loss={epoch_loss/steps:.4f} | "
              f"val_acc={val_acc:.4f} | val_F1={val_f1:.4f} | "
              f"{elapsed:.1f}s", flush=True)

        log_rows.append({"epoch": epoch, "train_loss": epoch_loss/steps,
                         "val_acc": val_acc, "val_macro_f1": val_f1})

        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    # Restore best
    model.load_state_dict(best_state)
    model.to("cpu")

    # Save PyTorch checkpoint
    slug = {"CE only": "student_ce_only", "CE + KL": "student_ce_kl",
            "CE + KL + Spec": "student_full_kd"}.get(name, name.replace(" ", "_").lower())
    pt_path = os.path.join(MODELS_DIR, f"{slug}_torch.pt")
    torch.save({"model_state": model.state_dict(),
                "best_val_f1": best_f1,
                "config": {"filters": STUDENT_FILTERS, "fc": STUDENT_FC_UNITS}},
               pt_path)
    print(f"\nSaved PyTorch -> {pt_path}  (val F1={best_f1:.4f})")

    # Save CSV log
    csv_path = os.path.join(LOGS_DIR, f"{slug}_history.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader()
        w.writerows(log_rows)

    # Transfer weights to Keras model and save .keras
    save_as_keras(model, slug)

    return model, best_f1


def save_as_keras(torch_model, slug):
    """Transfer PyTorch weights to equivalent Keras model and save."""
    import tensorflow as tf
    from models.student import build_student_with_intermediate
    print(f"  Transferring weights to Keras ...", flush=True)

    keras_model, _ = build_student_with_intermediate()

    # Get weight arrays from PyTorch
    sd = torch_model.state_dict()

    # Map: PyTorch layer → Keras layer (by structural order)
    # Both have the same architecture, so we map by index
    torch_weights = []
    for key in sd:
        arr = sd[key].numpy()
        # Conv1d in PyTorch: (out, in, k) → Keras SeparableConv1D (k, in, out)
        if arr.ndim == 3:
            arr = arr.transpose(2, 1, 0)
        # BatchNorm: gamma, beta, mean, var → same shape
        torch_weights.append(arr)

    # Set weights structurally
    idx = 0
    for layer in keras_model.layers:
        w = layer.get_weights()
        if not w:
            continue
        new_w = []
        for wi in w:
            if idx < len(torch_weights):
                tw = torch_weights[idx]
                if tw.shape == wi.shape:
                    new_w.append(tw)
                    idx += 1
                else:
                    new_w.append(wi)  # keep Keras init if shape mismatch
            else:
                new_w.append(wi)
        try:
            layer.set_weights(new_w)
        except Exception:
            pass  # skip layers where shapes don't match

    keras_path = os.path.join(MODELS_DIR, f"{slug}.keras")
    keras_model.save(keras_path)
    print(f"  Keras model saved -> {keras_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...", flush=True)
    Xtr, ytr, tl_tr, tf_tr = load_data("train")
    Xvl, yvl, tl_vl, tf_vl = load_data("val")
    print(f"  Train: {len(Xtr):,}  Val: {len(Xvl):,}", flush=True)

    train_loader = make_loader(Xtr, ytr, tl_tr, tf_tr, shuffle=True)
    val_loader   = make_loader(Xvl, yvl, tl_vl, tf_vl, shuffle=False)
    loaders = (train_loader, val_loader)

    results = {}
    for name, (use_kl, use_spec) in [
        ("CE only",        (False, False)),
        ("CE + KL",        (True,  False)),
        ("CE + KL + Spec", (True,  True)),
    ]:
        _, f1 = train_variant(name, use_kl, use_spec, loaders)
        results[name] = f1

    print(f"\n{'='*60}")
    print("  ABLATION SUMMARY (Val Macro-F1)")
    print(f"{'='*60}")
    for n, f in results.items():
        print(f"  {n:20s}: {f:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
