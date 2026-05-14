"""
balance_classes.py
──────────────────
Applies SMOTE oversampling on minority classes (VT, LBBB),
performs stratified train/val/test split, and saves the final
ready-to-train .npz files.

Also generates synthetic PPG signals from ECG for the software phase
(before real MAX30102 hardware is available).

Run:  python data/balance_classes.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from config import (
    PROCESSED_DIR, WINDOW_LEN, CLASSES, NUM_CLASSES,
    VAL_SPLIT, TEST_SPLIT, RANDOM_SEED
)


# ── Synthetic PPG generation ────────────────────────────────────────────────

def ecg_to_synthetic_ppg(ecg_segment: np.ndarray, cls: str) -> np.ndarray:
    """
    Generate a plausible synthetic PPG waveform from an ECG beat segment.
    
    Method:
    1. Detect approximate R-peak location from ECG
    2. Generate a Gaussian-shaped pulse at the corresponding PPG peak
       (PPG peak ≈ 200ms after R-peak, typical PTT)
    3. Add class-specific morphological variation:
       - Normal: clean dicrotic notch, regular amplitude
       - AF:     variable inter-beat interval, reduced amplitude
       - VT:     rapid beats, reduced perfusion amplitude
       - PVC:    reduced amplitude on ectopic beat, compensatory pause
       - LBBB:   slight delay, broadened pulse

    Returns synthetic PPG of same length as ECG segment.
    
    NOTE: This is for software-phase training only. Replace with
    real MAX30102 data once hardware is available.
    """
    t = np.arange(WINDOW_LEN, dtype=np.float32)
    ppg = np.zeros(WINDOW_LEN, dtype=np.float32)

    # Approximate R-peak: maximum of ECG
    r_idx = int(np.argmax(np.abs(ecg_segment)))

    # PPT (pulse transit time) ≈ 200–300ms → 72–108 samples at 360 Hz
    ptt_samples = 90   # ~250ms

    # Class-specific amplitude and width modifiers
    class_params = {
        "N":    {"amp": 1.0, "width": 45, "notch": 0.15, "noise": 0.02},
        "AF":   {"amp": 0.75, "width": 50, "notch": 0.08, "noise": 0.06},
        "VT":   {"amp": 0.55, "width": 30, "notch": 0.05, "noise": 0.08},
        "PVC":  {"amp": 0.60, "width": 40, "notch": 0.06, "noise": 0.05},
        "LBBB": {"amp": 0.85, "width": 55, "notch": 0.12, "noise": 0.03},
    }
    params = class_params.get(cls, class_params["N"])

    peak_idx = r_idx + ptt_samples
    if peak_idx >= WINDOW_LEN:
        peak_idx = WINDOW_LEN // 2

    # Main systolic peak (Gaussian)
    ppg += params["amp"] * np.exp(-((t - peak_idx) ** 2) / (2 * params["width"] ** 2))

    # Dicrotic notch (smaller secondary peak)
    notch_idx = peak_idx + int(0.4 * params["width"])
    if notch_idx < WINDOW_LEN:
        ppg += params["notch"] * np.exp(-((t - notch_idx) ** 2) / (2 * (params["width"] * 0.4) ** 2))

    # Baseline drift (slow sine, class-specific frequency)
    drift_freq = 0.3 if cls == "AF" else 0.1
    ppg += 0.05 * np.sin(2 * np.pi * drift_freq * t / WINDOW_LEN)

    # Add noise
    ppg += params["noise"] * np.random.randn(WINDOW_LEN).astype(np.float32)

    # Z-score normalise
    mu, sigma = ppg.mean(), ppg.std()
    if sigma > 1e-6:
        ppg = (ppg - mu) / sigma

    return ppg


# ── Main pipeline ────────────────────────────────────────────────────────────

def load_combined_data():
    path = os.path.join(PROCESSED_DIR, "combined_beats.npz")
    if not os.path.exists(path):
        # Fallback to MIT-BIH only
        path = os.path.join(PROCESSED_DIR, "mitbih_beats.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No beat data found. Run download_mitbih.py first."
        )
    data = np.load(path)
    return data["X"], data["y"]


def apply_smote(X: np.ndarray, y: np.ndarray):
    """
    SMOTE on flattened segments. Targets minority classes to
    at least 3000 samples each.
    """
    from collections import Counter
    counts = Counter(y.tolist())
    print(f"\nBefore SMOTE: {dict(counts)}")

    # Set desired counts per class
    strategy = {}
    for cls_idx in range(NUM_CLASSES):
        current = counts.get(cls_idx, 0)
        target  = max(current, 3000)
        strategy[cls_idx] = target

    sm = SMOTE(
        sampling_strategy=strategy,
        k_neighbors=min(5, min(counts.values()) - 1),
        random_state=RANDOM_SEED,
    )
    X_flat = X.reshape(len(X), -1)
    X_res, y_res = sm.fit_resample(X_flat, y)
    X_res = X_res.reshape(-1, WINDOW_LEN).astype(np.float32)

    counts_after = Counter(y_res.tolist())
    print(f"After  SMOTE: {dict(counts_after)}")
    return X_res, y_res


def augment_segment(seg: np.ndarray) -> np.ndarray:
    """Lightweight on-the-fly augmentation for variety."""
    aug = seg.copy()
    # Time warp ±2%
    warp = np.random.uniform(0.98, 1.02)
    from scipy.signal import resample
    warped = resample(aug, int(WINDOW_LEN * warp))
    if len(warped) >= WINDOW_LEN:
        aug = warped[:WINDOW_LEN]
    else:
        aug = np.pad(warped, (0, WINDOW_LEN - len(warped)), mode="edge")
    # Gaussian noise
    aug += 0.01 * np.random.randn(WINDOW_LEN).astype(np.float32)
    return aug


def generate_ppg_dataset(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Generate synthetic PPG for every ECG segment."""
    print("\nGenerating synthetic PPG signals …")
    X_ppg = np.zeros_like(X)
    for i in tqdm(range(len(X))):
        cls_name = CLASSES[y[i]]
        X_ppg[i] = ecg_to_synthetic_ppg(X[i], cls_name)
    return X_ppg


def split_and_save(X_ecg, X_ppg, y):
    """Stratified 70/15/15 split and save train/val/test .npz files."""
    # First split off test set
    idx = np.arange(len(y))
    idx_tv, idx_test = train_test_split(
        idx, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED
    )
    # Then split train/val
    val_ratio = VAL_SPLIT / (1.0 - TEST_SPLIT)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=val_ratio, stratify=y[idx_tv], random_state=RANDOM_SEED
    )

    splits = {
        "train": idx_train,
        "val":   idx_val,
        "test":  idx_test,
    }

    for split_name, indices in splits.items():
        out_path = os.path.join(PROCESSED_DIR, f"{split_name}.npz")
        np.savez_compressed(
            out_path,
            X_ecg=X_ecg[indices],
            X_ppg=X_ppg[indices],
            y=y[indices],
        )
        unique, counts = np.unique(y[indices], return_counts=True)
        cls_dist = {CLASSES[u]: int(c) for u, c in zip(unique, counts)}
        print(f"  {split_name:5s}: {len(indices):6,} samples | {cls_dist}")

    print(f"\nSaved train/val/test splits → {PROCESSED_DIR}")


if __name__ == "__main__":
    print("Loading data …")
    X, y = load_combined_data()
    print(f"Loaded {len(X):,} raw segments.")

    print("Applying SMOTE …")
    X_sm, y_sm = apply_smote(X, y)

    print("Generating synthetic PPG …")
    X_ppg = generate_ppg_dataset(X_sm, y_sm)

    print("Splitting dataset …")
    split_and_save(X_sm, X_ppg, y_sm)
    print("\nDone! Ready for training.")
