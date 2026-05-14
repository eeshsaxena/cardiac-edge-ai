"""
download_mitbih.py
──────────────────
Downloads all MIT-BIH Arrhythmia Database records via wfdb,
then extracts and labels beat segments for the 5 target classes.

Run:  python data/download_mitbih.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import wfdb
from tqdm import tqdm
import pickle

from config import DATA_DIR, PROCESSED_DIR, WINDOW_LEN, ECG_FS, CLASSES, CLASS_IDX

# ── MIT-BIH record list (all 48 standard records) ──────────────────────────
MITBIH_RECORDS = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
    122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
    209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
    222, 223, 224, 225, 228, 230, 231, 232, 233, 234,
]

# Annotation → class mapping (order matters: PVC before VT so isolated V→PVC)
# VT = 3+ consecutive V beats (run detection); single V = PVC
SYMBOL_TO_CLASS = {
    # Normal
    "N": "N", ".": "N",
    # LBBB
    "L": "LBBB",
    # RBBB — exclude (not in our 5 classes, skip)
    "R": None,
    # Supraventricular (AF category)
    "A": "AF", "a": "AF", "J": "AF", "S": "AF", "e": "AF", "j": "AF",
    # Ventricular — resolved to VT/PVC after run detection below
    "V": "PVC",   # default: PVC; overridden to VT for runs ≥3
    "E": "PVC",
    # Fusion / paced / unknown → skip
    "F": None, "/": None, "f": None, "Q": None, "!": None,
}

HALF_WIN = WINDOW_LEN // 2   # samples before/after R-peak


def extract_beats(record_name: str, db_path: str):
    """
    Returns list of (segment_np, class_str) for one MIT-BIH record.
    segment_np shape: (WINDOW_LEN,)  — lead II (channel 0)
    """
    try:
        record = wfdb.rdrecord(record_name, pn_dir="mitdb")
        annotation = wfdb.rdann(record_name, "atr", pn_dir="mitdb")
    except Exception as e:
        print(f"  [WARN] Skipping {record_name}: {e}")
        return []

    signal = record.p_signal[:, 0]   # lead II (MLII)
    sig_len = len(signal)

    symbols = annotation.symbol
    r_peaks = annotation.sample

    # ── VT run detection (≥3 consecutive V/E beats → label all as VT) ───
    vt_indices = set()
    run = []
    for idx, sym in enumerate(symbols):
        if sym in ("V", "E"):
            run.append(idx)
        else:
            if len(run) >= 3:
                vt_indices.update(run)
            run = []
    if len(run) >= 3:
        vt_indices.update(run)

    beats = []
    for i, (peak, sym) in enumerate(zip(r_peaks, symbols)):
        # Resolve class
        if i in vt_indices:
            cls = "VT"
        else:
            cls = SYMBOL_TO_CLASS.get(sym, None)
        if cls is None:
            continue  # skip unknown / excluded

        # Extract window centred on R-peak
        start = peak - HALF_WIN
        end   = peak + HALF_WIN
        if start < 0 or end > sig_len:
            continue  # skip edge segments

        segment = signal[start:end].astype(np.float32)

        # Per-segment z-score normalisation
        mu, sigma = segment.mean(), segment.std()
        if sigma < 1e-6:
            continue  # flat line — skip
        segment = (segment - mu) / sigma

        beats.append((segment, cls))

    return beats


def download_and_extract():
    all_beats  = {c: [] for c in CLASSES}
    total      = 0

    print("Downloading & extracting MIT-BIH beats …")
    for rec in tqdm(MITBIH_RECORDS, desc="Records"):
        rec_str = str(rec)
        beats   = extract_beats(rec_str, DATA_DIR)
        for seg, cls in beats:
            all_beats[cls].append(seg)
            total += 1

    print(f"\nClass distribution (MIT-BIH):")
    for cls in CLASSES:
        print(f"  {cls:6s}: {len(all_beats[cls]):6,} beats")
    print(f"  Total : {total:6,} beats\n")

    # Save as numpy arrays
    X, y = [], []
    for cls_idx, cls in enumerate(CLASSES):
        for seg in all_beats[cls]:
            X.append(seg)
            y.append(cls_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    out_path = os.path.join(PROCESSED_DIR, "mitbih_beats.npz")
    np.savez_compressed(out_path, X=X, y=y)
    print(f"Saved → {out_path}")
    return X, y


if __name__ == "__main__":
    download_and_extract()
