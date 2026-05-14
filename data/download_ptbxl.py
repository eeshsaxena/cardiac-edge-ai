"""
download_ptbxl.py
─────────────────
Downloads PTB-XL (21,837 records) from PhysioNet and extracts
additional AF and LBBB samples to supplement MIT-BIH.

PTB-XL uses 12-lead ECG; we take lead I (index 0) resampled to 360 Hz.
SCP codes used: AFIB→AF, LBBB→LBBB

Run:  python data/download_ptbxl.py
      (first download: ~1.7 GB, cached in data/raw/ptb-xl/)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import wfdb
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm

from config import DATA_DIR, PROCESSED_DIR, WINDOW_LEN, CLASSES, CLASS_IDX

PTB_DB      = "ptb-xl/1.0.3"
PTB_LOCAL   = os.path.join(DATA_DIR, "ptb-xl")
MAX_PER_CLS = 4000   # cap per class to avoid massive imbalance

# SCP-ECG codes → our class labels
SCP_TO_CLASS = {
    "AFIB":  "AF",
    "AFL":   "AF",    # atrial flutter → AF bucket
    "LBBB":  "LBBB",
}


def load_ptbxl_metadata() -> pd.DataFrame:
    """Load the PTB-XL index CSV from PhysioNet."""
    try:
        meta_path = os.path.join(PTB_LOCAL, "ptbxl_database.csv")
        if not os.path.exists(meta_path):
            print("Downloading PTB-XL metadata CSV …")
            import urllib.request
            os.makedirs(PTB_LOCAL, exist_ok=True)
            url = ("https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv")
            urllib.request.urlretrieve(url, meta_path)
        return pd.read_csv(meta_path, index_col="ecg_id")
    except Exception as e:
        print(f"[ERROR] Could not load PTB-XL metadata: {e}")
        return None


def extract_ptbxl_beats(df: pd.DataFrame):
    """
    For each record labelled AF or LBBB in PTB-XL, extract multiple
    1-second windows (non-overlapping, skip first/last 2s for noise).
    """
    all_beats = {c: [] for c in CLASSES}
    counts    = {c: 0  for c in CLASSES}

    import ast
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    target_records = []
    for ecg_id, row in df.iterrows():
        codes = row["scp_codes"]
        for scp, conf in codes.items():
            if scp in SCP_TO_CLASS and conf >= 80:   # ≥80% confidence
                cls = SCP_TO_CLASS[scp]
                if counts[cls] < MAX_PER_CLS:
                    target_records.append((ecg_id, row["filename_lr"], cls))
                    counts[cls] += 1
                break

    print(f"PTB-XL target records: {len(target_records)}")
    for ecg_id, fname, cls in tqdm(target_records, desc="PTB-XL"):
        try:
            rec = wfdb.rdrecord(
                os.path.join(PTB_LOCAL, fname),
                channels=[0],     # Lead I
                sampfrom=0,
            )
        except Exception:
            try:
                rec = wfdb.rdrecord(fname, pn_dir=PTB_DB, channels=[0])
            except Exception as e:
                continue

        sig = rec.p_signal[:, 0].astype(np.float32)

        # Resample from 100 Hz (PTB-XL LR) → 360 Hz
        target_len = int(len(sig) * 360 / rec.fs)
        sig = resample(sig, target_len).astype(np.float32)

        # Extract non-overlapping 1-second windows (skip first+last 2s)
        start_s = 2 * 360
        for w_start in range(start_s, len(sig) - WINDOW_LEN, WINDOW_LEN):
            window = sig[w_start : w_start + WINDOW_LEN]
            mu, sigma = window.mean(), window.std()
            if sigma < 1e-6:
                continue
            window = (window - mu) / sigma
            all_beats[cls].append(window)

    print(f"\nPTB-XL class distribution:")
    for cls in CLASSES:
        print(f"  {cls:6s}: {len(all_beats[cls]):6,} windows")

    return all_beats


def merge_and_save(all_beats_ptbxl: dict):
    """Merge PTB-XL beats into existing MIT-BIH .npz file."""
    mitbih_path = os.path.join(PROCESSED_DIR, "mitbih_beats.npz")
    if not os.path.exists(mitbih_path):
        print("[ERROR] Run download_mitbih.py first.")
        return

    data  = np.load(mitbih_path)
    X_old = data["X"]
    y_old = data["y"]

    X_new, y_new = [], []
    for cls_idx, cls in enumerate(CLASSES):
        for seg in all_beats_ptbxl[cls]:
            X_new.append(seg)
            y_new.append(cls_idx)

    if len(X_new) == 0:
        print("No PTB-XL beats extracted. Keeping MIT-BIH only.")
        return

    X_combined = np.concatenate([X_old, np.array(X_new, dtype=np.float32)])
    y_combined = np.concatenate([y_old, np.array(y_new, dtype=np.int32)])

    out_path = os.path.join(PROCESSED_DIR, "combined_beats.npz")
    np.savez_compressed(out_path, X=X_combined, y=y_combined)
    print(f"\nCombined dataset saved → {out_path}")
    print(f"  Total samples: {len(X_combined):,}")

    unique, counts = np.unique(y_combined, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {CLASSES[u]:6s}: {c:6,}")


if __name__ == "__main__":
    df = load_ptbxl_metadata()
    if df is not None:
        beats = extract_ptbxl_beats(df)
        merge_and_save(beats)
    else:
        print("Falling back to MIT-BIH only — copying as combined_beats.npz")
        import shutil
        src = os.path.join(PROCESSED_DIR, "mitbih_beats.npz")
        dst = os.path.join(PROCESSED_DIR, "combined_beats.npz")
        shutil.copy(src, dst)
