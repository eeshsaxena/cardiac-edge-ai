"""
precompute_teacher.py
─────────────────────
Run ONCE: loads the teacher Keras model and saves:
  - teacher_train_outputs.npz  → logits + feature maps for train set
  - teacher_val_outputs.npz    → logits + feature maps for val set
  - teacher_test_outputs.npz   → logits for test evaluation

PyTorch student training then reads these directly — no TF needed in GPU loop.
Run: python training/precompute_teacher.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import tensorflow as tf
import training.losses  # register custom loss

from config import PROCESSED_DIR, MODELS_DIR, LOGS_DIR
from models.teacher import build_teacher

TEACHER_PATH = os.path.join(MODELS_DIR, "teacher_best.keras")
OUT_DIR = os.path.join(LOGS_DIR, "teacher_cache")
os.makedirs(OUT_DIR, exist_ok=True)

BATCH = 512  # large batch fine for CPU inference


def load_split(split):
    data = np.load(os.path.join(PROCESSED_DIR, f"{split}.npz"))
    X = data["X_ecg"][..., np.newaxis].astype(np.float32)
    y = data["y"].astype(np.int32)
    return X, y


def infer_teacher(model, feat_model, X, batch=BATCH):
    """Returns (logits, features) as numpy arrays."""
    logits_list, feat_list = [], []
    for i in range(0, len(X), batch):
        xb = X[i:i+batch]
        logits_list.append(model(xb, training=False).numpy())
        feat_list.append(feat_model(xb, training=False).numpy())
        if i % (batch * 10) == 0:
            print(f"  {i}/{len(X)} ...", flush=True)
    return np.concatenate(logits_list), np.concatenate(feat_list)


def main():
    print("=" * 60)
    print("  TEACHER PRE-COMPUTATION")
    print("=" * 60)

    # Load teacher
    print(f"\nLoading {TEACHER_PATH} ...")
    loaded = tf.keras.models.load_model(TEACHER_PATH, compile=False)
    teacher_full, teacher_feats = build_teacher()
    teacher_full.set_weights(loaded.get_weights())
    print(f"Teacher loaded: {teacher_full.count_params():,} params")

    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split} split ...")
        X, y = load_split(split)
        print(f"  Samples: {len(X)}")

        if split == "test":
            # Only need logits for test evaluation
            logits_list = []
            for i in range(0, len(X), BATCH):
                logits_list.append(teacher_full(X[i:i+BATCH], training=False).numpy())
            logits = np.concatenate(logits_list)
            out_path = os.path.join(OUT_DIR, f"teacher_{split}.npz")
            np.savez_compressed(out_path, logits=logits, y=y)
        else:
            logits, features = infer_teacher(teacher_full, teacher_feats, X)
            out_path = os.path.join(OUT_DIR, f"teacher_{split}.npz")
            np.savez_compressed(out_path, logits=logits, features=features, y=y)

        print(f"  Saved -> {out_path}")
        print(f"  logits shape: {logits.shape}  max: {logits.max():.4f}")

    print("\n✓ Teacher pre-computation complete.")
    print(f"  Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
