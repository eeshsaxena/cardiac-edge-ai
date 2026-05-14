"""
train_student_kd.py
───────────────────
Custom training loop for knowledge distillation.
Uses the full L_total = λ₁·CE + λ₂·KL + λ₃·L_spectral loss.

Saves per-epoch metrics to CSV for ablation study.
Three variants trained automatically:
  1. CE only        → student_ce_only.keras
  2. CE + KL        → student_ce_kl.keras
  3. CE + KL + Spec → student_full_kd.keras  ← main contribution

Run:  python training/train_student_kd.py
"""
import os, sys, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tensorflow as tf

from config import (
    PROCESSED_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR,
    BATCH_SIZE, STUDENT_EPOCHS, LEARNING_RATE,
    LAMBDA_CE, LAMBDA_KL, LAMBDA_SPECTRAL,
    KD_TEMPERATURE, CLASS_WEIGHTS, CLASSES, NUM_CLASSES, RANDOM_SEED
)
from models.teacher import build_teacher
from models.student import build_student_with_intermediate
from training.losses import KnowledgeDistillationLoss, WeightedCrossEntropy

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_split(split: str):
    path = os.path.join(PROCESSED_DIR, f"{split}.npz")
    data = np.load(path)
    X = data["X_ecg"][..., np.newaxis]
    y = data["y"]
    return X, y


def make_tf_dataset(X, y, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=RANDOM_SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_macro_f1(y_true, y_pred_probs, num_classes=NUM_CLASSES):
    y_pred = np.argmax(y_pred_probs, axis=1)
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ── Single-epoch training step ────────────────────────────────────────────────

@tf.function
def train_step(
    x_batch, y_batch,
    student_model, student_feat_extractor,
    teacher_model, teacher_feat_extractor,
    optimizer, kd_loss_fn,
    use_kl: bool, use_spectral: bool,
):
    with tf.GradientTape() as tape:
        # Teacher forward (no gradient)
        t_probs    = teacher_model(x_batch,          training=False)
        t_logits   = tf.math.log(t_probs + 1e-8)    # approximate logits from softmax
        t_features = teacher_feat_extractor(x_batch, training=False)

        # Student forward
        s_probs    = student_model(x_batch,          training=True)
        s_logits   = tf.math.log(s_probs + 1e-8)
        s_features = student_feat_extractor(x_batch, training=True)

        # Compute loss
        if use_kl and use_spectral:
            loss, sub_losses = kd_loss_fn(
                y_batch, s_probs, s_logits, t_logits, t_features, s_features
            )
        elif use_kl:
            from training.losses import kl_divergence_loss
            l_ce = kd_loss_fn.ce_loss(y_batch, s_probs)
            l_kl = kl_divergence_loss(t_logits, s_logits, KD_TEMPERATURE)
            loss = LAMBDA_CE * l_ce + LAMBDA_KL * l_kl
            sub_losses = {"loss_ce": l_ce, "loss_kl": l_kl, "loss_spectral": 0.0}
        else:
            l_ce  = kd_loss_fn.ce_loss(y_batch, s_probs)
            loss  = l_ce
            sub_losses = {"loss_ce": l_ce, "loss_kl": 0.0, "loss_spectral": 0.0}

    grads = tape.gradient(loss, student_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
    return loss, sub_losses, s_probs


def train_variant(
    variant_name: str,
    use_kl: bool,
    use_spectral: bool,
    teacher_model: tf.keras.Model,
    teacher_feat_extractor: tf.keras.Model,
    X_train, y_train, X_val, y_val,
):
    print(f"\n{'='*60}")
    print(f"  STUDENT KD — Variant: {variant_name}")
    print(f"  use_kl={use_kl}  use_spectral={use_spectral}")
    print(f"{'='*60}")

    student, feat_ext = build_student_with_intermediate()
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    kd_loss   = KnowledgeDistillationLoss()

    train_ds = make_tf_dataset(X_train, y_train, shuffle=True)
    val_ds   = make_tf_dataset(X_val,   y_val,   shuffle=False)

    best_val_f1  = 0.0
    best_weights = None
    log_rows     = []
    lr_schedule  = LEARNING_RATE
    patience_cnt = 0
    PATIENCE     = 10

    for epoch in range(1, STUDENT_EPOCHS + 1):
        t0 = time.time()
        epoch_loss = 0.0
        steps      = 0

        for x_batch, y_batch in train_ds:
            loss, sub_losses, s_probs = train_step(
                x_batch, y_batch,
                student, feat_ext,
                teacher_model, teacher_feat_extractor,
                optimizer, kd_loss,
                use_kl, use_spectral,
            )
            epoch_loss += loss.numpy()
            steps += 1

        # Validation
        val_probs_list, val_y_list = [], []
        for x_val_b, y_val_b in val_ds:
            p = student(x_val_b, training=False).numpy()
            val_probs_list.append(p)
            val_y_list.append(y_val_b.numpy())

        val_probs = np.concatenate(val_probs_list)
        val_y     = np.concatenate(val_y_list)
        val_acc   = np.mean(np.argmax(val_probs, 1) == val_y)
        val_f1    = compute_macro_f1(val_y, val_probs)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{STUDENT_EPOCHS} | "
            f"loss={epoch_loss/steps:.4f} | "
            f"val_acc={val_acc:.4f} | val_F1={val_f1:.4f} | "
            f"{elapsed:.1f}s"
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": epoch_loss / steps,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        })

        # Best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = student.get_weights()
            patience_cnt = 0
        else:
            patience_cnt += 1

        # LR decay
        if patience_cnt == 5:
            lr_schedule *= 0.5
            optimizer.learning_rate.assign(lr_schedule)
            print(f"  ↓ LR → {lr_schedule:.2e}")

        # Early stop
        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    # Restore best
    student.set_weights(best_weights)

    # Save
    save_name = {
        "CE only":        "student_ce_only",
        "CE + KL":        "student_ce_kl",
        "CE + KL + Spec": "student_full_kd",
    }.get(variant_name, variant_name.replace(" ", "_").lower())

    save_path = os.path.join(MODELS_DIR, f"{save_name}.keras")
    student.save(save_path)
    print(f"\nSaved → {save_path}")
    print(f"Best val Macro-F1 = {best_val_f1:.4f}")

    # Save CSV log
    csv_path = os.path.join(LOGS_DIR, f"{save_name}_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    return student, best_val_f1


def train_all_variants():
    print("Loading data …")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}")

    # Load trained teacher
    teacher_path = os.path.join(MODELS_DIR, "teacher_best.keras")
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(
            f"Teacher model not found at {teacher_path}\n"
            "Run: python training/train_teacher.py first."
        )
    print(f"\nLoading teacher from {teacher_path} …")
    teacher_model = tf.keras.models.load_model(teacher_path)
    _, teacher_feat_extractor = build_teacher()
    # Copy weights from loaded model to feature extractor
    teacher_feat_extractor.set_weights(
        [w for w in teacher_model.weights
         if w.name.split("/")[0] in
            [l.name for l in teacher_feat_extractor.layers]]
    )
    # Simpler: rebuild and load by layer name
    teacher_full, teacher_feats = build_teacher()
    teacher_full.set_weights(teacher_model.get_weights())
    teacher_feats.set_weights(
        [teacher_full.get_layer(l.name).get_weights()[0]
         if teacher_full.get_layer(l.name).get_weights() else
         teacher_full.get_layer(l.name).get_weights()
         for l in teacher_feats.layers
         if teacher_full.get_layer(l.name).get_weights()]
    )
    # Cleanest approach: use functional sub-model sharing weights
    teacher_full, teacher_feats = build_teacher()
    teacher_full.load_weights(teacher_path, by_name=True, skip_mismatch=True)

    results = {}
    for name, (use_kl, use_spec) in [
        ("CE only",        (False, False)),
        ("CE + KL",        (True,  False)),
        ("CE + KL + Spec", (True,  True)),
    ]:
        _, f1 = train_variant(
            name, use_kl, use_spec,
            teacher_full, teacher_feats,
            X_train, y_train, X_val, y_val
        )
        results[name] = f1

    print("\n" + "=" * 60)
    print("  ABLATION SUMMARY (Val Macro-F1)")
    print("=" * 60)
    for name, f1 in results.items():
        print(f"  {name:20s}: {f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train_all_variants()
