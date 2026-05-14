"""
losses.py
─────────
Custom loss functions for knowledge distillation:

  L_total = λ₁·L_CE + λ₂·L_KL + λ₃·L_spectral

  L_CE       — CrossEntropy with class weights (handles PVC/VT imbalance)
  L_KL       — Soft-label KL divergence at temperature τ (dark knowledge)
  L_spectral — Wavelet-domain L2 loss between teacher and student
               feature maps, restricted to 0.5–40 Hz arrhythmia band
               using Daubechies-4 decomposition

Reference:
  Frequency band selection: AHA ECG standard (0.05–150 Hz diagnostic;
  arrhythmia-discriminating: 0.5–40 Hz)
  Wavelet choice: db4 — 4 vanishing moments, optimal for QRS morphology
"""
import tensorflow as tf
import keras
import numpy as np
import pywt
from config import (
    KD_TEMPERATURE, LAMBDA_CE, LAMBDA_KL, LAMBDA_SPECTRAL,
    WAVELET, WAVELET_LEVEL, ECG_FS, SPECTRAL_BAND_HZ, CLASS_WEIGHTS
)


# ── Utility: wavelet band mask ───────────────────────────────────────────────

def _compute_band_mask(signal_length: int, fs: int = ECG_FS) -> list[bool]:
    """
    For each wavelet level, determine whether its frequency band
    overlaps with SPECTRAL_BAND_HZ = (0.5, 40) Hz.

    Daubechies-4 at level k covers: [fs/2^(k+1), fs/2^k] Hz
    Level 1: [90, 180] Hz  ← above arrhythmia band
    Level 2: [45, 90]  Hz  ← above
    Level 3: [22.5, 45] Hz ← partially overlaps (keep)
    Level 4: [11.25, 22.5] Hz ← inside band (keep)
    Approx:  [0, 11.25]  Hz ← inside band (keep)
    """
    f_low, f_high = SPECTRAL_BAND_HZ
    mask = []
    for lvl in range(1, WAVELET_LEVEL + 1):
        band_hi = fs / (2 ** lvl)
        band_lo = fs / (2 ** (lvl + 1))
        overlaps = band_lo <= f_high and band_hi >= f_low
        mask.append(overlaps)
    # Approximation coefficients (lowest band) — always in band
    mask.append(True)
    return mask


BAND_MASK = _compute_band_mask(signal_length=360)


# ── L_CE: Class-weighted cross-entropy ──────────────────────────────────────

@keras.saving.register_keras_serializable(package="cardioedge")
class WeightedCrossEntropy(tf.keras.losses.Loss):
    """
    Standard cross-entropy with per-class weights to handle
    PVC/VT class imbalance.
    """
    def __init__(self, class_weights: dict = CLASS_WEIGHTS, **kwargs):
        super().__init__(**kwargs)
        self.weights = tf.constant(
            [class_weights[i] for i in sorted(class_weights)],
            dtype=tf.float32
        )

    def call(self, y_true, y_pred):
        # y_true: integer labels (batch,)
        # y_pred: softmax probabilities (batch, num_classes)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        # Gather per-sample weights
        sample_weights = tf.gather(self.weights, tf.cast(y_true, tf.int32))
        return tf.reduce_mean(ce * sample_weights)


# ── L_KL: Soft-label KL divergence ──────────────────────────────────────────

def kl_divergence_loss(
    teacher_logits: tf.Tensor,
    student_logits: tf.Tensor,
    temperature: float = KD_TEMPERATURE,
) -> tf.Tensor:
    """
    KL divergence between teacher and student soft distributions.
    Scaled by τ² as per Hinton et al. (2015).

    Args:
        teacher_logits: raw (pre-softmax) teacher outputs (batch, C)
        student_logits: raw (pre-softmax) student outputs (batch, C)
    """
    T = temperature
    p_teacher = tf.nn.softmax(teacher_logits / T)
    p_student = tf.nn.softmax(student_logits / T)

    kl = tf.keras.losses.KLDivergence()(p_teacher, p_student)
    return kl * (T ** 2)


# ── L_spectral: Wavelet-domain feature map loss ──────────────────────────────

def _wavelet_decompose_numpy(feature_map: np.ndarray) -> list[np.ndarray]:
    """
    Apply Daubechies-4 wavelet decomposition to each channel of a
    feature map and return only the arrhythmia-band coefficients.

    Args:
        feature_map: shape (T, C) — one sample's feature map
    Returns:
        list of coefficient arrays from relevant frequency bands
    """
    band_coeffs = []
    for c in range(feature_map.shape[-1]):
        signal = feature_map[:, c].astype(np.float64)
        coeffs = pywt.wavedec(signal, WAVELET, level=WAVELET_LEVEL)
        # coeffs[0] = approx, coeffs[1..] = details (high→low freq)
        # Band mask: details are indexed 1..WAVELET_LEVEL (high→low)
        # mask[0] = level 1 (highest detail), mask[-1] = approx
        for k, (coeff, keep) in enumerate(zip(reversed(coeffs[1:]), BAND_MASK[:-1])):
            if keep:
                band_coeffs.append(coeff)
        if BAND_MASK[-1]:  # approximation
            band_coeffs.append(coeffs[0])
    return band_coeffs


@tf.function
def spectral_loss(
    teacher_features: tf.Tensor,
    student_features: tf.Tensor,
) -> tf.Tensor:
    """
    L_spectral = ||WT_band(F_student) - WT_band(F_teacher)||²_F

    Computed per-sample, averaged over batch.

    Both inputs: (batch, T, C) — intermediate conv feature maps.
    Teacher and student feature maps may have different channel counts;
    we match channels via global average and compare spectral energy.

    Strategy for channel mismatch:
      1. GAP over channels → (batch, T)  one representative signal
      2. Wavelet decompose that signal
      3. Compare band-filtered coefficients
    """
    # Reduce channels → (batch, T)
    t_sig = tf.reduce_mean(teacher_features, axis=-1)  # (batch, T)
    s_sig = tf.reduce_mean(student_features, axis=-1)  # (batch, T)

    # Pad/crop to same length
    t_len = tf.shape(t_sig)[1]
    s_len = tf.shape(s_sig)[1]
    min_len = tf.minimum(t_len, s_len)
    t_sig = t_sig[:, :min_len]
    s_sig = s_sig[:, :min_len]

    # Compute wavelet loss via numpy_function (pywt is numpy-based)
    def _batch_wavelet_loss(t_np, s_np):
        total = 0.0
        count = 0
        for i in range(len(t_np)):
            t_feat = t_np[i].reshape(-1, 1)   # (T, 1)
            s_feat = s_np[i].reshape(-1, 1)
            t_coeffs = _wavelet_decompose_numpy(t_feat)
            s_coeffs = _wavelet_decompose_numpy(s_feat)
            for tc, sc in zip(t_coeffs, s_coeffs):
                diff = tc - sc[:len(tc)] if len(sc) >= len(tc) else tc[:len(sc)] - sc
                total += np.sum(diff ** 2)
                count += len(diff)
        return np.float32(total / max(count, 1))

    loss = tf.numpy_function(
        _batch_wavelet_loss,
        [t_sig, s_sig],
        tf.float32
    )
    return loss


# ── Combined KD loss ─────────────────────────────────────────────────────────

class KnowledgeDistillationLoss:
    """
    Combines CE + KL + Spectral losses with configurable weights.
    Used in the custom training loop in train_student_kd.py.
    """
    def __init__(
        self,
        lambda_ce:       float = LAMBDA_CE,
        lambda_kl:       float = LAMBDA_KL,
        lambda_spectral: float = LAMBDA_SPECTRAL,
        temperature:     float = KD_TEMPERATURE,
        class_weights:   dict  = CLASS_WEIGHTS,
    ):
        self.lambda_ce       = lambda_ce
        self.lambda_kl       = lambda_kl
        self.lambda_spectral = lambda_spectral
        self.temperature     = temperature
        self.ce_loss         = WeightedCrossEntropy(class_weights)

    def __call__(
        self,
        y_true:           tf.Tensor,   # integer labels (batch,)
        student_probs:    tf.Tensor,   # softmax output  (batch, C)
        student_logits:   tf.Tensor,   # pre-softmax     (batch, C)
        teacher_logits:   tf.Tensor,   # pre-softmax     (batch, C)
        teacher_features: tf.Tensor,   # (batch, T, C_t)
        student_features: tf.Tensor,   # (batch, T, C_s)
    ) -> tuple[tf.Tensor, dict]:

        l_ce       = self.ce_loss(y_true, student_probs)
        l_kl       = kl_divergence_loss(teacher_logits, student_logits, self.temperature)
        l_spec     = spectral_loss(teacher_features, student_features)

        total = (self.lambda_ce * l_ce
                 + self.lambda_kl * l_kl
                 + self.lambda_spectral * l_spec)

        return total, {
            "loss_ce":       l_ce,
            "loss_kl":       l_kl,
            "loss_spectral": l_spec,
        }
