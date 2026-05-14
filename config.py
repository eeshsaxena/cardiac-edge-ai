"""
Central configuration for the CardioEdge pipeline.
Change paths and hyperparameters here — everything else reads from this file.
"""
import os

# ─────────────────────────── Paths ───────────────────────────
ROOT          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "saved_models")
LOGS_DIR      = os.path.join(ROOT, "experiments", "logs")
FIGURES_DIR   = os.path.join(ROOT, "experiments", "figures")
TFLITE_DIR    = os.path.join(ROOT, "deployment", "tflite")

for d in [DATA_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR, TFLITE_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────── Signal settings ────────────────────
ECG_FS        = 360          # Hz — MIT-BIH native sampling rate
WINDOW_LEN    = 360          # samples = 1 second per beat segment
PPG_FS        = 100          # Hz — MAX30102 native (upsampled to ECG_FS in model)
PPG_WINDOW    = WINDOW_LEN   # after upsampling

# ─────────────────────── Class definitions ──────────────────
CLASSES       = ["N", "AF", "VT", "PVC", "LBBB"]
NUM_CLASSES   = len(CLASSES)
CLASS_IDX     = {c: i for i, c in enumerate(CLASSES)}

# MIT-BIH annotation symbols for each class
MITBIH_LABELS = {
    "N":    ["N", "·", "L", "R", "e", "j"],   # normal variants
    "AF":   ["A", "a", "J", "S"],              # supraventricular ectopic
    "VT":   ["V", "E"],                         # ventricular ectopic (use run detection for VT)
    "PVC":  ["V"],                              # isolated PVC (single beat)
    "LBBB": ["L"],                              # left bundle branch block
}

# Class weights for imbalanced loss (N is dominant → lower weight)
CLASS_WEIGHTS = {0: 0.15, 1: 1.8, 2: 2.2, 3: 1.6, 4: 2.5}

# ─────────────────────── Model hyperparams ──────────────────
TEACHER_FILTERS    = [64, 128, 256, 256]
STUDENT_FILTERS    = [32, 64, 64]
PPG_FILTERS        = [16, 32]
LSTM_UNITS         = 128
STUDENT_FC_UNITS   = 32
FUSION_FC_UNITS    = 64

# ─────────────────────── Training settings ──────────────────
BATCH_SIZE         = 64
TEACHER_EPOCHS     = 60
STUDENT_EPOCHS     = 80
FUSION_EPOCHS      = 40
LEARNING_RATE      = 1e-3
KD_TEMPERATURE     = 4.0       # τ for KL soft labels

# KD loss weights
LAMBDA_CE          = 1.0
LAMBDA_KL          = 0.7
LAMBDA_SPECTRAL    = 0.4

# Wavelet settings for L_spectral
WAVELET            = "db4"     # Daubechies-4 — standard for ECG
WAVELET_LEVEL      = 4         # decomposition levels
SPECTRAL_BAND_HZ   = (0.5, 40) # arrhythmia-relevant band

# ─────────────────────── Evaluation ─────────────────────────
VAL_SPLIT          = 0.15
TEST_SPLIT         = 0.15
RANDOM_SEED        = 42
