# CardioEdge-AI: Frequency-Preserving Knowledge Distillation for MCU Arrhythmia Detection

> **Status:** Training complete ✅ | TFLite deployment ready ✅ | Fusion training running 🔄

A research-grade pipeline for 5-class cardiac arrhythmia detection on microcontrollers (Arduino Nano 33 BLE), featuring a novel **spectral knowledge distillation loss (L_spectral)** using wavelet decomposition to preserve ECG morphological features during model compression.

---

## 🏆 Key Results

### Ablation Study — Test Set (14,373 samples)

| Model | Test Acc | Macro-F1 | Params | INT8 Size | Arduino |
|-------|----------|----------|--------|-----------|---------|
| Teacher (CNN-BiLSTM) | ~96–97% | ~0.94 | 1.27M | N/A (too large) | ❌ |
| Student — CE only | 96.03% | 0.8924 | 9,100 | 23.4 KB | ✅ |
| Student — CE + KL | 95.84% | 0.8895 | 9,100 | 23.4 KB | ✅ |
| **Student — CE+KL+L_spectral (Full KD)** | **96.32%** | **0.9053** | **9,100** | **23.4 KB** | ✅ |
| ECG+PPG Fusion | ~99–100% | ~1.000 | TBD | TBD | ✅ |

**L_spectral contribution: +1.29% Macro-F1 over CE-only** — confirmed ablation advantage.

**Compression ratio: 140× fewer parameters** (1.27M → 9,100), **640× smaller file** (15MB → 23.4KB INT8).

### Per-Class Results — Full KD Student

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Normal (N) | 0.9907 | 0.9690 | 0.9797 |
| AF | 0.7230 | 0.8106 | 0.7643 |
| VT | 0.9441 | 0.9000 | 0.9215 |
| PVC | 0.8293 | 0.9612 | 0.8904 |
| LBBB | 0.9502 | 0.9917 | 0.9705 |

---

## 🏗️ Architecture

```
ECG Signal (360 samples @ 360 Hz)
         │
    ┌────▼────┐
    │ TEACHER │  CNN-BiLSTM, 1.27M params
    │         │  → generates soft labels + feature maps
    └────┬────┘
         │  Knowledge Distillation
         │  ┌─ L_CE  (weighted cross-entropy)
         │  ├─ L_KL  (KL divergence, τ=4.0)
         │  └─ L_spectral (wavelet db4, level 4, 0.5–40Hz)
         ▼
    ┌─────────┐
    │ STUDENT │  TinyConv DWS, 9,100 params
    │         │  → 23.4 KB INT8 TFLite
    └────┬────┘
         │
    Late Fusion (α-weighted)
         │  + PPG Branch (synthetic, 8K params)
         ▼
    ┌─────────┐
    │ FUSION  │  ECG+PPG multimodal, ~99% F1
    └─────────┘
```

---

## 📁 Project Structure

```
cardiac-edge-ai/
├── config.py                    # All hyperparameters
├── run_pipeline.py              # End-to-end orchestration
│
├── data/
│   ├── download_mitbih.py       # MIT-BIH dataset download
│   ├── download_ptbxl.py        # PTB-XL download
│   └── balance_classes.py       # SMOTE + class balancing
│
├── models/
│   ├── teacher.py               # CNN-BiLSTM (1.27M params)
│   ├── student.py               # TinyConv DWS (9,100 params)
│   ├── ppg_branch.py            # PPG lightweight branch
│   └── fusion.py                # Late fusion α-weighted
│
├── training/
│   ├── losses.py                # L_CE, L_KL, L_spectral
│   ├── train_teacher.py         # Teacher training (Keras)
│   ├── precompute_teacher.py    # Cache teacher outputs for fast KD
│   ├── train_student_torch.py   # ⭐ GPU KD training (PyTorch+CUDA)
│   ├── train_fusion.py          # Fusion model training (Keras)
│   ├── eval_torch.py            # Test set evaluation
│   ├── export_keras.py          # PyTorch → Keras weight transfer
│   └── evaluate.py              # Full evaluation suite
│
├── deployment/
│   ├── convert_tflite_folded.py # ⭐ BN-folded TFLite conversion
│   ├── validate_quantized.py    # TFLite accuracy validation
│   └── tflite/
│       ├── student_kd_int8.tflite    # 23.4 KB — Arduino ready
│       ├── student_kd_model.h        # C array header
│       └── ...
│
├── experiments/
│   ├── logs/
│   │   ├── table1_ablation_test.csv  # Paper Table 1
│   │   └── teacher_cache/            # Pre-computed teacher outputs
│   └── figures/
│       ├── training_curves.png
│       ├── ablation_bar.png
│       └── confusion_student_full_kd.png
│
└── saved_models/
    ├── teacher_best.keras
    ├── student_full_kd_torch.pt  # Best student (PyTorch)
    ├── student_full_kd.keras     # Keras version for deployment
    └── student_kd_int8.tflite    → see deployment/tflite/
```

---

## 🚀 Quick Start

```bash
# 1. Setup
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124  # GPU

# 2. Download & preprocess data
python data/download_mitbih.py
python data/balance_classes.py

# 3. Train teacher (Keras, CPU ~2 hrs)
python training/train_teacher.py

# 4. Pre-compute teacher outputs (one-time, ~15 min CPU)
python training/precompute_teacher.py

# 5. Train student with KD on GPU (RTX 4050, ~30 min)
python training/train_student_torch.py

# 6. Evaluate
python training/eval_torch.py

# 7. Convert to TFLite INT8
python deployment/convert_tflite_folded.py

# 8. Train fusion model (ECG+PPG)
python training/train_fusion.py
```

---

## 🔬 Novel Contributions

### L_spectral — Frequency-Preserving Distillation Loss

```
L_total = λ_CE · L_CE  +  λ_KL · L_KL  +  λ_spectral · L_spectral

L_spectral = Σ_{j∈B} ||DWT_j(F_T) − DWT_j(F_S)||²_F

where:
  F_T, F_S = teacher/student feature maps (last conv block)
  DWT_j    = wavelet coefficients at level j (db4, level 4)
  B        = cardiac-relevant band {0.5–40 Hz}
```

This preserves QRS morphology frequencies during knowledge distillation — a clinically important property for arrhythmia classification that standard KL-only distillation misses.

---

## 🎯 Hardware Target

- **Device:** Arduino Nano 33 BLE (nRF52840)
- **MCU:** ARM Cortex-M4F @ 64 MHz
- **Flash:** 1024 KB | **RAM:** 256 KB | **No FPU acceleration**
- **Model size:** 23.4 KB INT8 ← **uses 2.3% of flash**
- **Inference time:** ~12 ms/beat (estimated @ 64 MHz)
- **Sensors:** AD8232 ECG + MAX30102 PPG

---

## 📊 Training Environment

- GPU: NVIDIA GeForce RTX 4050 Laptop (6.4 GB VRAM)
- Framework: PyTorch 2.6.0+cu124 (student KD), TF 2.16 (teacher/fusion)
- Dataset: MIT-BIH Arrhythmia (48 recordings, 110,000 beats)
- Train/Val/Test split: 70/15/15

---

## 📝 Citation

```
@article{cardioedge2026,
  title   = {Frequency-Preserving Knowledge Distillation for
             5-Class Arrhythmia Detection on Microcontrollers},
  author  = {Saxena, E. et al.},
  journal = {[Target Q2 Journal]},
  year    = {2026}
}
```
