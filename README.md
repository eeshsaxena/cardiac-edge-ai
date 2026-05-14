# CardioEdge — Edge AI Cardiac Arrhythmia Detection

> **First MCU-deployed 5-class arrhythmia detector** with frequency-preserving knowledge distillation, ECG+PPG late fusion, and rigorous power profiling.

## Novel Contributions
| # | Contribution | Status |
|---|---|---|
| 1 | Frequency-preserving KD loss (L_spectral, Daubechies-4) | 🔄 In progress |
| 2 | 5-class MCU detection (AF, VT, PVC, LBBB, Normal) | 🔄 In progress |
| 3 | ECG + PPG late fusion with learned scalar α | 🔄 In progress |
| 4 | INA219 power profiling (hardware phase) | ⏳ Pending hardware |
| 5 | Physical BLE prototype + React Native app | ⏳ Pending hardware |

## Targets
- Accuracy > 97% | Macro-F1 > 0.95 | Latency < 50ms | Power < 10mW

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline (Software Phase)

```bash
# 1. Download & segment MIT-BIH
python data/download_mitbih.py

# 2. Supplement with PTB-XL (AF + LBBB)
python data/download_ptbxl.py

# 3. Balance classes + generate synthetic PPG
python data/balance_classes.py

# 4. Train teacher
python training/train_teacher.py

# 5. Knowledge distillation → student
python training/train_student_kd.py

# 6. Train ECG+PPG fusion model
python training/train_fusion.py

# 7. Full evaluation (all tables + confusion matrix)
python training/evaluate.py

# 8. Convert to TFLite INT8
python deployment/convert_tflite.py

# 9. Validate quantized model accuracy
python deployment/validate_quantized.py

# OR run entire pipeline in one command:
python run_pipeline.py
```

## Repository Structure
```
cardiac-edge-ai/
├── config.py               ← All hyperparameters & paths
├── run_pipeline.py         ← One-command full pipeline
├── data/
│   ├── download_mitbih.py
│   ├── download_ptbxl.py
│   └── balance_classes.py  ← SMOTE + synthetic PPG
├── models/
│   ├── teacher.py          ← CNN-BiLSTM (1.8M params)
│   ├── student.py          ← TinyConv DWS (45K params)
│   ├── ppg_branch.py       ← PPG sub-network (8K params)
│   └── fusion.py           ← Late fusion with α
├── training/
│   ├── losses.py           ← L_CE + L_KL + L_spectral
│   ├── train_teacher.py
│   ├── train_student_kd.py ← Custom KD training loop
│   ├── train_fusion.py
│   └── evaluate.py         ← 4 paper tables + plots
└── deployment/
    ├── convert_tflite.py   ← INT8 quantization
    └── validate_quantized.py
```

## Citation
```bibtex
@article{cardioedge2026,
  title  = {CardioEdge: Frequency-Preserving Knowledge Distillation for
             5-Class Arrhythmia Detection on Microcontrollers},
  author = {[Your Name]},
  year   = {2026}
}
```
