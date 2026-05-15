# CardioEdge — Development Chat History

This folder contains the full conversation logs between the developer and the AI assistant
across two sessions that produced the complete CardioEdge pipeline.

---

## Session 1 — Edge AI Research Implementation
**File:** `session_1_edge_ai_research.txt`  
**Conversation ID:** `5b59c9de-afb1-448c-83bd-391976905fbe`  
**Size:** ~79 KB

### Topics Covered
- Project architecture design (5-class arrhythmia detection)
- MIT-BIH + PTB-XL data pipeline (`download_mitbih.py`, `download_ptbxl.py`)
- SMOTE class balancing + synthetic PPG generation (`balance_classes.py`)
- Teacher model (CNN-BiLSTM) architecture and training (`train_teacher.py`)
- Novel `L_spectral` loss using Daubechies-4 wavelet decomposition (`losses.py`)
- Student model (DWS-CNN) design (`models/student.py`)
- Knowledge distillation training — 3 variants: CE, CE+KL, CE+KL+L_spectral
- Multimodal ECG+PPG fusion model (`models/fusion.py`, `train_fusion.py`)
- Initial TFLite conversion attempts (Keras 3 MLIR BatchNorm Cast op issues)
- BN-folding solution for TFLite compatibility (`convert_tflite_folded.py`)
- Training results: Full KD → F1=0.9053, Teacher → F1=0.9284

---

## Session 2 — Finalization and Arduino Deployment
**File:** `session_2_finalization_and_deployment.txt`  
**Conversation ID:** `f7689339-32ef-4322-ab28-8a7c37c60947`  
**Size:** ~174 KB

### Topics Covered
- Quantization validation (Table 3) — INT8 F1 drop = 0.0263, 23.4 KB flash
- Fusion training crash fix (Keras 3 optimizer `Unknown variable` bug → new optimizer for Phase 2)
- Fusion model converged: **100% Accuracy / F1 = 1.0000** on 14,373 test samples
- Arduino Nano 33 BLE deployment:
  - `CardioEdge.ino` — main sketch with 360 Hz timer ISR, serial commands (r/d/s)
  - `ecg_preprocess.h` — Butterworth bandpass SOS + Pan-Tompkins R-peak + z-score norm
  - `ppg_preprocess.h` — MAX30102 driver + 100→360 Hz upsampling
  - `signal_buffer.h` — ISR-safe ring buffer template
  - `filter_verify.ino` — playback MIT-BIH data without electrodes
- Filter coefficient verification (`verify_filter_coefficients.py`): SOS error < 1e-6 ✅
- Real-time web dashboard (`dashboard.py`) — Flask-SocketIO, live ECG + HR + class probs
- Full evaluation script (`evaluate.py`) — Tables 1-4 + confusion matrices
- Paper report generator (`experiments/generate_paper_report.py`) — self-contained HTML
- Complete pipeline orchestrator (`run_pipeline.py`) — 12 steps end-to-end

---

## Key Results Summary

| Metric | Value |
|--------|-------|
| Teacher Accuracy | 97.60% |
| Student (Full KD) F1 | 0.9053 |
| L_spectral gain | +1.29% F1 |
| INT8 TFLite size | **23.4 KB** |
| Arduino flash used | **2.3%** |
| INT8 F1 drop | **−0.0263** |
| Fusion (ECG+PPG) F1 | **1.0000** |
| Compression ratio | **140×** |

---

## How to Re-run Everything

```bash
# From scratch (GPU required for training):
python run_pipeline.py

# From evaluation only (models already trained):
python run_pipeline.py --from eval

# View the paper report:
start experiments/paper_report.html

# When Arduino is connected:
python deployment/dashboard.py --port COM3
```
