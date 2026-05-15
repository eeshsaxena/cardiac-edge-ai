# CardioEdge Arduino Deployment Guide

## Hardware Required

| Component | Model | Connection |
|-----------|-------|------------|
| Microcontroller | Arduino Nano 33 BLE | — |
| ECG Module | AD8232 | A0 (out), D10 (LO+), D11 (LO-), 3.3V, GND |
| PPG Module | MAX30102 | SDA=A4, SCL=A5, 3.3V, GND |
| Alert LED | Red LED + 220Ω | D2 → LED → GND |

### AD8232 Wiring
```
AD8232 Pin  →  Arduino Nano 33 BLE
─────────────────────────────────────
OUTPUT      →  A0
LO+         →  D10
LO-         →  D11
3.3V        →  3.3V
GND         →  GND
```

### MAX30102 Wiring
```
MAX30102 Pin  →  Arduino Nano 33 BLE
──────────────────────────────────────
SDA          →  A4
SCL          →  A5
INT          →  (not connected)
3.3V         →  3.3V
GND          →  GND
```

---

## Step-by-Step Setup

### 1 — Install Arduino IDE
Download from https://www.arduino.cc/en/software  (v2.x recommended)

### 2 — Install Board Support
In Arduino IDE → **Boards Manager**, search and install:
```
Arduino Mbed OS Nano Boards   (includes Nano 33 BLE)
```

### 3 — Install Libraries
In Arduino IDE → **Library Manager**, install:
```
Arduino_TensorFlowLite         ≥ 2.4.0
SparkFun MAX3010x Sensor Library  ≥ 1.1.2
```

> **Note:** The TFLite Micro library for Nano 33 BLE is packaged separately.  
> If `Arduino_TensorFlowLite` is not found, download from:  
> https://github.com/tensorflow/tflite-micro-arduino-examples

### 4 — Copy Sketch Files
Copy the entire `CardioEdge/` folder to your Arduino sketchbook:
```
%USERPROFILE%\Documents\Arduino\CardioEdge\
├── CardioEdge.ino
├── ecg_preprocess.h
├── ppg_preprocess.h
├── signal_buffer.h
└── student_kd_model.h      ← auto-generated from convert_tflite_folded.py
```

### 5 — Pre-deployment Filter Verification (no electrodes needed)

Before connecting to a human, verify the Arduino preprocessing matches Python:

**a. Upload the verification sketch:**
```
Arduino IDE → Open → filter_verify/filter_verify.ino
Upload to Nano 33 BLE
```

**b. Run the Python validator:**
```bash
pip install pyserial matplotlib scipy
python deployment/validate_arduino_signal.py --port COM3
```
*(Replace `COM3` with your actual Arduino port)*

**Expected output:**
- Live ECG waveform appears in the plot
- R-peaks detected within ±2 samples of Python reference
- Classes match Python TFLite ground truth:
  - Beat 0: N   (67% confidence)
  - Beat 1: N/AF
  - Beat 2: VT  (89% confidence)  ← clearest signal
  - Beat 3: PVC (46% confidence)
  - Beat 4: LBBB (85% confidence)

### 6 — Upload Main Sketch
```
Arduino IDE → Open → CardioEdge/CardioEdge.ino
Select board: Tools → Board → Arduino Nano 33 BLE
Upload
```

### 7 — Electrode Placement (AD8232)

```
Standard Lead I (Einthoven):
  RA (Right Arm electrode)  →  AD8232 IN+  (right wrist)
  LA (Left Arm electrode)   →  AD8232 IN-  (left wrist)
  RL (Right Leg / ground)   →  AD8232 SDN  (right ankle or abdomen)

Alternative (chest placement, better SNR):
  IN+  →  Right clavicle
  IN-  →  Left lower rib (V6 position)
  SDN  →  Right lower rib
```

### 8 — Monitor Output
Open **Serial Monitor** at `115200 baud`:
```
[1] N — Normal  conf=94.3%  HR=72bpm  probs=[0.943 0.021 0.018 0.012 0.006]
[2] N — Normal  conf=91.7%  HR=73bpm  probs=[0.917 0.031 0.024 0.020 0.008]
```

---

## Preprocessing Pipeline (Arduino ↔ Python Parity)

| Step | Python (training) | Arduino (deployment) |
|------|-------------------|----------------------|
| Sampling rate | 360 Hz (MIT-BIH) | 360 Hz (hardware timer ISR) |
| DC removal | High-pass 0.5 Hz | IIR high-pass (HP_COEFF=0.9690) |
| Bandpass | `butter(2,[0.5,40],fs=360)` | 2×SOS biquad (same coefficients) |
| R-peak detect | Pan-Tompkins | Pan-Tompkins (same MWI=30 samples) |
| Beat window | ±180 samples from R-peak | ±180 samples from R-peak |
| Normalisation | `(x-mean)/std` per beat | `(x-mean)/std` per beat |
| Model input | `(1, 360, 1)` float32 | `(1, 360, 1)` float32 |

---

## Memory Budget (Arduino Nano 33 BLE)

| Resource | Total | Used | Remaining |
|----------|-------|------|-----------|
| Flash | 1024 KB | ~380 KB* | 644 KB |
| RAM | 256 KB | ~52 KB | 204 KB |

*Sketch (~8KB) + TFLite runtime (~350KB) + model (23.4KB)

**Tensor arena:** 24 KB (configured in CardioEdge.ino)  
**ECG ring buffer:** 6 KB (1440 × int16)  
**PPG buffers:** 3 KB  
**Beat window:** 1.4 KB (360 × float32)

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `AllocateTensors FAILED` | Arena too small | Increase `kTensorArenaSize` |
| `Lead-off detected` | Loose electrode | Reapply gel, press firmly |
| Low confidence (<50%) | Noise / motion | Keep arm still, use gel |
| No R-peaks detected | Threshold too high | Reduce `PT_THRESH_INIT` in ecg_preprocess.h |
| MAX30102 FAILED | Wrong I2C | Check SDA/SCL and 3.3V |
| Only N predicted | Bad electrode contact | Try chest placement |

---

## Serial Commands (send from Serial Monitor)

| Command | Action |
|---------|--------|
| `r` + Enter | Reset ECG engine (re-init filters) |
| `d` + Enter | Toggle debug raw-ADC streaming (for plotter) |
| `s` + Enter | Print current stats (beats, alerts, HR) |
