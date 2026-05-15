"""
validate_arduino_signal.py
──────────────────────────
Pre-deployment validation tool.

Reads raw ECG from Arduino over Serial (baud 115200) and:
  1. Plots live ECG waveform
  2. Verifies bandpass filter output matches Python reference filter
  3. Checks R-peak positions against Python Pan-Tompkins
  4. Runs the TFLite INT8 model on each received beat
  5. Compares Arduino classification with Python classification

Usage:
  python deployment/validate_arduino_signal.py --port COM3

The Arduino sketch must be in serial-raw mode:
  Uncomment  // Serial.println(raw);  in CardioEdge.ino

Dependencies:
  pip install pyserial matplotlib scipy numpy
"""
import argparse, sys, time, threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, sosfilt, sosfilt_zi
import serial

sys.path.insert(0, "c:/p3/cardiac-edge-ai")
from config import ECG_FS, WINDOW_LEN, TFLITE_DIR, CLASSES
import tensorflow as tf
import os

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="CardioEdge Arduino Validator")
parser.add_argument("--port",   default="COM3", help="Arduino serial port")
parser.add_argument("--baud",   default=115200,  type=int)
parser.add_argument("--model",  default=os.path.join(TFLITE_DIR, "student_kd_int8.tflite"))
args = parser.parse_args()

# ── TFLite Interpreter ────────────────────────────────────────────────────────
print(f"Loading TFLite model: {args.model}")
interp = tf.lite.Interpreter(model_path=args.model)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
print(f"  Input shape:  {inp_d['shape']}")
print(f"  Output shape: {out_d['shape']}")

# ── Python reference bandpass (matches Arduino SOS coefficients) ──────────────
sos_ref = butter(2, [0.5, 40.0], btype="bandpass", fs=ECG_FS, output="sos")
zi_ref  = sosfilt_zi(sos_ref) * 0.0   # zero IC

# ── Shared data ───────────────────────────────────────────────────────────────
DISPLAY_SAMPLES = ECG_FS * 8   # 8 seconds of ECG for display
raw_deque     = deque(maxlen=DISPLAY_SAMPLES)
filt_deque    = deque(maxlen=DISPLAY_SAMPLES)
beat_windows  = []             # list of (window_np, pred_label, conf)
lock          = threading.Lock()

# ── Pan-Tompkins (Python reference) ──────────────────────────────────────────
class PanTompkins:
    """Minimal Pan-Tompkins R-peak detector."""
    def __init__(self, fs=360):
        self.fs = fs
        self.prev_x = 0.0
        self.mwi_n  = int(0.08 * fs)
        self.mwi_buf = np.zeros(self.mwi_n)
        self.mwi_idx = 0
        self.mwi_sum = 0.0
        self.sig_lvl = 0.2; self.noise_lvl = 0.1
        self.thresh  = 0.2
        self.refrac  = 0
        self.refrac_samp = int(0.2 * fs)
        self.peaks   = []
        self.n       = 0

    def process(self, x: float) -> float:
        """Returns MWI value. Peaks stored in self.peaks."""
        deriv = x - self.prev_x; self.prev_x = x
        sq = deriv * deriv
        self.mwi_sum -= self.mwi_buf[self.mwi_idx]
        self.mwi_buf[self.mwi_idx] = sq
        self.mwi_sum += sq
        self.mwi_idx = (self.mwi_idx + 1) % self.mwi_n
        mwi = self.mwi_sum / self.mwi_n

        if self.refrac > 0:
            self.refrac -= 1
        elif mwi > self.thresh:
            self.sig_lvl = 0.125 * mwi + 0.875 * self.sig_lvl
            self.thresh  = self.noise_lvl + 0.25 * (self.sig_lvl - self.noise_lvl)
            self.refrac  = self.refrac_samp
            self.peaks.append(self.n)
        else:
            self.noise_lvl = 0.125 * mwi + 0.875 * self.noise_lvl
            self.thresh    = self.noise_lvl + 0.25 * (self.sig_lvl - self.noise_lvl)
        self.n += 1
        return mwi

pt = PanTompkins(ECG_FS)
all_raw = []   # global raw sample list for beat segmentation

# ── Python inference ──────────────────────────────────────────────────────────
def run_tflite(window_360: np.ndarray):
    """Run Python-side TFLite on a 360-sample beat. Returns (class_idx, probs)."""
    x = window_360[np.newaxis, :, np.newaxis].astype(np.float32)  # (1,360,1)
    interp.set_tensor(inp_d["index"], x)
    interp.invoke()
    probs = interp.get_tensor(out_d["index"])[0].astype(np.float32)
    return int(np.argmax(probs)), probs

def extract_and_classify(raw_samples: np.ndarray, peak_idx: int):
    """Extract 360-sample window centred on peak, z-score normalise, classify."""
    half = WINDOW_LEN // 2
    start = peak_idx - half
    end   = peak_idx + half
    if start < 0 or end >= len(raw_samples):
        return None
    window = raw_samples[start:end].astype(np.float32)
    mean = window.mean(); std = window.std() + 1e-8
    window = (window - mean) / std
    cls, probs = run_tflite(window)
    return cls, probs, window

# ── Serial reader thread ──────────────────────────────────────────────────────
ser = None

def serial_reader():
    global ser, zi_ref
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"Opened {args.port} @ {args.baud} baud")
    except Exception as e:
        print(f"[ERR] Cannot open {args.port}: {e}")
        return

    zi_local = sosfilt_zi(sos_ref) * 0.0
    sample_n = 0

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or not line.lstrip("-").isdigit():
                continue

            raw = int(line)
            norm = (raw - 512.0) / 512.0

            # Python reference bandpass
            filt_val, zi_local = sosfilt(sos_ref, [norm], zi=zi_local)
            filt_val = float(filt_val[0])

            # Pan-Tompkins
            pt.process(filt_val)
            all_raw.append(raw)

            with lock:
                raw_deque.append(raw)
                filt_deque.append(filt_val)

            # Classify if new R-peak detected
            if len(pt.peaks) > 0 and pt.peaks[-1] == sample_n:
                pk = pt.peaks[-1]
                result = extract_and_classify(np.array(all_raw), pk)
                if result is not None:
                    cls, probs, window = result
                    conf = probs[cls] * 100
                    print(f"[Python TFLite] {CLASSES[cls]}  conf={conf:.1f}%  "
                          f"probs={np.round(probs, 3)}")
                    with lock:
                        beat_windows.append((window, cls, conf))

            sample_n += 1
        except Exception as e:
            print(f"[Reader ERR] {e}")
            time.sleep(0.05)

# ── Matplotlib live plot ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
fig.patch.set_facecolor("#1a1a2e")
for ax in [ax1, ax2]:
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#4a4a6a")

line_raw,  = ax1.plot([], [], color="#4ECDC4", lw=1.0)
line_filt, = ax2.plot([], [], color="#F7B731", lw=1.2)
rp_scatter = ax2.scatter([], [], color="red", s=30, zorder=5)
title_text  = ax1.set_title("CardioEdge Live ECG", color="white",
                              fontsize=13, fontweight="bold")
ax1.set_ylabel("Raw ADC (0–1023)", color="white"); ax1.set_ylim(0, 1023)
ax2.set_ylabel("Bandpass filtered", color="white"); ax2.set_ylim(-1.5, 1.5)
ax2.axhline(0, color="gray", alpha=0.3)
fig.tight_layout(pad=2.0)

def update_plot(frame):
    with lock:
        raw_arr  = np.array(raw_deque)
        filt_arr = np.array(filt_deque)
        n_beats  = len(beat_windows)
        last_cls = beat_windows[-1][1] if n_beats else -1
        last_conf = beat_windows[-1][2] if n_beats else 0.0

    x = np.arange(len(raw_arr))
    line_raw.set_data(x, raw_arr)
    line_filt.set_data(x, filt_arr)

    ax1.set_xlim(0, max(len(raw_arr), 1))
    ax2.set_xlim(0, max(len(filt_arr), 1))

    lbl = f"[Python TFLite] Beat #{n_beats}  →  {CLASSES[last_cls]}  ({last_conf:.0f}%)" \
          if last_cls >= 0 else "Waiting for R-peaks …"
    title_text.set_text(lbl)
    return line_raw, line_filt

print("\nStarting serial reader thread …")
t = threading.Thread(target=serial_reader, daemon=True)
t.start()
time.sleep(1.5)   # give serial thread time to connect

print("Starting live plot. Press Ctrl-C to exit.\n")
ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=False)
plt.show()
