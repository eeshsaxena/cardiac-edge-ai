"""
gen_mit_playback.py — generate mit_playback.h for filter_verify sketch.
Run once: python deployment/arduino_sketch/gen_mit_playback.py
"""
import sys, os
sys.path.insert(0, "c:/p3/cardiac-edge-ai")
import numpy as np
from config import PROCESSED_DIR

data = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
X = data["X_ecg"]; y = data["y"]

# One beat from each class (360 samples each)
beats = []
class_names = ["N", "AF", "VT", "PVC", "LBBB"]
for cls in range(5):
    idx = np.where(y == cls)[0]
    if len(idx) > 0:
        beats.append(X[idx[0]])
        print(f"  Class {class_names[cls]}: record index {idx[0]}")

combined = np.concatenate(beats)   # 1800 samples
lo, hi   = combined.min(), combined.max()
adc      = ((combined - lo) / (hi - lo + 1e-8) * 900 + 61).astype(int)
adc      = np.clip(adc, 0, 1023)

out = ["// Auto-generated: 5 ECG beats (N, AF, VT, PVC, LBBB) from MIT-BIH test set"]
out.append("// 360 samples per beat, ADC-scaled 0-1023, 360 Hz")
out.append(f"static const int16_t MIT_PLAYBACK[] PROGMEM = {{")
vals = [str(v) for v in adc]
for i in range(0, len(vals), 16):
    out.append("  " + ",".join(vals[i:i+16]) + ",")
out.append("};")
out.append(f"static const int N_PLAYBACK = {len(adc)};")
out.append("// Beat boundaries: 0, 360, 720, 1080, 1440")
out.append("// Classes:         N   AF   VT   PVC  LBBB")

h_path = os.path.join(os.path.dirname(__file__), "filter_verify", "mit_playback.h")
with open(h_path, "w") as f:
    f.write("\n".join(out))
print(f"Written {len(adc)} samples -> {h_path}")

# Also verify with Python TFLite for ground truth
import tensorflow as tf
from config import TFLITE_DIR

tflite_path = os.path.join(TFLITE_DIR, "student_kd_int8.tflite")
if os.path.exists(tflite_path):
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    print("\nPython TFLite ground truth for each beat:")
    for i, cls in enumerate(range(5)):
        window = beats[i].astype(np.float32)
        mean = window.mean(); std = window.std() + 1e-8
        window_n = (window - mean) / std
        x = window_n[np.newaxis, :, np.newaxis]
        interp.set_tensor(inp_d["index"], x)
        interp.invoke()
        probs = interp.get_tensor(out_d["index"])[0]
        pred = int(np.argmax(probs))
        print(f"  Beat {i} (true={class_names[cls]}): pred={class_names[pred]} "
              f"conf={probs[pred]*100:.1f}%")

if __name__ == "__main__":
    pass
