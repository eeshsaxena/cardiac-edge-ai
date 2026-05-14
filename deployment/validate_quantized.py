"""
validate_quantized.py — Compare INT8 TFLite accuracy vs Keras baseline.
Run: python deployment/validate_quantized.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from config import PROCESSED_DIR, MODELS_DIR, TFLITE_DIR, BATCH_SIZE


def load_test():
    data = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
    return data["X_ecg"][..., np.newaxis].astype(np.float32), data["y"].astype(np.int32)


def keras_eval(path, X, y):
    if not os.path.exists(path):
        return None, None
    m = tf.keras.models.load_model(path)
    ds = tf.data.Dataset.from_tensor_slices(X).batch(BATCH_SIZE)
    probs = np.concatenate([m(x, training=False).numpy() for x in ds])
    p = np.argmax(probs, 1)
    return accuracy_score(y, p), f1_score(y, p, average="macro", zero_division=0)


def tflite_eval(path, X, y, is_int8=False):
    if not os.path.exists(path):
        return None, None
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]
    preds = []
    for i in range(len(X)):
        x = X[i:i+1]
        if is_int8:
            sc, zp = inp_d["quantization"]
            x = (x / sc + zp).astype(np.int8)
        interp.set_tensor(inp_d["index"], x)
        interp.invoke()
        out = interp.get_tensor(out_d["index"])
        if is_int8:
            sc, zp = out_d["quantization"]
            out = (out.astype(np.float32) - zp) * sc
        preds.append(int(np.argmax(out)))
    preds = np.array(preds)
    return accuracy_score(y, preds), f1_score(y, preds, average="macro", zero_division=0)


def validate():
    X, y = load_test()
    fmt = "{:<22} {:<14} {:<10} {:<10} {:<10}"
    print("\n" + "="*68)
    print(fmt.format("Model", "Format", "Accuracy", "Macro-F1", "F1 Drop"))
    print("-"*68)

    for name, keras_f, fp16_f, int8_f in [
        ("Student Full KD",
         "student_full_kd.keras", "student_kd_fp16.tflite", "student_kd_int8.tflite"),
    ]:
        base_acc, base_f1 = keras_eval(os.path.join(MODELS_DIR, keras_f), X, y)
        if base_acc:
            print(fmt.format(name, "Keras FP32", f"{base_acc*100:.2f}%", f"{base_f1:.4f}", "—"))

        for label, fname, is8 in [
            ("TFLite FP16", fp16_f, False),
            ("TFLite INT8", int8_f, True),
        ]:
            p = os.path.join(TFLITE_DIR, fname)
            acc, f1 = tflite_eval(p, X, y, is_int8=is8)
            if acc:
                drop = f"{(base_f1-f1):.4f}" if base_f1 else "—"
                size = f"{os.path.getsize(p)/1024:.0f} KB"
                print(fmt.format("", f"{label} ({size})", f"{acc*100:.2f}%", f"{f1:.4f}", drop))

    print("="*68)
    print("✓ Done. Use these numbers for Table 3 in the paper.")


if __name__ == "__main__":
    validate()
