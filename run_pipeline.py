"""
run_pipeline.py — Complete CardioEdge software pipeline.

Usage:
  python run_pipeline.py                    # full pipeline from scratch
  python run_pipeline.py --from eval        # start from evaluation step
  python run_pipeline.py --skip data ptbxl  # skip data download steps
  python run_pipeline.py --only tflite      # run one step only
  python run_pipeline.py --list             # list all steps
"""
import subprocess, sys, os, argparse, time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY   = sys.executable

STEPS = [
    # (id,          script,                                    description)
    ("data",       "data/download_mitbih.py",                 "Download MIT-BIH Arrhythmia dataset"),
    ("ptbxl",      "data/download_ptbxl.py",                  "Download PTB-XL + merge with MIT-BIH"),
    ("balance",    "data/balance_classes.py",                  "SMOTE balancing + synthetic PPG + train/val/test split"),
    ("teacher",    "training/train_teacher.py",                "Train teacher (CNN-BiLSTM, ~60 epochs)"),
    ("precompute", "training/precompute_teacher.py",           "Pre-compute teacher soft labels for fast KD"),
    ("kd",         "training/train_student_torch.py",          "Knowledge distillation — 3 variants (GPU, PyTorch)"),
    ("fusion",     "training/train_fusion.py",                 "Train ECG+PPG late-fusion model"),
    ("eval",       "training/evaluate.py",                     "Full evaluation — generate Tables 1-4 + confusion matrices"),
    ("tflite",     "deployment/convert_tflite_folded.py",      "BN-fold + TFLite INT8 quantization"),
    ("validate",   "deployment/validate_quantized.py",         "Validate INT8 accuracy (Table 3)"),
    ("coeff",      "deployment/verify_filter_coefficients.py", "Verify Arduino biquad coefficients vs scipy"),
    ("report",     "experiments/generate_paper_report.py",     "Generate HTML paper report with all tables + figures"),
]

STEP_IDS = [s[0] for s in STEPS]


def banner(text, width=65, char="─"):
    pad = max(0, width - len(text) - 4)
    return f"\n┌{'─'*width}┐\n│  {text}{' '*pad}│\n└{'─'*width}┘"


def run_step(step_id, script, description, env=None):
    print(banner(f"STEP [{step_id}]  {description}"))
    print(f"  Script: {script}\n")
    t0 = time.time()
    result = subprocess.run(
        [PY, os.path.join(ROOT, script)],
        cwd=ROOT,
        env={**os.environ, **(env or {})}
    )
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    if result.returncode != 0:
        print(f"\n  ✗  Step '{step_id}' FAILED after {elapsed_str} (exit {result.returncode})")
        print(f"     Fix the error above, then resume with:")
        print(f"       python run_pipeline.py --from {step_id}")
        return False
    print(f"\n  ✓  Step '{step_id}' completed in {elapsed_str}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="CardioEdge — full software pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--from",  dest="from_step", default=None,
                        choices=STEP_IDS, metavar="STEP",
                        help="Start pipeline from this step")
    parser.add_argument("--skip",  dest="skip_steps", nargs="+", default=[],
                        choices=STEP_IDS, metavar="STEP",
                        help="Skip these steps")
    parser.add_argument("--only",  dest="only_step", default=None,
                        choices=STEP_IDS, metavar="STEP",
                        help="Run only this one step")
    parser.add_argument("--list",  action="store_true",
                        help="List all pipeline steps and exit")
    args = parser.parse_args()

    if args.list:
        print("\nCardioEdge Pipeline Steps:\n")
        for i, (sid, script, desc) in enumerate(STEPS, 1):
            print(f"  {i:2}. {sid:<12}  {desc}")
            print(f"       {script}")
        print()
        return

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         CardioEdge — Full Software Pipeline                  ║
║  5-class Arrhythmia Detection · TinyML · Arduino Nano 33 BLE ║
╚══════════════════════════════════════════════════════════════╝
  Steps: {" → ".join(STEP_IDS)}
""")

    started = (args.from_step is None)
    done, failed, skipped = [], [], []
    t_total = time.time()

    for sid, script, desc in STEPS:
        # --only mode
        if args.only_step and sid != args.only_step:
            continue
        # --from mode
        if not started:
            if sid == args.from_step:
                started = True
            else:
                skipped.append(sid); continue
        # --skip mode
        if sid in args.skip_steps:
            print(f"\n  [SKIP] {sid}")
            skipped.append(sid); continue

        ok = run_step(sid, script, desc)
        if ok:
            done.append(sid)
        else:
            failed.append(sid)
            break

    total_mins, total_secs = divmod(int(time.time() - t_total), 60)
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   PIPELINE SUMMARY                           ║
╚══════════════════════════════════════════════════════════════╝
  Total time : {total_mins}m {total_secs}s
  Completed  : {done}
  Skipped    : {skipped}
  Failed     : {failed}
""")
    if not failed:
        print("  ✅ All steps complete!\n")
        print("  Key outputs:")
        print("    experiments/logs/          ← Tables 1-4 (CSV)")
        print("    experiments/figures/       ← All paper figures (PNG)")
        print("    experiments/paper_report.html ← Full HTML report")
        print("    deployment/tflite/         ← TFLite + Arduino headers")
        print("    deployment/arduino_sketch/ ← Upload CardioEdge/ to Arduino IDE")
        print()
        print("  Next steps (hardware):")
        print("    1. Upload filter_verify.ino → verify coefficients")
        print("    2. Upload CardioEdge.ino   → connect electrodes")
        print("    3. Run: python deployment/dashboard.py --port COM3")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
