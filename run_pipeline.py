"""
run_pipeline.py — Run the entire software pipeline end-to-end.
Usage:
  python run_pipeline.py              # full pipeline
  python run_pipeline.py --from eval  # start from evaluation step
  python run_pipeline.py --skip data  # skip data download
"""
import subprocess, sys, os, argparse, time

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("data",     "data/download_mitbih.py",         "Download MIT-BIH"),
    ("ptbxl",    "data/download_ptbxl.py",          "Download PTB-XL + merge"),
    ("balance",  "data/balance_classes.py",          "SMOTE + synthetic PPG + split"),
    ("teacher",  "training/train_teacher.py",        "Train teacher (CNN-BiLSTM)"),
    ("kd",       "training/train_student_kd.py",     "Knowledge distillation (3 variants)"),
    ("fusion",   "training/train_fusion.py",         "Train ECG+PPG fusion model"),
    ("eval",     "training/evaluate.py",             "Evaluate — generate all 4 tables"),
    ("tflite",   "deployment/convert_tflite.py",     "Convert to TFLite INT8"),
    ("validate", "deployment/validate_quantized.py", "Validate quantized accuracy"),
]

STEP_NAMES = [s[0] for s in STEPS]


def run_step(name: str, script: str, description: str) -> bool:
    print(f"\n{'#'*64}")
    print(f"  STEP: {description}")
    print(f"  Script: {script}")
    print(f"{'#'*64}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, script)],
        cwd=ROOT
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAIL] Step '{name}' failed after {elapsed:.0f}s (exit code {result.returncode})")
        return False
    print(f"\n[OK] Step '{name}' completed in {elapsed:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="CardioEdge full pipeline")
    parser.add_argument("--from",  dest="from_step", default=None,
                        choices=STEP_NAMES, help="Start from this step")
    parser.add_argument("--skip",  dest="skip_steps", nargs="+", default=[],
                        choices=STEP_NAMES, help="Skip these steps")
    parser.add_argument("--only",  dest="only_step",  default=None,
                        choices=STEP_NAMES, help="Run only this step")
    args = parser.parse_args()

    print("\n" + "="*64)
    print("  CardioEdge — Full Software Pipeline")
    print("="*64)
    print(f"  Steps: {' → '.join(STEP_NAMES)}")

    started  = args.from_step is None
    failed   = []
    skipped  = []
    done     = []

    for name, script, desc in STEPS:
        # Only mode
        if args.only_step and name != args.only_step:
            continue
        # From mode
        if not started:
            if name == args.from_step:
                started = True
            else:
                skipped.append(name)
                continue
        # Skip mode
        if name in args.skip_steps:
            print(f"\n[SKIP] {name}")
            skipped.append(name)
            continue

        ok = run_step(name, script, desc)
        if ok:
            done.append(name)
        else:
            failed.append(name)
            print("\n[ABORT] Pipeline stopped. Fix the error above and re-run with:")
            print(f"  python run_pipeline.py --from {name}")
            break

    print(f"\n{'='*64}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*64}")
    print(f"  Completed : {done}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")
    if not failed:
        print("\n  ✓ All steps completed successfully!")
        print("  Results in: experiments/logs/ and experiments/figures/")
        print("  TFLite models in: deployment/tflite/")


if __name__ == "__main__":
    main()
