"""
run_all_training.py
--------------------
Master script: runs the ENTIRE training pipeline in order.

Usage:
    python run_all_training.py

What it runs (in order):
  1. src/data/split.py      — create stratified splits
  2. src/data/eda.py        — EDA plots
  3. src/models/train_baseline.py   — M2: MLP baseline
  4. src/models/train_cnn.py        — M3: Custom CNN + ablations
  5. src/models/train_transfer.py   — M4: MobileNetV2 transfer learning
  6. src/models/train_unet.py       — M5: U-Net segmentation
  7. src/explainability/run_gradcam.py — M6: Grad-CAM gallery
  8. src/models/train_siamese.py    — M7: Siamese network
  9. src/models/train_autoencoder.py — M8: Autoencoder anomaly detection

Each step is timed and any failure is reported without crashing the pipeline.
Final summary table is printed at the end.
"""

import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STEPS = [
    ("M1: Split",          "src/data/split.py"),
    ("M1: EDA",            "src/data/eda.py"),
    ("M2: Baseline",       "src/models/train_baseline.py"),
    ("M3: Custom CNN",     "src/models/train_cnn.py"),
    ("M4: Transfer",       "src/models/train_transfer.py"),
    ("M5: U-Net",          "src/models/train_unet.py"),
    ("M6: Grad-CAM",       "src/explainability/run_gradcam.py"),
    ("M7: Siamese",        "src/models/train_siamese.py"),
    ("M8: Autoencoder",    "src/models/train_autoencoder.py"),
]

results = []

print("\n" + "="*65)
print("  PotholeIQ — Full Training Pipeline")
print("="*65 + "\n")

for name, script in STEPS:
    script_path = ROOT / script
    print(f"\n{'─'*65}")
    print(f"  RUNNING: {name}  ({script})")
    print(f"{'─'*65}")

    if not script_path.exists():
        print(f"  ⚠ Script not found: {script_path}")
        results.append((name, "SKIPPED", 0))
        continue

    start = time.time()
    ret = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - start

    if ret.returncode == 0:
        status = "✅ OK"
    else:
        status = f"❌ FAILED (exit {ret.returncode})"

    results.append((name, status, elapsed))
    print(f"\n  {status} in {elapsed:.1f}s")

# Summary table
print("\n\n" + "="*65)
print("  PIPELINE SUMMARY")
print("="*65)
print(f"  {'Step':<30} {'Status':<20} {'Time':>8}")
print(f"  {'─'*30} {'─'*20} {'─'*8}")
for name, status, t in results:
    print(f"  {name:<30} {status:<20} {t:>7.1f}s")
print("="*65 + "\n")

failed = [r for r in results if "FAILED" in r[1]]
if failed:
    print(f"  ⚠ {len(failed)} step(s) failed. Check output above.")
    sys.exit(1)
else:
    print("  ✅ All steps completed successfully.")
    print("  Run:  streamlit run app/app.py\n")
