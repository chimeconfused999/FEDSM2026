"""
Carotid workflow: train on carotid dataset, then validate.
Does not read or write venous valve folders.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "carotid" / "scripts" / "train_carotid.py"
VALIDATE = ROOT / "carotid" / "scripts" / "validate_carotid.py"


def run_step(name, cmd):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(" ", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Step failed ({name}): exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Train and validate carotid U-Net.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_train:
        train_cmd = [py, str(TRAIN)]
        if args.epochs is not None:
            train_cmd.extend(["--epochs", str(args.epochs)])
        run_step("1/2 Train carotid model", train_cmd)

    if not args.skip_validate:
        run_step("2/2 Validate carotid model", [py, str(VALIDATE)])

    print(f"\n{'=' * 60}")
    print("Carotid workflow complete")
    print("  Model:   carotid/models/trained_carotid_model.pth")
    print("  Metrics: carotid/validation/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
