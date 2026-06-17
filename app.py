#!/usr/bin/env python3
"""CineValve — unified CLI for venous valve ultrasound analysis."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cinevalve.config import DEFAULT_MODEL, DEFAULT_THRESHOLD, DEFAULT_VIDEO  # noqa: E402

SCRIPTS = {
    "extract": ROOT / "scripts" / "data" / "extractframes.py",
    "prepare-masks": ROOT / "scripts" / "data" / "binarymasks2.py",
    "train": ROOT / "scripts" / "train" / "train_venous.py",
    "predict": ROOT / "scripts" / "infer" / "predict_masks.py",
    "motion": ROOT / "scripts" / "infer" / "valve_motion_analysis.py",
    "validate-seg": ROOT / "scripts" / "validate" / "validate_segmentation.py",
    "validate-geom": ROOT / "scripts" / "validate" / "validate_geometry.py",
    "carotid": ROOT / "carotid" / "scripts" / "run_carotid.py",
    "cfd": ROOT / "scripts" / "optional" / "cfd_preliminary.py",
    "check": ROOT / "scripts" / "utils" / "check_dataset_safety.py",
    "full": ROOT / "run_all.py",
}


def run_script(script: Path, extra: list[str] | None = None) -> int:
    if not script.is_file():
        raise SystemExit(f"Script not found: {script}")
    cmd = [sys.executable, str(script)] + (extra or [])
    print(">", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


def cmd_full(args: argparse.Namespace) -> int:
    extra = []
    if args.threshold is not None:
        extra.extend(["--threshold", str(args.threshold)])
    if args.skip_predict:
        extra.append("--skip-predict")
    if args.skip_cfd:
        extra.append("--skip-cfd")
    return run_script(SCRIPTS["full"], extra)


def cmd_train(args: argparse.Namespace) -> int:
    extra = []
    if args.epochs:
        extra.extend(["--epochs", str(args.epochs)])
    if args.pretrain:
        extra.extend(["--pretrain", args.pretrain])
    return run_script(SCRIPTS["train"], extra)


def cmd_predict(args: argparse.Namespace) -> int:
    extra = ["--video", args.video, "--model", args.model]
    if args.threshold is not None:
        extra.extend(["--threshold", str(args.threshold)])
    extra.append("--confirm-overwrite-venous")
    return run_script(SCRIPTS["predict"], extra)


def cmd_validate(args: argparse.Namespace) -> int:
    extra = ["--model", args.model]
    if args.threshold is not None:
        extra.extend(["--threshold", str(args.threshold)])
    rc = run_script(SCRIPTS["validate-seg"], extra)
    rc |= run_script(SCRIPTS["validate-geom"], extra)
    return rc


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cinevalve",
        description="Automated venous valve segmentation and analysis from 2D B-mode ultrasound.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_full = sub.add_parser("full", help="Run predict -> motion -> validation -> CFD")
    p_full.add_argument("--threshold", type=float, default=None)
    p_full.add_argument("--skip-predict", action="store_true")
    p_full.add_argument("--skip-cfd", action="store_true")
    p_full.set_defaults(func=cmd_full)

    p_train = sub.add_parser("train", help="Train venous U-Net")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--pretrain", default=None)
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Segment valve in video frames")
    p_pred.add_argument("--video", default=DEFAULT_VIDEO)
    p_pred.add_argument("--model", default=DEFAULT_MODEL)
    p_pred.add_argument("--threshold", type=float, default=None)
    p_pred.set_defaults(func=cmd_predict)

    for name, help_text in [
        ("extract", "Extract frames from ultrasound video"),
        ("prepare-masks", "Build training masks from annotations"),
        ("motion", "Motion metrics from predicted masks"),
        ("check", "Verify venous/carotid path separation"),
        ("carotid", "Carotid train + validate (see carotid/README.md)"),
        ("cfd", "Optional geometry export"),
    ]:
        p = sub.add_parser(name, help=help_text)

        def _bind(cmd=name):
            def _run(_args):
                return run_script(SCRIPTS[cmd])
            return _run

        p.set_defaults(func=_bind())

    p_val = sub.add_parser("validate", help="Segmentation + geometry validation")
    p_val.add_argument("--model", default=DEFAULT_MODEL)
    p_val.add_argument("--threshold", type=float, default=None)
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    code = args.func(args)
    raise SystemExit(code or 0)


if __name__ == "__main__":
    main()
