#!/usr/bin/env python3
"""
FEDSM 2026 — unified application for venous valve ultrasound analysis.

Usage:
  python app.py full              # run entire venous pipeline
  python app.py train             # train U-Net on expert masks
  python app.py predict           # segment video frames
  python app.py extract           # video -> images/
  python app.py prepare-masks     # annotations -> masks_binary2/
  python app.py motion            # temporal metrics from masks
  python app.py validate          # segmentation + geometry validation
  python app.py carotid           # optional carotid workflow
  python app.py poster            # regenerate poster figures
  python app.py check             # dataset safety checks
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedsm.config import DEFAULT_MODEL, DEFAULT_THRESHOLD, DEFAULT_VIDEO  # noqa: E402

SCRIPTS = {
    "extract": ROOT / "scripts" / "data" / "extractframes.py",
    "prepare-masks": ROOT / "scripts" / "data" / "binarymasks2.py",
    "train": ROOT / "scripts" / "train" / "train_venous.py",
    "predict": ROOT / "scripts" / "infer" / "predict_masks.py",
    "motion": ROOT / "scripts" / "infer" / "valve_motion_analysis.py",
    "validate-seg": ROOT / "scripts" / "validate" / "validate_segmentation.py",
    "validate-geom": ROOT / "scripts" / "validate" / "validate_geometry.py",
    "carotid": ROOT / "scripts" / "optional" / "run_carotid.py",
    "cfd": ROOT / "scripts" / "optional" / "cfd_preliminary.py",
    "check": ROOT / "scripts" / "utils" / "check_dataset_safety.py",
    "flowchart": ROOT / "scripts" / "poster" / "make_pipeline_flowchart.py",
    "table": ROOT / "scripts" / "poster" / "make_dataset_table.py",
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
    extra.extend(["--confirm-overwrite-venous"])
    return run_script(SCRIPTS["predict"], extra)


def cmd_validate(args: argparse.Namespace) -> int:
    rc = 0
    extra = ["--model", args.model]
    if args.threshold is not None:
        extra.extend(["--threshold", str(args.threshold)])
    rc |= run_script(SCRIPTS["validate-seg"], extra)
    rc |= run_script(SCRIPTS["validate-geom"], extra)
    return rc


def cmd_poster(_: argparse.Namespace) -> int:
    rc = run_script(SCRIPTS["flowchart"])
    rc |= run_script(SCRIPTS["table"])
    return rc


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fedsm",
        description="Automated venous valve analysis from 2D B-mode ultrasound (FEDSM 2026).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_full = sub.add_parser("full", help="Run predict -> motion -> validation -> CFD")
    p_full.add_argument("--threshold", type=float, default=None)
    p_full.add_argument("--skip-predict", action="store_true")
    p_full.add_argument("--skip-cfd", action="store_true")
    p_full.set_defaults(func=cmd_full)

    p_train = sub.add_parser("train", help="Train venous U-Net on masks_binary2/")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--pretrain", default=None, help="Optional carotid checkpoint")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Segment valve in video frames")
    p_pred.add_argument("--video", default=DEFAULT_VIDEO)
    p_pred.add_argument("--model", default=DEFAULT_MODEL)
    p_pred.add_argument("--threshold", type=float, default=None)
    p_pred.set_defaults(func=cmd_predict)

    for name, help_text in [
        ("extract", "Extract frames from ultrasound video"),
        ("prepare-masks", "Build masks_binary2/ from masks_annotated/"),
        ("motion", "Compute valve motion metrics from predicted masks"),
        ("check", "Verify venous/carotid dataset separation"),
        ("carotid", "Train and validate carotid model (optional)"),
        ("cfd", "Export preliminary CFD geometry (optional)"),
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

    p_poster = sub.add_parser("poster", help="Regenerate flowchart and dataset table PNGs")
    p_poster.set_defaults(func=cmd_poster)

    args = parser.parse_args()
    code = args.func(args)
    raise SystemExit(code or 0)


if __name__ == "__main__":
    main()
