"""
Run the full venous valve analysis pipeline with one command:

  predict masks -> motion analysis -> segmentation validation ->
  geometry validation -> CFD geometry export

All steps use the same model checkpoint (trained_valve_model.pth by default).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pipeline_config import (
    CFD_OUTPUT_DIR,
    DEFAULT_FPS,
    DEFAULT_MASKS_DIR,
    DEFAULT_MODEL,
    DEFAULT_PIXEL_TO_MM,
    DEFAULT_THRESHOLD,
    DEFAULT_VIDEO,
    VALVE_METRICS_PREFIX,
    VALIDATION_GEOMETRY_OUT,
    VALIDATION_SEGMENTATION_OUT,
)
from dataset_safety import CAROTID_MODEL, is_protected_venous_dir

ROOT = Path(__file__).resolve().parent


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")
    print(" ", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Step failed ({name}): exit code {result.returncode}")


def read_fps_from_masks(masks_dir: Path, fallback: float) -> float:
    meta_path = masks_dir / "metadata.json"
    if not meta_path.is_file():
        return fallback
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    fps = meta.get("fps")
    if fps and fps > 0:
        return float(fps)
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run predict -> motion -> validation -> CFD with shared settings."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model weights path")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Input ultrasound video")
    parser.add_argument("--masks", default=DEFAULT_MASKS_DIR, help="Predicted masks directory")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--fps", type=float, default=None, help="Override FPS for motion analysis")
    parser.add_argument("--pixel-to-mm", type=float, default=DEFAULT_PIXEL_TO_MM)
    parser.add_argument("--cfd-frames", type=int, default=3)
    parser.add_argument("--skip-predict", action="store_true", help="Reuse existing predicted masks")
    parser.add_argument("--skip-motion", action="store_true")
    parser.add_argument("--skip-segmentation", action="store_true")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--skip-cfd", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")

    if Path(args.model).name == CAROTID_MODEL:
        raise SystemExit(
            f"REFUSED: run_all.py is the venous pipeline; do not use carotid model '{CAROTID_MODEL}'."
        )

    masks_dir = Path(args.masks)

    if not args.skip_predict:
        predict_cmd = [
            py,
            "predict_masks.py",
            "--video",
            args.video,
            "--model",
            str(model_path),
            "--out",
            args.masks,
            "--threshold",
            str(args.threshold),
        ]
        if is_protected_venous_dir(args.masks):
            predict_cmd.append("--confirm-overwrite-venous")
            print("[safety] Venous pipeline: refreshing protected folder", args.masks)
        run_step("1/5 Predict masks", predict_cmd)
    elif not masks_dir.is_dir() or not any(masks_dir.glob("frame_*.png")):
        raise SystemExit(f"--skip-predict set but no masks found in {masks_dir}")

    fps = args.fps if args.fps is not None else read_fps_from_masks(masks_dir, DEFAULT_FPS)
    metrics_prefix = VALVE_METRICS_PREFIX

    if not args.skip_motion:
        run_step(
            "2/5 Valve motion analysis",
            [
                py,
                "valve_motion_analysis.py",
                "--masks",
                args.masks,
                "--fps",
                str(fps),
                "--out",
                metrics_prefix,
            ],
        )

    if not args.skip_segmentation:
        run_step(
            "3/5 Segmentation validation",
            [
                py,
                "validate_segmentation.py",
                "--model",
                str(model_path),
                "--threshold",
                str(args.threshold),
                "--out",
                VALIDATION_SEGMENTATION_OUT,
            ],
        )

    if not args.skip_geometry:
        run_step(
            "4/5 Geometry validation",
            [
                py,
                "validate_geometry.py",
                "--model",
                str(model_path),
                "--threshold",
                str(args.threshold),
                "--out",
                VALIDATION_GEOMETRY_OUT,
            ],
        )

    if not args.skip_cfd:
        run_step(
            "5/5 CFD geometry export",
            [
                py,
                "cfd_preliminary.py",
                "--masks",
                args.masks,
                "--metrics-summary",
                f"{metrics_prefix}_summary.json",
                "--pixel-to-mm",
                str(args.pixel_to_mm),
                "--out",
                CFD_OUTPUT_DIR,
                "--num-frames",
                str(args.cfd_frames),
            ],
        )

    print(f"\n{'=' * 60}")
    print("Pipeline complete")
    print(f"  Model:      {model_path}")
    print(f"  Masks:      {masks_dir}/")
    print(f"  Motion:     {metrics_prefix}.csv, {metrics_prefix}_plots.png")
    print(f"  Seg val:    {VALIDATION_SEGMENTATION_OUT}/")
    print(f"  Geom val:   {VALIDATION_GEOMETRY_OUT}/")
    print(f"  CFD:        {CFD_OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
