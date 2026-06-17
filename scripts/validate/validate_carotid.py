"""
Validate trained_carotid_model.pth on the Common Carotid Artery dataset.

Read-only on carotid folders; never touches venous valve data.
"""

import argparse
import csv
import json
import os

from fedsm.safety import CAROTID_IMAGE_DIR, CAROTID_MASK_DIR, CAROTID_MODEL
from validate_segmentation import (
    evaluate_image_pairs,
    evaluate_val_split,
    load_model,
    plot_metric_histogram,
    summarize_rows,
)


def main():
    parser = argparse.ArgumentParser(description="Validate carotid segmentation model.")
    parser.add_argument("--model", default=CAROTID_MODEL)
    parser.add_argument("--images", default=CAROTID_IMAGE_DIR)
    parser.add_argument("--masks", default=CAROTID_MASK_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default="validation_carotid")
    parser.add_argument("--overlay-count", type=int, default=6)
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    overlay_dir = os.path.join(args.out, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Images: {args.images}")
    print(f"Masks: {args.masks}")

    model = load_model(args.model, device)

    pairs = []
    for fname in sorted(os.listdir(args.images)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(args.images, fname)
        mask_path = os.path.join(args.masks, fname)
        if os.path.isfile(mask_path):
            pairs.append((fname, img_path, mask_path))

    print(f"Evaluating {len(pairs)} carotid image-mask pairs...")
    rows = evaluate_image_pairs(
        model,
        pairs,
        device,
        threshold=args.threshold,
        out_dir=overlay_dir,
        dilate_gt=0,
    )
    summary = summarize_rows(rows)

    print("Evaluating 20% holdout split (seed 42, same as training)...")
    holdout = evaluate_val_split(
        model, args.images, args.masks, device, threshold=args.threshold
    )

    csv_path = os.path.join(args.out, "per_frame_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "dice", "iou", "accuracy", "precision", "recall", "f1"]
        )
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        plot_metric_histogram(rows, os.path.join(args.out, "segmentation_histograms.png"))

    result = {
        "model_path": args.model,
        "dataset": "Common Carotid Artery Ultrasound Images",
        "num_frames": len(rows),
        "full_dataset_metrics": summary,
        "holdout_20pct_seed42": holdout,
        "note": "Filled lumen expert masks; holdout uses same random split as train_valve_features.",
    }
    summary_path = os.path.join(args.out, "segmentation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== Full dataset (1100 frames) ===")
    for key in ("mean_dice", "mean_iou", "mean_precision", "mean_recall"):
        print(f"  {key}: {summary[key]:.4f}")
    print("\n=== 20% holdout ===")
    print(f"  mean_dice: {holdout['mean_dice']:.4f} +/- {holdout['std_dice']:.4f}")
    print(f"  mean_iou:  {holdout['mean_iou']:.4f} +/- {holdout['std_iou']:.4f}")
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {csv_path}")
    print(f"Overlays: {overlay_dir}/")


if __name__ == "__main__":
    main()
