"""
Segmentation validation: Dice, IoU, precision, recall on held-out manual frames
and/or the training validation split (seed 42).
"""

import argparse
import csv
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, random_split

from geometry_utils import dice_coefficient, iou_coefficient, pixel_accuracy
from pipeline_config import DEFAULT_MODEL, VALIDATION_SEGMENTATION_OUT
from model import UNetEdgeDetector
from train_valve_features import ValveFeatureDataset, IMG_SIZE, VALIDATION_SPLIT


def load_model(model_path, device):
    model = UNetEdgeDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_mask(model, image_path, device, img_size=(256, 256), threshold=0.5):
    transform = T.Compose([T.Resize(img_size), T.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    binary = (prob > threshold).astype(np.uint8)
    binary_full = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return binary_full, prob


def load_gt_mask(path, dilate=0):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    binary = (mask > 0).astype(np.uint8)
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


def load_gt_from_annotation(path, dilate=3):
    """Convert hand-drawn white-line annotation to binary mask."""
    img = cv2.imread(path)
    if img is None:
        return None
    white = cv2.inRange(img, np.array([200, 200, 200]), np.array([255, 255, 255]))
    binary = (white > 0).astype(np.uint8)
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


def precision_recall(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return float(precision), float(recall), float(f1)


def evaluate_image_pairs(model, pairs, device, threshold=0.5, out_dir=None, dilate_gt=0, from_annotation=False):
    rows = []
    for i, (fname, img_path, gt_path) in enumerate(pairs):
        if from_annotation:
            gt = load_gt_from_annotation(gt_path, dilate=dilate_gt)
        else:
            gt = load_gt_mask(gt_path, dilate=dilate_gt)
        if gt is None:
            continue
        pred, _ = predict_mask(model, img_path, device, threshold=threshold)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        dice = dice_coefficient(pred, gt)
        iou = iou_coefficient(pred, gt)
        acc = pixel_accuracy(pred, gt)
        prec, rec, f1 = precision_recall(pred, gt)

        rows.append({
            "filename": fname,
            "dice": dice,
            "iou": iou,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

        if out_dir and i < 6:
            overlay = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            panel = np.zeros((*gt.shape, 3), dtype=np.uint8)
            panel[gt > 0] = [0, 200, 0]
            panel[pred > 0] = [255, 80, 80]
            panel[np.logical_and(gt > 0, pred > 0)] = [255, 255, 0]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(overlay)
            axes[0].set_title("Input")
            axes[1].imshow(panel)
            axes[1].set_title("GT(green) / Pred(red) / Overlap(yellow)")
            axes[2].imshow(pred, cmap="gray")
            axes[2].set_title(f"Dice={dice:.3f}")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(fname)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"overlay_{fname.replace('.jpg','.png').replace('.jpeg','.png')}"), dpi=120)
            plt.close(fig)

    return rows


def evaluate_val_split(model, image_dir, mask_dir, device, threshold=0.5):
    dataset = ValveFeatureDataset(image_dir, mask_dir, augment=False)
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.augment = False
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    dice_list, iou_list = [], []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            prob = torch.sigmoid(logits)
            pred = (prob > threshold).float()
            for p, g in zip(pred, masks):
                p_np = p.squeeze().cpu().numpy().astype(bool)
                g_np = g.squeeze().cpu().numpy().astype(bool)
                dice_list.append(dice_coefficient(p_np, g_np))
                iou_list.append(iou_coefficient(p_np, g_np))

    return {
        "num_frames": len(dice_list),
        "mean_dice": float(np.mean(dice_list)),
        "std_dice": float(np.std(dice_list)),
        "mean_iou": float(np.mean(iou_list)),
        "std_iou": float(np.std(iou_list)),
        "per_frame_dice": dice_list,
        "per_frame_iou": iou_list,
    }


def summarize_rows(rows):
    if not rows:
        return {}
    keys = ["dice", "iou", "accuracy", "precision", "recall", "f1"]
    summary = {"num_frames": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows]
        summary[f"mean_{k}"] = float(np.mean(vals))
        summary[f"std_{k}"] = float(np.std(vals))
        summary[f"median_{k}"] = float(np.median(vals))
    return summary


def plot_metric_histogram(rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    dice_vals = [r["dice"] for r in rows]
    iou_vals = [r["iou"] for r in rows]
    axes[0].hist(dice_vals, bins=min(12, max(3, len(dice_vals))), color="#2dd4bf", edgecolor="white")
    axes[0].axvline(np.mean(dice_vals), color="red", linestyle="--", label=f"mean={np.mean(dice_vals):.3f}")
    axes[0].set_title("Dice Distribution (Manual GT)")
    axes[0].legend()
    axes[1].hist(iou_vals, bins=min(12, max(3, len(iou_vals))), color="#38bdf8", edgecolor="white")
    axes[1].axvline(np.mean(iou_vals), color="red", linestyle="--", label=f"mean={np.mean(iou_vals):.3f}")
    axes[1].set_title("IoU Distribution (Manual GT)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def collect_manual_pairs(image_dir, gt_dir):
    pairs = []
    if not os.path.isdir(gt_dir):
        return pairs
    for fname in sorted(os.listdir(gt_dir)):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(image_dir, fname)
        gt_path = os.path.join(gt_dir, fname)
        if os.path.isfile(img_path):
            pairs.append((fname, img_path, gt_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Validate segmentation against manual ground truth.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--images", default="images")
    parser.add_argument("--gt", default="masks_binary2", help="Ground-truth masks for expert frames")
    parser.add_argument("--annotated-dir", default="masks_annotated", help="Raw hand annotations")
    parser.add_argument("--dilate-gt", type=int, default=0, help="Dilate GT mask (px) for thin edge tolerance")
    parser.add_argument("--dilate-annotated", type=int, default=5, help="Dilate when loading raw annotations")
    parser.add_argument("--full-mask-dir", default="masks_binary2", help="For val-split evaluation")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default=VALIDATION_SEGMENTATION_OUT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    overlay_dir = os.path.join(args.out, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    print(f"Device: {device}")
    model = load_model(args.model, device)

    manual_pairs = collect_manual_pairs(args.images, args.gt)
    expert_names = set(os.listdir(args.annotated_dir))
    expert_pairs = [p for p in manual_pairs if p[0] in expert_names]
    print(f"Expert-annotated subset: {len(expert_pairs)} frames")

    expert_rows = evaluate_image_pairs(
        model, expert_pairs, device, threshold=args.threshold,
        out_dir=overlay_dir, dilate_gt=args.dilate_gt,
    )
    expert_summary = summarize_rows(expert_rows)

    raw_pairs = collect_manual_pairs(args.images, args.annotated_dir)
    raw_rows = evaluate_image_pairs(
        model, raw_pairs, device, threshold=args.threshold,
        dilate_gt=args.dilate_annotated, from_annotation=True,
    )
    raw_summary = summarize_rows(raw_rows)

    val_split_summary = None
    if os.path.isdir(args.full_mask_dir):
        print("Evaluating training validation split (seed 42)...")
        val_split_summary = evaluate_val_split(
            model, args.images, args.full_mask_dir, device, threshold=args.threshold
        )

    csv_path = os.path.join(args.out, "per_frame_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "dice", "iou", "accuracy", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(expert_rows)

    if expert_rows:
        plot_metric_histogram(expert_rows, os.path.join(args.out, "segmentation_histograms.png"))

    summary = {
        "model_path": args.model,
        "threshold": args.threshold,
        "expert_annotated_frames": len(expert_pairs),
        "expert_validation_masks_binary2": expert_summary,
        "raw_annotation_validation_dilated": {
            **raw_summary,
            "dilate_px": args.dilate_annotated,
            "source": args.annotated_dir,
        },
        "validation_split_20pct": val_split_summary,
        "note": (
            "Expert validation uses the 32 hand-annotated frames compared against "
            "masks_binary2 ground truth (same format as training). "
            "Raw annotation validation dilates thin white-line tracings for boundary tolerance."
        ),
    }
    summary_path = os.path.join(args.out, "segmentation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Expert Frame Validation (masks_binary2 GT) ===")
    for k, v in expert_summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n=== Raw Annotation Validation (dilated white lines) ===")
    for k, v in raw_summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    if val_split_summary:
        print("\n=== 20% Validation Split ===")
        print(f"  mean_dice: {val_split_summary['mean_dice']:.4f} ± {val_split_summary['std_dice']:.4f}")
        print(f"  mean_iou:  {val_split_summary['mean_iou']:.4f} ± {val_split_summary['std_iou']:.4f}")
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
