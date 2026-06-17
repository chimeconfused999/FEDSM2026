"""
Visualize carotid expert masks (green) on ultrasound images.

No trained model required for --mode expert.
Use --mode validate with trained_carotid_model.pth for green/red/yellow panels.
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from cinevalve.safety import CAROTID_IMAGE_DIR, CAROTID_MASK_DIR, CAROTID_MODEL


def load_us(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 0).astype(np.uint8)


def green_overlay(us_rgb, mask, alpha=0.45):
    """Expert lumen in green over grayscale ultrasound."""
    out = us_rgb.copy().astype(np.float32)
    green = np.zeros_like(out)
    green[:, :, 1] = 255.0
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * green[m]
    return out.astype(np.uint8)


def save_expert_overlay(us_path, mask_path, out_path, alpha=0.45):
    us = load_us(us_path)
    mask = load_mask(mask_path)
    if us is None or mask is None:
        return False
    if mask.shape[:2] != us.shape[:2]:
        mask = cv2.resize(mask, (us.shape[1], us.shape[0]), interpolation=cv2.INTER_NEAREST)
    blended = green_overlay(us, mask, alpha=alpha)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(us)
    axes[0].set_title("Ultrasound")
    axes[1].imshow(blended)
    axes[1].set_title("Expert mask (green)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(os.path.basename(us_path))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def save_validation_overlay(us_path, mask_path, pred, out_path, dice):
    us = load_us(us_path)
    gt = load_mask(mask_path)
    if us is None or gt is None:
        return False
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    panel = np.zeros((*gt.shape, 3), dtype=np.uint8)
    panel[gt > 0] = [0, 200, 0]
    panel[pred > 0] = [255, 80, 80]
    panel[np.logical_and(gt > 0, pred > 0)] = [255, 255, 0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(us)
    axes[0].set_title("Input")
    axes[1].imshow(panel)
    axes[1].set_title("GT(green) / Pred(red) / Overlap(yellow)")
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title(f"Dice={dice:.3f}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(os.path.basename(us_path))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Carotid green overlay visualizations.")
    parser.add_argument("--images", default=CAROTID_IMAGE_DIR)
    parser.add_argument("--masks", default=CAROTID_MASK_DIR)
    parser.add_argument("--out", default="carotid_overlays")
    parser.add_argument("--mode", choices=["expert", "validate"], default="expert",
                        help="expert=green expert mask only; validate=GT vs model prediction")
    parser.add_argument("--model", default=CAROTID_MODEL)
    parser.add_argument("--count", type=int, default=12, help="Number of sample frames")
    parser.add_argument("--alpha", type=float, default=0.45, help="Green blend strength (expert mode)")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    expert_dir = os.path.join(args.out, "expert_green")
    validate_dir = os.path.join(args.out, "validation_panels")
    os.makedirs(expert_dir, exist_ok=True)

    pairs = []
    for fname in sorted(os.listdir(args.images)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(args.images, fname)
        mask_path = os.path.join(args.masks, fname)
        if os.path.isfile(mask_path):
            pairs.append((fname, img_path, mask_path))

    if not pairs:
        raise SystemExit(f"No image-mask pairs in {args.images}")

    # Evenly spaced samples across dataset
    n = min(args.count, len(pairs))
    if n == 1:
        indices = [0]
    else:
        indices = [int(i * (len(pairs) - 1) / (n - 1)) for i in range(n)]

    saved = 0
    for idx in indices:
        fname, img_path, mask_path = pairs[idx]
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(expert_dir, f"green_{stem}.png")
        if save_expert_overlay(img_path, mask_path, out_path, alpha=args.alpha):
            saved += 1
            print(f"Saved {out_path}")

    print(f"\nExpert green overlays: {saved} images in {expert_dir}/")

    if args.mode == "validate":
        import torch
        from validate_segmentation import load_model, predict_mask
        from cinevalve.geometry import dice_coefficient

        if not os.path.isfile(args.model):
            raise SystemExit(f"Model not found for validate mode: {args.model}")

        os.makedirs(validate_dir, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(args.model, device)
        print(f"Validate mode on {device}")

        vsaved = 0
        for idx in indices:
            fname, img_path, mask_path = pairs[idx]
            gt = load_mask(mask_path)
            pred, _ = predict_mask(model, img_path, device, threshold=args.threshold)
            dice = dice_coefficient(pred, gt)
            stem = os.path.splitext(fname)[0]
            out_path = os.path.join(validate_dir, f"overlay_{stem}.png")
            if save_validation_overlay(img_path, mask_path, pred, out_path, dice):
                vsaved += 1
                print(f"Saved {out_path} (Dice={dice:.3f})")
        print(f"\nValidation panels: {vsaved} images in {validate_dir}/")


if __name__ == "__main__":
    main()
