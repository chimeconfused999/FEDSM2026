"""Verify image–mask pair counts for venous training."""

import os

IMAGE_DIR = "data/images"
MASK_DIR = "data/masks/training"

if not os.path.isdir(IMAGE_DIR):
    print(f"ERROR: Image directory '{IMAGE_DIR}' not found!")
    raise SystemExit(1)

if not os.path.isdir(MASK_DIR):
    print(f"ERROR: Mask directory '{MASK_DIR}' not found!")
    raise SystemExit(1)

image_files = {f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))}
mask_files = {f for f in os.listdir(MASK_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))}
matching = image_files & mask_files

print("Data summary")
print(f"  Images:           {len(image_files)}")
print(f"  Training masks:   {len(mask_files)}")
print(f"  Matching pairs:   {len(matching)}")

if not matching:
    print("\nWARNING: No matching pairs. Filenames must match between images and masks.")
    raise SystemExit(1)

print(f"\nReady to train: python app.py train")
