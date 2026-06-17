"""
Quick script to verify your training data setup.
Checks how many image-mask pairs you have for training.
"""

import os

IMAGE_DIR = "images"
MASK_DIR = "masks_binary"

# Check both directories
if not os.path.exists(IMAGE_DIR):
    print(f"ERROR: Image directory '{IMAGE_DIR}' not found!")
    exit(1)

if not os.path.exists(MASK_DIR):
    print(f"ERROR: Mask directory '{MASK_DIR}' not found!")
    exit(1)

# Get all images
image_files = set([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
])

# Get all masks
mask_files = set([
    f for f in os.listdir(MASK_DIR)
    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
])

# Find matching pairs
matching_pairs = image_files.intersection(mask_files)

print("Data Summary:")
print(f"   Images: {len(image_files)}")
print(f"   Masks: {len(mask_files)}")
print(f"   Matching pairs: {len(matching_pairs)}")

if len(matching_pairs) == 0:
    print("\nWARNING: No matching pairs found!")
    print("\nPossible issues:")
    print("   1. Filenames don't match between images/ and masks_binary/")
    print("   2. File extensions differ (e.g., .jpg vs .png)")
    print("\nTip: Make sure mask filenames exactly match image filenames")
    
    # Show some examples
    if image_files:
        print(f"\nExample image files:")
        for f in sorted(list(image_files))[:5]:
            print(f"   {f}")
    if mask_files:
        print(f"\nExample mask files:")
        for f in sorted(list(mask_files))[:5]:
            print(f"   {f}")
else:
    print(f"\nSUCCESS: You have {len(matching_pairs)} image-mask pairs ready for training!")
    
    if len(matching_pairs) < 10:
        print("WARNING: You have very few training samples.")
        print("   Consider creating more masks or using data augmentation.")
    elif len(matching_pairs) < 50:
        print("Tip: You have a small dataset. Data augmentation will be important.")
    else:
        print("Good dataset size for training!")
    
    # Check if there are other mask directories
    if os.path.exists("masks_binary2"):
        mask_files2 = set([
            f for f in os.listdir("masks_binary2")
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        matching_pairs2 = image_files.intersection(mask_files2)
        print(f"\nmasks_binary2: {len(mask_files2)} masks, {len(matching_pairs2)} matching pairs")
        
        if len(matching_pairs2) > len(matching_pairs):
            print(f"Tip: masks_binary2 has more matching pairs! Consider using that directory.")

print("\n" + "="*50)
print("Ready to train? Run: python train_valve_features.py")
