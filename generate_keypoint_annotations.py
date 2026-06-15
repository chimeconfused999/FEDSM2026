"""
Helper script to generate keypoint annotations from binary masks.
This extracts valve keypoints (tips and bases) from existing masks
and saves them in JSON format for training keypoint detection models.
"""

import os
import json
import cv2
import numpy as np
from valve_motion_analysis import classify_keypoints, load_binary_mask
from tqdm import tqdm

# ========== Configuration ==========
MASK_DIR = "masks_binary"
IMAGE_DIR = "images"  # Used to match filenames
OUTPUT_JSON = "valve_keypoints.json"
MIN_KEYPOINTS = 2  # Minimum number of keypoints required to save annotation


def generate_annotations(mask_dir, image_dir, output_json, min_keypoints=2):
    """
    Generate keypoint annotations from binary masks.
    
    Args:
        mask_dir: Directory containing binary masks
        image_dir: Directory containing corresponding images (for filename matching)
        output_json: Path to save JSON annotations
        min_keypoints: Minimum keypoints required to save annotation
    """
    annotations = {}
    mask_files = sorted([
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    if not mask_files:
        print(f"❌ No mask files found in {mask_dir}")
        return
    
    print(f"🔍 Processing {len(mask_files)} masks...")
    
    valid_count = 0
    invalid_count = 0
    
    for fname in tqdm(mask_files, desc="Extracting keypoints"):
        mask_path = os.path.join(mask_dir, fname)
        
        # Check if corresponding image exists
        image_path = os.path.join(image_dir, fname)
        if not os.path.exists(image_path):
            # Try to find image with same base name but different extension
            base_name = os.path.splitext(fname)[0]
            found = False
            for ext in ['.jpg', '.png', '.jpeg']:
                alt_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(alt_path):
                    found = True
                    break
            if not found:
                print(f"⚠️  No corresponding image found for {fname}, skipping...")
                invalid_count += 1
                continue
        
        # Load and process mask
        mask = load_binary_mask(mask_path)
        if mask is None:
            invalid_count += 1
            continue
        
        # Extract keypoints
        keypoints = classify_keypoints(mask)
        
        if keypoints is None:
            invalid_count += 1
            continue
        
        # Count valid keypoints
        valid_kp_count = sum(1 for v in keypoints.values() if v is not None)
        
        if valid_kp_count >= min_keypoints:
            # Convert numpy arrays to lists for JSON serialization
            keypoints_serializable = {}
            for k, v in keypoints.items():
                if v is not None:
                    keypoints_serializable[k] = [float(v[0]), float(v[1])]
                else:
                    keypoints_serializable[k] = None
            
            annotations[fname] = keypoints_serializable
            valid_count += 1
        else:
            invalid_count += 1
    
    # Save annotations
    if annotations:
        with open(output_json, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"\n✅ Generated {valid_count} valid annotations")
        print(f"❌ Skipped {invalid_count} invalid/missing annotations")
        print(f"💾 Saved to {output_json}")
        
        # Print statistics
        print("\n📊 Keypoint Statistics:")
        kp_counts = {'tip_left': 0, 'tip_right': 0, 'base_left': 0, 'base_right': 0}
        for ann in annotations.values():
            for kp_name in kp_counts.keys():
                if ann.get(kp_name) is not None:
                    kp_counts[kp_name] += 1
        
        for kp_name, count in kp_counts.items():
            percentage = (count / len(annotations)) * 100
            print(f"   {kp_name}: {count}/{len(annotations)} ({percentage:.1f}%)")
    else:
        print(f"\n❌ No valid annotations generated. Check your masks and try again.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate keypoint annotations from binary masks"
    )
    parser.add_argument(
        "--masks", 
        default=MASK_DIR, 
        help="Directory containing binary masks"
    )
    parser.add_argument(
        "--images", 
        default=IMAGE_DIR, 
        help="Directory containing corresponding images"
    )
    parser.add_argument(
        "--output", 
        default=OUTPUT_JSON, 
        help="Output JSON file path"
    )
    parser.add_argument(
        "--min-keypoints", 
        type=int, 
        default=MIN_KEYPOINTS,
        help="Minimum number of keypoints required"
    )
    
    args = parser.parse_args()
    
    generate_annotations(
        mask_dir=args.masks,
        image_dir=args.images,
        output_json=args.output,
        min_keypoints=args.min_keypoints
    )
