import argparse
import os

import cv2
import numpy as np

from cinevalve.safety import assert_safe_output_dir

LOWER_GREEN = np.array([40, 40, 40])
UPPER_GREEN = np.array([90, 255, 255])


def convert_green_to_binary(input_folder, output_folder, confirm_overwrite_venous=False):
    assert_safe_output_dir(
        output_folder,
        confirm_overwrite_venous=confirm_overwrite_venous,
        purpose="write binary masks",
    )

    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(input_folder, filename)
        img_bgr = cv2.imread(img_path)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(img_hsv, LOWER_GREEN, UPPER_GREEN)
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, green_mask)
        count += 1

    print(f"Converted {count} images to binary masks in '{output_folder}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert green annotations to binary masks.")
    parser.add_argument("--input", default="images2", help="Folder with green-annotated images")
    parser.add_argument("--out", default="data/masks/training", help="Output folder for binary masks")
    parser.add_argument(
        "--confirm-overwrite-venous",
        action="store_true",
        help="Required when writing into a protected venous folder",
    )
    args = parser.parse_args()

    convert_green_to_binary(args.input, args.out, args.confirm_overwrite_venous)
