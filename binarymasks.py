import os
import cv2
import numpy as np

# --- Folder paths ---
input_folder = 'masks_annotated'   # where your current annotated edge images are
output_folder = 'masks_binary'     # where binary masks will be saved

os.makedirs(output_folder, exist_ok=True)

# --- White color threshold ---
LOWER_WHITE = np.array([200, 200, 200])  # adjust if needed
UPPER_WHITE = np.array([255, 255, 255])

# --- Process each annotated mask ---
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        # Create mask: white where lines are drawn
        white_mask = cv2.inRange(img, LOWER_WHITE, UPPER_WHITE)

        # Save result to output folder
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, white_mask)

        print(f"✅ Converted: {filename}")

print("\n🎉 All annotated images converted to binary masks.")
