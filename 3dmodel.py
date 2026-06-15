import os
import numpy as np
import cv2
from skimage import measure
from scipy.ndimage import gaussian_filter
from stl import mesh


# --- Config ---
MASK_DIR = "masks_binary"
OUTPUT_FILE = "vein_model_smoothed.stl"
TARGET_WIDTH = 150  # Resize width
TARGET_HEIGHT = 120  # Resize height
Z_SPACING = 1.0  # Vertical spacing per slice

# --- Load and process masks ---
volume = []

mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith((".png", ".jpg"))])
print(f"🔍 Found {len(mask_files)} mask slices.")

for fname in mask_files:
    path = os.path.join(MASK_DIR, fname)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue

    # Focus on middle 50% region of the mask only
    h, w = mask.shape
    cropped = mask[h//4:3*h//4, w//4:3*w//4]
    resized = cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT))

    # Binary threshold
    binary = (resized > 50).astype(np.uint8)

    # Filter contours by area
    min_area = 500  # you can increase to 1000 if needed
    filled = np.zeros_like(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area (descending)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Keep only top 2 largest contours (adjust if needed)
    for cnt in contours[:2]:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
            


    # Optional: close small holes/gaps to smooth shape
    kernel = np.ones((3, 3), np.uint8)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)

    # Add to volume
    volume.append((filled > 0).astype(np.uint8))




if len(volume) < 3:
    raise ValueError("Not enough mask slices to build a 3D model.")

volume = np.array(volume, dtype=np.uint8)

from scipy.ndimage import label

# Label all connected 3D regions
labeled_volume, num_features = label(volume)

# Calculate sizes of all regions
sizes = [(labeled_volume == i).sum() for i in range(1, num_features + 1)]

# Keep top 2 largest regions
if sizes:
    top_labels = np.argsort(sizes)[-2:]  # top 2
    filtered = np.isin(labeled_volume, [l + 1 for l in top_labels]).astype(np.uint8)
    volume = filtered

# --- Smooth the volume ---
smoothed_volume = gaussian_filter(volume.astype(np.float32), sigma=1.2)

# --- Generate 3D surface ---
verts, faces, normals, _ = measure.marching_cubes(
    smoothed_volume, level=0.3, spacing=(Z_SPACING, 1.0, 1.0)
)

# --- Export to STL ---
vein_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        vein_mesh.vectors[i][j] = verts[f[j], :]

vein_mesh.save(OUTPUT_FILE)
print(f"✅ 3D vein STL model saved as '{OUTPUT_FILE}'")
