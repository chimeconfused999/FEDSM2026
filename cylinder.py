import numpy as np
import cv2
from stl import mesh
from skimage import measure

# --- Parameters ---
HEIGHT = 100         # Number of slices (Z-dimension)
RADIUS_OUTER = 60
RADIUS_INNER = 40
IMG_SIZE = 150       # Width & height of each 2D mask
OUTPUT_FILE = "hollow_cylinder.stl"

# --- Create binary cylinder volume ---
volume = []

for _ in range(HEIGHT):
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    center = (IMG_SIZE // 2, IMG_SIZE // 2)

    # Outer circle
    cv2.circle(img, center, RADIUS_OUTER, 255, thickness=-1)
    # Inner cut-out (hollow core)
    cv2.circle(img, center, RADIUS_INNER, 0, thickness=-1)

    volume.append((img > 0).astype(np.uint8))

volume = np.array(volume)

# --- Generate 3D surface ---
verts, faces, normals, _ = measure.marching_cubes(volume.astype(np.float32), level=0.5, spacing=(1.0, 1.0, 1.0))

# --- Export STL ---
vein_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        vein_mesh.vectors[i][j] = verts[f[j], :]

vein_mesh.save(OUTPUT_FILE)
print(f"✅ Hollow cylinder STL saved as '{OUTPUT_FILE}'")
