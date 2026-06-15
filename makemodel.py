import pandas as pd
import numpy as np
import trimesh
from scipy.interpolate import interp1d

# --- Load diameter data ---
df = pd.read_csv("vein_parameters.csv")
z_vals = df["frame"].to_numpy()
diameters = df["avg_diameter_px"].to_numpy()
radii = diameters / 2.0

# --- Optional: convert px to mm ---
px_to_mm = 0.1
z_vals = z_vals * px_to_mm
radii = radii * px_to_mm

# --- Interpolate radius profile ---
n_z = 100
z_interp = np.linspace(z_vals.min(), z_vals.max(), n_z)
f_interp = interp1d(z_vals, radii, kind='cubic', fill_value="extrapolate")
r_interp = f_interp(z_interp)

# --- Revolve to 3D surface ---
n_circle = 64
verts = []
faces = []

for i, z in enumerate(z_interp):
    r = r_interp[i]
    for j in range(n_circle):
        angle = 2 * np.pi * j / n_circle
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        verts.append([x, y, z])

for i in range(n_z - 1):
    for j in range(n_circle):
        a = i * n_circle + j
        b = i * n_circle + (j + 1) % n_circle
        c = (i + 1) * n_circle + j
        d = (i + 1) * n_circle + (j + 1) % n_circle
        faces.append([a, b, c])
        faces.append([b, d, c])

mesh = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces))
mesh.export("vein_model_from_profile.stl")

print("✅ STL file saved as 'vein_model_from_profile.stl'")
