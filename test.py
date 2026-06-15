import pandas as pd
from copy_of_sinusstl import Geo2STL, save_d2f

# --- Load the .csv file ---
csv_path = "vein_geometry_params.csv"  # Update if your file name differs
df = pd.read_csv(csv_path)

# Convert first row to dictionary
cleaned_params = df.iloc[0].to_dict()

# Make sure necessary parameters are cast correctly
for key in ["nheight", "ntheta", "nsd", "nen"]:
    if key in cleaned_params:
        cleaned_params[key] = int(cleaned_params[key])

# --- Preview the result ---
for k, v in cleaned_params.items():
    print(f"{k}: {v} (type={type(v)})")

# Initialize mesh generator
cyl = Geo2STL({"fn": "generated_from_csv.stl"})
cyl.Init(cleaned_params)
cyl.mesh_x3d()
cyl.IEN()
cyl.Extractx2D()
cyl.mesh2stl()

# Save STL
save_d2f("generated_from_csv.stl", cyl.stl)
print("✅ STL generated and saved as 'generated_from_csv.stl'")
