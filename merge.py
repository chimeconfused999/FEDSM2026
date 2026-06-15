import os
import numpy as np
import cv2
from skimage import measure
from scipy.ndimage import gaussian_filter, binary_dilation, label
from stl import mesh

# --- Config ---
MASK_DIR = "masks_binary"
OUTPUT_FILE_FULL_VEIN = "vein_model_full_with_valve.stl"
OUTPUT_FILE_HALF_VEIN = "vein_model_half_with_valve.stl" # New output file for the half-vein
TARGET_WIDTH = 150  # Resize width for consistency across slices
TARGET_HEIGHT = 120 # Resize height for consistency across slices
Z_SPACING = 1.0     # Vertical spacing per slice in your 3D reconstruction.
                    # IMPORTANT: Adjust this to reflect the real-world distance between your image slices.
                    # If your slices are very far apart, increase this value.
                    # If they are very close, decrease it.
                    # This dramatically affects the perceived "squash" or "stretch" of the model.

# --- Valve Parameters ---
VALVE_START_SLICE_INDEX = 10  # Approximate slice index where the valve begins (0-indexed)
VALVE_END_SLICE_INDEX = 25    # Approximate slice index where the valve ends (exclusive)
                              # Adjust these indices based on where you want the bulge.
                              # E.g., if you have 50 slices, a valve in the middle might be 20 to 30.
VALVE_MAX_BULGE_FACTOR = 1.4  # Max multiplier for the vein's size at the bulge (e.g., 1.4 means 40% wider).
                              # Controls how pronounced the bulge is.
VALVE_DILATION_KERNEL_SIZE = 3 # Kernel size for binary_dilation (e.g., 3 for a 3x3 kernel).
                               # Keep this small for smooth growth.
VALVE_MAX_DILATION_ITERATIONS = 5 # Maximum number of dilation iterations for the bulge.
                                  # Adjust this if the bulge is too thin or too fat.
                                  # This is a fallback/cap for the heuristic calculation.

# --- Half-Vein Cut Parameter ---
CUT_AXIS = 'y' # 'x' or 'y' - which axis to cut along in the 3D model.
               # 'y' (verts[:,1]) for a vertical cut through the "height" of the vein.
               # 'x' (verts[:,2]) for a horizontal cut through the "width" of the vein.
CUT_POSITION_FACTOR = 0.5 # 0.5 means cut in the middle. 0.25 means cut the first quarter, etc.
                          # The "half" will be the part where the coordinate is GREATER than the cut_value.
                          # To get the other half, change `>` to `<` in the cutting logic.

# --- Mask Processing Parameters ---
MIN_CONTOUR_AREA = 500 # Minimum area for contours to be considered.
                       # If your masks are very small or noisy, lower this.
                       # If it's too low, you'll pick up noise.

# --- Smoothing Parameters ---
# INCREASED SIGMA FOR MORE CYLINDRICAL SHAPE
GAUSSIAN_SMOOTHING_SIGMA = 2.5 # Sigma for Gaussian smoothing. Higher values mean smoother,
                               # but potentially less detailed results. Try 2.0, 2.5, 3.0.
MARCHING_CUBES_LEVEL = 0.3     # Iso-surface level for Marching Cubes. Affects how "thick" or "thin"
                               # the final mesh appears. Typically 0.2-0.8 for smoothed volumes.

# --- Load and process masks ---
volume = []
mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith((".png", ".jpg", ".jpeg"))])
print(f"🔍 Found {len(mask_files)} mask slices in '{MASK_DIR}'.")

if not mask_files:
    raise FileNotFoundError(f"No mask files found in '{MASK_DIR}'. Please check the directory and file extensions.")

total_slices = len(mask_files)

# --- Optional: Visualize intermediate steps for debugging ---
# Set to True to see masks at various stages. Press any key to advance.
# Remember to disable for final runs to prevent pop-up windows.
DEBUG_VISUALIZATION = False 

for i, fname in enumerate(mask_files):
    path = os.path.join(MASK_DIR, fname)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"⚠️ Warning: Could not read mask file '{fname}'. Skipping.")
        continue

    # Ensure mask is binary (0 or 255) if it's not already
    mask = (mask > 0).astype(np.uint8) * 255

    # Focus on middle 50% region of the mask only
    h, w = mask.shape
    cropped = mask[h//4:3*h//4, w//4:3*w//4]
    
    # Resize to target dimensions
    resized = cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    # Binary threshold (already done by (mask > 0) above, but good for clarity)
    binary = (resized > 0).astype(np.uint8)

    # Filter contours by area and fill
    filled = np.zeros_like(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (descending) to easily pick largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Keep only the top 2 largest contours (adjust if needed, or remove if always one main structure)
    for cnt in contours[:2]:
        if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
            cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
            
    # Optional: close small holes/gaps to smooth shape
    # This helps fill tiny internal holes or connect small breaks.
    morph_kernel = np.ones((3, 3), np.uint8)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, morph_kernel)

    # --- Apply Valve Bulge ---
    current_dilation_iterations = 0

    if VALVE_START_SLICE_INDEX <= i < VALVE_END_SLICE_INDEX:
        valve_range = VALVE_END_SLICE_INDEX - VALVE_START_SLICE_INDEX
        
        # Ensure valve_range is positive to avoid issues
        if valve_range <= 0:
            progress = 0.5 # Treat as peak if range is 0 or negative
        else:
            progress = (i - VALVE_START_SLICE_INDEX) / valve_range

        # Create a bell-shaped curve for dilation strength (0 at ends, peak in middle)
        # Using a cosine wave shifted and scaled for a smooth bulge
        # Value goes from 0 to 1 and back to 0 over the valve range
        bulge_strength = (1 - np.cos(progress * 2 * np.pi)) / 2 

        # Calculate approximate number of iterations needed
        # This heuristic estimates iterations to achieve a certain bulge factor relative to original size
        # Assuming each iteration adds ~1 voxel thickness to the "radius"
        base_radius_approx = np.sqrt(np.sum(filled > 0) / np.pi) # Approximate radius from area
        if base_radius_approx > 0:
            # Target radius = base_radius * VALVE_MAX_BULGE_FACTOR
            # Iterations = (target_radius - base_radius) / (dilation_step_per_iteration)
            # Dilation step per iteration is roughly 1 voxel for a 3x3 kernel
            target_radius_increase = (base_radius_approx * VALVE_MAX_BULGE_FACTOR) - base_radius_approx
            current_dilation_iterations = int(round(target_radius_increase * bulge_strength))
        else:
            current_dilation_iterations = 0 # No vein to dilate if no area

        # Cap iterations to prevent excessive dilation
        current_dilation_iterations = min(current_dilation_iterations, VALVE_MAX_DILATION_ITERATIONS)
        current_dilation_iterations = max(0, current_dilation_iterations) # Ensure non-negative

        if current_dilation_iterations > 0:
             filled = cv2.dilate(filled, np.ones((VALVE_DILATION_KERNEL_SIZE, VALVE_DILATION_KERNEL_SIZE), np.uint8),
                                 iterations=current_dilation_iterations)
             filled = (filled > 0).astype(np.uint8) # Ensure it's still binary (0 or 1)

    # Add processed 2D slice to the 3D volume
    volume.append(filled)

    if DEBUG_VISUALIZATION:
        debug_img = filled * 255 # Scale to 0-255 for visualization
        cv2.putText(debug_img, f"Slice {i+1}/{total_slices}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_img, f"Dilation: {current_dilation_iterations}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Processed Slice", debug_img)
        if cv2.waitKey(0) & 0xFF == ord('q'): # Press 'q' to quit visualization early
            break 

if DEBUG_VISUALIZATION:
    cv2.destroyAllWindows()

if len(volume) < 3:
    raise ValueError(
        f"Not enough mask slices to build a 3D model. Found {len(volume)} usable slices. "
        "Please check if 'masks_binary' is correct and contains at least 3 valid image files."
    )

volume = np.array(volume, dtype=np.uint8)

# --- 3D Connected Component Filtering ---
print("Applying 3D connected component analysis...")
labeled_volume, num_features = label(volume)

sizes = [(labeled_volume == i).sum() for i in range(1, num_features + 1)]

if sizes:
    num_components_to_keep = min(2, num_features) # Keep up to 2 largest if available
    top_labels_indices = np.argsort(sizes)[-num_components_to_keep:]
    
    filtered_labels = [l + 1 for l in top_labels_indices]
    
    filtered_volume = np.isin(labeled_volume, filtered_labels).astype(np.uint8)
    volume = filtered_volume
    print(f"Kept {len(filtered_labels)} largest 3D connected components.")
else:
    print("No 3D connected components found after filtering.")

# --- Smooth the 3D volume ---
print(f"Smoothing 3D volume with sigma={GAUSSIAN_SMOOTHING_SIGMA}...")
smoothed_volume = gaussian_filter(volume.astype(np.float32), sigma=GAUSSIAN_SMOOTHING_SIGMA)

# --- Generate 3D surface (Full Vein) using Marching Cubes ---
print(f"Generating full 3D surface with Marching Cubes (level={MARCHING_CUBES_LEVEL})...")
verts_full, faces_full, normals_full, _ = measure.marching_cubes(
    smoothed_volume, level=MARCHING_CUBES_LEVEL, spacing=(Z_SPACING, 1.0, 1.0)
)

if len(verts_full) == 0 or len(faces_full) == 0:
    print("❌ No surface generated for the full vein. Check input masks, smoothing, and marching cubes level.")
else:
    # --- Export Full Vein to STL ---
    print(f"Exporting full vein model to '{OUTPUT_FILE_FULL_VEIN}'...")
    vein_mesh_full = mesh.Mesh(np.zeros(faces_full.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces_full):
        for j in range(3):
            vein_mesh_full.vectors[i][j] = verts_full[f[j], :]

    vein_mesh_full.save(OUTPUT_FILE_FULL_VEIN)
    print(f"✅ Full vein STL model with valve saved as '{OUTPUT_FILE_FULL_VEIN}'")


    # --- Create and Export Half Vein ---
    print(f"Attempting to generate half vein model along '{CUT_AXIS}' axis...")
    verts_half = []
    faces_half = []

    if CUT_AXIS == 'x':
        min_x = np.min(verts_full[:, 2])
        max_x = np.max(verts_full[:, 2])
        cut_value = min_x + (max_x - min_x) * CUT_POSITION_FACTOR
        axis_idx = 2
    elif CUT_AXIS == 'y':
        min_y = np.min(verts_full[:, 1])
        max_y = np.max(verts_full[:, 1])
        cut_value = min_y + (max_y - min_y) * CUT_POSITION_FACTOR
        axis_idx = 1
    else:
        print(f"⚠️ Warning: Invalid CUT_AXIS '{CUT_AXIS}'. Please choose 'x' or 'y'. No half-vein model generated.")
        faces_half = []
        axis_idx = -1

    if axis_idx != -1:
        for i, f in enumerate(faces_full):
            v1, v2, v3 = verts_full[f[0]], verts_full[f[1]], verts_full[f[2]]
            
            # Change `>` to `<` to get the other half of the vein.
            if v1[axis_idx] > cut_value and v2[axis_idx] > cut_value and v3[axis_idx] > cut_value:
                faces_half.append(f)

    if faces_half:
        vein_mesh_half = mesh.Mesh(np.zeros(len(faces_half), dtype=mesh.Mesh.dtype))
        for i, f_idx in enumerate(faces_half):
            for j in range(3):
                vein_mesh_half.vectors[i][j] = verts_full[f_idx[j], :]

        vein_mesh_half.save(OUTPUT_FILE_HALF_VEIN)
        print(f"✅ Half vein STL model with valve saved as '{OUTPUT_FILE_HALF_VEIN}'")
    else:
        print("❌ No faces remained after cutting for the half-vein model. Check cut parameters and full model generation.")