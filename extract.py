import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

def extract_vein_parameters_from_masks_directory(masks_directory, pixel_to_mm_ratio=0.1, ntheta_value=30):
    """
    Extracts geometric parameters from 2D binary vein masks located in a directory.

    Args:
        masks_directory (str): Path to the directory containing binary mask images.
                                Expects files to be named in a way that allows sorting
                                into correct z-order (e.g., mask_001.png, mask_002.png).
        pixel_to_mm_ratio (float): Conversion factor from pixels in the image to
                                   real-world units (e.g., mm). This is crucial for
                                   accurate spatial measurements from 2D masks.
        ntheta_value (int): The desired ntheta value for the final parametric STL generation.
                            This is a mesh resolution parameter for the STL model,
                            not directly extracted from the mask geometry, but included
                            in the output for convenience when using with 'copy_of_sinusstl.py'.

    Returns:
        dict: A dictionary containing the extracted parameters:
              'r', 'height', 'rbmax', 'hb1', 'hb2', 'nheight', 'ntheta'.
              Returns None if no masks are found or no valid vein contours are extracted.
    """
    if not os.path.isdir(masks_directory):
        print(f"Error: Directory '{masks_directory}' not found.")
        return None

    # Define common image file extensions to search for.
    # Add or remove extensions here if your mask files use different formats.
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    mask_files = []
    # Collect all image files from the specified directory
    for ext in image_extensions:
        mask_files.extend(glob.glob(os.path.join(masks_directory, ext)))
    # Sort the files to ensure they are processed in the correct Z-order (e.g., mask_001, mask_002, ...)
    mask_files = sorted(mask_files)

    if not mask_files:
        print(f"No image files found in '{masks_directory}'. Please check the directory path and file extensions.")
        return None

    radii_at_height = [] # Stores radius for each slice in real-world units (mm)
    heights = []        # Stores the corresponding height (Z-coordinate) for each slice (mm)

    # 'slice_thickness_mm' represents the real-world spacing between each 2D mask slice.
    # It's assumed that each mask corresponds to a single 'pixel-depth' in Z, so the
    # pixel_to_mm_ratio applies directly to Z-spacing as well.
    # Adjust this value if your ultrasound frames have a different real-world spacing.
    slice_thickness_mm = pixel_to_mm_ratio

    print(f"Processing {len(mask_files)} mask files from '{masks_directory}'...")
    for i, mask_filepath in enumerate(mask_files):
        # Read the mask image in grayscale
        mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Warning: Could not read mask file '{mask_filepath}'. Skipping.")
            continue

        # Ensure the mask is strictly binary (0 or 255).
        # This handles cases where input might be slightly off-binary grayscale.
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask.
        # cv2.RETR_EXTERNAL retrieves only the outermost contours.
        # cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments.
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_radius_mm = 0.0 # Default radius if no valid contour is found in this slice
        if contours:
            # Assume the largest contour corresponds to the main vein lumen
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate the minimum enclosing circle for the largest contour.
            # This gives a robust estimate of the equivalent circular radius.
            (x, y), radius_pixels = cv2.minEnclosingCircle(largest_contour)

            # Convert the radius from pixels to real-world units (mm)
            current_radius_mm = radius_pixels * pixel_to_mm_ratio
        else:
            print(f"No vein contour found in '{os.path.basename(mask_filepath)}'. Radius set to 0.0.")

        radii_at_height.append(current_radius_mm)
        # Calculate the height (Z-coordinate) for this slice based on its index and thickness
        heights.append(i * slice_thickness_mm)

    # Convert lists to NumPy arrays for efficient numerical operations
    radii_at_height = np.array(radii_at_height)
    heights = np.array(heights)

    # Filter out slices where no valid contour was found (i.e., radius was 0).
    # This prevents empty or problematic slices from skewing the overall calculations.
    valid_indices = radii_at_height > 0
    radii_at_height = radii_at_height[valid_indices]
    heights = heights[valid_indices]

    if not radii_at_height.size > 0:
        print("After filtering, no valid radii remain. All masks might be empty or problematic.")
        return None

    # Calculate the total height of the vein segment, from the first valid slice to the last.
    total_height = heights[-1] - heights[0] if len(heights) > 1 else 0

    # Calculate the baseline radius (r).
    # A common approach is to trim the top/bottom percentages to avoid noise or partial scans
    # at the extremities of the volume. This is a heuristic that can be adjusted.
    trim_percentage = 0.1 # Trim 10% from both start and end
    num_valid_slices = len(radii_at_height)
    start_trim = int(num_valid_slices * trim_percentage)
    end_trim = int(num_valid_slices * (1 - trim_percentage))

    if end_trim <= start_trim: # Handle cases with very few valid slices where trimming might remove all data
        baseline_radii = radii_at_height
    else:
        baseline_radii = radii_at_height[start_trim:end_trim]

    # Calculate the average radius from the (optionally trimmed) baseline radii.
    r_avg = np.mean(baseline_radii) if baseline_radii.size > 0 else np.mean(radii_at_height)

    # Detect bulge parameters (rbmax, hb1, hb2)
    # Initialize these parameters to 0.0, indicating no significant bulge by default.
    rbmax = 0.0
    hb1 = 0.0
    hb2 = 0.0

    if radii_at_height.size > 0:
        max_radius = np.max(radii_at_height)
        if max_radius > r_avg: # A bulge exists if the maximum radius is greater than the average radius
            # rbmax: The maximum increase in radius from the average baseline radius.
            rbmax = max_radius - r_avg

            # Identify the "bulge region" where the radius exceeds a certain threshold
            # above the average radius. This threshold can be tuned.
            threshold_factor = 1.05 # e.g., radius must be at least 5% greater than average 'r'
            bulge_indices = np.where(radii_at_height > r_avg * threshold_factor)[0]

            if bulge_indices.size > 0:
                # hb1: The height (Z-coordinate) where the bulge region starts.
                hb1 = heights[bulge_indices[0]]
                # hb2: The height (Z-coordinate) where the bulge region ends.
                hb2 = heights[bulge_indices[-1]]
            else:
                # Fallback: If no clear bulge region is detected by the threshold,
                # use the location of the maximum radius and a small surrounding region
                # as a heuristic for hb1 and hb2.
                max_radius_idx = np.argmax(radii_at_height)
                hb1 = heights[max(0, max_radius_idx - 1)] # One slice before max
                hb2 = heights[min(len(heights) - 1, max_radius_idx + 1)] # One slice after max

    # Ensure hb1 is always less than or equal to hb2 for logical consistency.
    if hb1 > hb2:
        hb1, hb2 = hb2, hb1

    # nheight_param: Total number of mask files processed. This represents the axial resolution
    # for the parametric model.
    nheight_param = len(mask_files)

    # Store all extracted parameters in a dictionary.
    extracted_params = {
        "r": r_avg,
        "height": total_height,
        "rbmax": rbmax,
        "hb1": hb1,
        "hb2": hb2,
        "nheight": nheight_param,
        "ntheta": ntheta_value # This is a user-defined resolution for the STL generation
    }

    print("\n--- Extracted Parameters ---")
    for k, v in extracted_params.items():
        # Format floats to 3 decimal places for readability
        print(f"{k}: {v:.3f}" if isinstance(v, (float, np.float64)) else f"{k}: {v}")
    print("----------------------------")

    # Save the extracted parameters to a CSV file.
    csv_output_path = "vein_geometry_params.csv"
    try:
        df_params = pd.DataFrame([extracted_params])
        df_params.to_csv(csv_output_path, index=False)
        print(f"\n✅ Parameters successfully saved to '{csv_output_path}'")
    except Exception as e:
        print(f"Error saving parameters to CSV: {e}")

    # Optional: Plotting the radii profile along the vein's height for visualization.
    if heights.size > 0 and radii_at_height.size > 0:
        plt.figure(figsize=(8, 6))
        plt.plot(heights, radii_at_height, marker='o', linestyle='-', label='Vein Radius Profile')
        plt.axhline(y=r_avg, color='r', linestyle='--', label=f'Average Radius (r): {r_avg:.3f} mm')
        plt.axvline(x=hb1, color='g', linestyle=':', label=f'Bulge Start (h1): {hb1:.3f} mm')
        plt.axvline(x=hb2, color='b', linestyle=':', label=f'Bulge End (h2): {hb2:.3f} mm')
        plt.xlabel("Height (mm)")
        plt.ylabel("Radius (mm)")
        plt.title("Vein Radius Profile Along Height")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Skipping plot: No valid height or radius data to display after processing.")

    return extracted_params

# --- Example Usage: Simulate Loading Masks from a Directory ---
# THIS SECTION IS FOR DEMONSTRATION PURPOSES ONLY.
# FOR ACTUAL USE, YOU SHOULD REPLACE THIS ENTIRE SECTION WITH YOUR CODE
# THAT POINTS TO YOUR 'masks_binary' DIRECTORY CONTAINING YOUR REAL MASK IMAGES.

dummy_masks_dir = "masks_binary2"

# Check if the dummy directory exists AND contains files.
# If it doesn't exist, or if it's empty, create it and generate dummy masks.
# This prevents overwriting your actual mask files if you've already placed them there.
if not os.path.exists(dummy_masks_dir) or not glob.glob(os.path.join(dummy_masks_dir, "*")):
    os.makedirs(dummy_masks_dir, exist_ok=True) # Ensure the directory exists
    image_size = 100
    num_frames = 25 # Number of dummy mask files to create

    print(f"Simulating creation of {num_frames} dummy mask files in '{dummy_masks_dir}' for demonstration purposes...")
    for i in range(num_frames):
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        center_x, center_y = image_size // 2, image_size // 2
        current_radius = 20 # Base radius for the dummy vein
        # Simulate a cylindrical vein with a bulge in the middle (between frames 5 and 20)
        if 5 <= i <= 20:
            bulge_factor = np.sin(np.pi * (i - 5) / (20 - 5)) # Sinusoidal bulge shape
            current_radius = 20 + 15 * bulge_factor # Max radius increases to 35 pixels
        cv2.circle(mask, (center_x, center_y), int(current_radius), 255, -1) # Draw a filled white circle
        cv2.imwrite(os.path.join(dummy_masks_dir, f"dummy_mask_{i:03d}.png"), mask) # Save as PNG
    print("Dummy mask files created.")
else:
    print(f"Using existing mask files in '{dummy_masks_dir}'. To regenerate dummy files, please delete the directory first.")


# Call the function to extract parameters and save them to CSV.
# Set 'pixel_to_mm_ratio' based on your ultrasound image resolution.
# 'ntheta_value' is a resolution parameter for the STL generation, set to 30 as commonly used.
extracted_parameters = extract_vein_parameters_from_masks_directory(
    masks_directory=dummy_masks_dir,
    pixel_to_mm_ratio=0.1, # Example: 1 pixel in your image corresponds to 0.1 mm in reality
    ntheta_value=30
)

# --- Optional Cleanup: Remove the dummy directory and files ---
# Uncomment the lines below if you want the script to automatically remove the
# dummy 'masks_binary' directory and its contents after execution.
# import shutil
# if os.path.exists(dummy_masks_dir):
#     print(f"\nCleaning up: Removing dummy directory '{dummy_masks_dir}'...")
#     shutil.rmtree(dummy_masks_dir)
#     print("Cleanup complete.")