"""
Preliminary CFD integration: export valve geometry and run a simplified
2D steady Stokes flow simulation on the lumen domain derived from segmentation.

Outputs:
  - cfd_output/geometry/frame_XXXX_profile.csv   (2D boundary points, mm)
  - cfd_output/meshes/valve_domain_XXXX.stl      (extruded 2D lumen mesh)
  - cfd_output/simulations/velocity_XXXX.png     (preliminary flow field)
  - cfd_output/cfd_summary.json
"""

import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from stl import mesh as stl_mesh


def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 0).astype(np.uint8)


def extract_lumen_contour(binary_mask, min_area=200):
    contours, _ = cv2.findContours(
        (binary_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    return contour.squeeze()


def contour_to_mm(contour, pixel_to_mm=0.1):
    pts = contour.astype(np.float64)
    pts_mm = pts.copy()
    pts_mm[:, 0] *= pixel_to_mm
    pts_mm[:, 1] *= pixel_to_mm
    return pts_mm


def export_profile_csv(contour_mm, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("x_mm,y_mm\n")
        for x, y in contour_mm:
            f.write(f"{x:.4f},{y:.4f}\n")


def extrude_contour_stl(contour_mm, out_path, depth_mm=5.0):
    """Extrude closed 2D contour into a thin 3D STL (CFD wall boundary proxy)."""
    n = len(contour_mm)
    if n < 3:
        return False

    z0, z1 = 0.0, depth_mm
    verts = []
    faces = []

    for x, y in contour_mm:
        verts.append([x, y, z0])
    for x, y in contour_mm:
        verts.append([x, y, z1])

    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, j + n])
        faces.append([i, j + n, i + n])

    # Cap bottom (fan from vertex 0)
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])
    # Cap top
    for i in range(1, n - 1):
        faces.append([n, n + i, n + i + 1])

    verts = np.array(verts, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32)

    valve_mesh = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            valve_mesh.vectors[i][j] = verts[face[j], :]

    valve_mesh.save(out_path)
    return True


def edge_mask_to_lumen(binary_mask, close_kernel=15):
    """Approximate fluid lumen from thin edge/leaflet segmentation."""
    h, w = binary_mask.shape
    closed = cv2.morphologyEx(
        (binary_mask * 255).astype(np.uint8), cv2.MORPH_CLOSE,
        np.ones((close_kernel, close_kernel), np.uint8),
    )
    filled = closed.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    seed = (w // 2, h // 4)
    cv2.floodFill(filled, flood_mask, seed, 255)
    lumen = filled > 0
    if lumen.sum() < 0.01 * h * w:
        # Fallback: dilate edges and invert outer background
        dilated = cv2.dilate(binary_mask.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=2)
        lumen = dilated > 0
    return lumen.astype(bool)


def build_flow_domain(mask, pixel_to_mm=0.1):
    """Downsample lumen domain derived from segmentation mask."""
    lumen = edge_mask_to_lumen(mask)
    h, w = lumen.shape
    scale = 4 if max(h, w) > 256 else 2
    small = cv2.resize(lumen.astype(np.uint8), (w // scale, h // scale), interpolation=cv2.INTER_NEAREST)
    fluid = small > 0
    solid = ~fluid
    return fluid, solid, scale, pixel_to_mm * scale


def solve_flow_proxy(fluid):
    """
    Preliminary flow field proxy using distance-from-wall transform.
    High velocity in open lumen; low velocity in sinus pockets (recirculation proxy).
    """
    if not np.any(fluid):
        return None, None, None

    dist = distance_transform_edt(fluid)
    ny, nx = fluid.shape

    # Poiseuille-like axial gradient (left → right)
    x_coords = np.linspace(1.0, 0.2, nx)
    axial = np.tile(x_coords, (ny, 1))

    speed = dist * axial
    speed[~fluid] = np.nan
    if np.nanmax(speed) > 0:
        speed = speed / np.nanmax(speed)

    # Pseudo velocity components for recirculation detection
    ux = np.gradient(np.nan_to_num(speed, nan=0), axis=1)
    uy = np.gradient(np.nan_to_num(speed, nan=0), axis=0)
    ux[~fluid] = np.nan
    uy[~fluid] = np.nan

    return speed, ux, uy


def identify_recirculation(ux, uy, fluid, x_split=None):
    """Low-speed regions and reverse-flow fraction as recirculation proxy."""
    speed = np.sqrt(ux ** 2 + uy ** 2)
    ny, nx = fluid.shape
    if x_split is None:
        x_split = nx // 2

    downstream = fluid.copy()
    downstream[:, :x_split] = False

    valid = downstream & np.isfinite(speed)
    if not np.any(valid):
        return {"low_speed_fraction": np.nan, "reverse_flow_fraction": np.nan}

    spd = speed[valid]
    low_thresh = np.percentile(spd, 25)
    low_speed_frac = float(np.mean(spd < low_thresh))

    ux_d = ux[valid]
    reverse_frac = float(np.mean(ux_d < 0))

    return {
        "low_speed_fraction": low_speed_frac,
        "reverse_flow_fraction": reverse_frac,
        "low_speed_threshold": float(low_thresh),
    }


def plot_flow_field(speed, fluid, contour_orig, scale, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    display = np.where(fluid, speed, np.nan)
    im = ax.imshow(display, cmap="viridis", origin="upper")
    plt.colorbar(im, ax=ax, label="Velocity magnitude (a.u.)")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def select_representative_frames(metrics_summary_path, mask_dir, num=3):
    """Pick closed, partial-open, and max-open frames from valve metrics if available."""
    csv_path = metrics_summary_path.replace("_summary.json", ".csv")
    if os.path.isfile(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "angle_smooth_deg" in df.columns:
            valid = df.dropna(subset=["angle_smooth_deg"])
            if len(valid) >= 3:
                idx_min = int(valid["angle_smooth_deg"].idxmin())
                idx_max = int(valid["angle_smooth_deg"].idxmax())
                idx_mid = int(valid.iloc[len(valid) // 2].name)
                frames = []
                for i in sorted(set([idx_min, idx_mid, idx_max])):
                    fname = f"frame_{i:04d}.png"
                    if os.path.isfile(os.path.join(mask_dir, fname)):
                        frames.append(fname)
                if frames:
                    return frames

    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    if len(mask_files) >= 3:
        picks = [0, len(mask_files) // 2, len(mask_files) - 1]
        return [mask_files[i] for i in picks]
    return mask_files[:1]


def main():
    parser = argparse.ArgumentParser(description="Preliminary CFD geometry export and flow simulation.")
    parser.add_argument("--masks", default="outputs/predictions", help="Directory of predicted binary masks")
    parser.add_argument("--metrics-summary", default="valve_metrics_summary.json")
    parser.add_argument("--pixel-to-mm", type=float, default=0.1)
    parser.add_argument("--out", default="cfd_output")
    parser.add_argument("--num-frames", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    geom_dir = os.path.join(args.out, "geometry")
    mesh_dir = os.path.join(args.out, "meshes")
    sim_dir = os.path.join(args.out, "simulations")
    for d in (geom_dir, mesh_dir, sim_dir):
        os.makedirs(d, exist_ok=True)

    frames = select_representative_frames(args.metrics_summary, args.masks)
    frames = frames[: args.num_frames]
    results = []

    for fname in frames:
        path = os.path.join(args.masks, fname)
        mask = load_mask(path)
        if mask is None:
            continue

        stem = os.path.splitext(fname)[0]
        lumen = edge_mask_to_lumen(mask)
        contour = extract_lumen_contour(lumen.astype(np.uint8))
        if contour is None or len(contour) < 5:
            print(f"Skipping {fname}: no valid contour")
            continue

        contour_mm = contour_to_mm(contour, args.pixel_to_mm)
        export_profile_csv(contour_mm, os.path.join(geom_dir, f"{stem}_profile.csv"))
        stl_path = os.path.join(mesh_dir, f"valve_domain_{stem}.stl")
        extrude_contour_stl(contour_mm, stl_path)

        fluid, solid, scale, mm_scale = build_flow_domain(mask, args.pixel_to_mm)
        flow = solve_flow_proxy(fluid)
        if flow[0] is None:
            continue
        speed, ux, uy = flow
        recirc = identify_recirculation(ux, uy, fluid)
        plot_path = os.path.join(sim_dir, f"velocity_{stem}.png")
        plot_flow_field(speed, fluid, contour, scale, plot_path, f"Preliminary flow — {stem}")

        from cinevalve.geometry import extract_frame_geometry
        geom = extract_frame_geometry(lumen.astype(np.uint8))

        frame_result = {
            "frame": stem,
            "profile_csv": f"geometry/{stem}_profile.csv",
            "stl_mesh": f"meshes/valve_domain_{stem}.stl",
            "velocity_plot": f"simulations/velocity_{stem}.png",
            "pixel_to_mm": args.pixel_to_mm,
            "recirculation_proxy": recirc,
            "geometry": {k: (float(v) if np.isfinite(v) else None) for k, v in geom.items()},
        }
        results.append(frame_result)
        print(f"Processed {stem}: recirculation proxy = {recirc}")

    summary = {
        "description": "Preliminary CFD integration from automated segmentation masks",
        "mask_dir": args.masks,
        "pixel_to_mm_ratio": args.pixel_to_mm,
        "frames_processed": len(results),
        "frames": results,
        "openfoam_note": (
            "Import STL meshes into snappyHexMesh or extrude in OpenFOAM. "
            "Profile CSV files provide 2D boundary polylines in millimeters."
        ),
    }
    summary_path = os.path.join(args.out, "cfd_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Combined figure for paper
    if results:
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
        if len(results) == 1:
            axes = [axes]
        for ax, res in zip(axes, results):
            img = plt.imread(os.path.join(args.out, res["velocity_plot"]))
            ax.imshow(img)
            ax.set_title(res["frame"])
            ax.axis("off")
        fig.suptitle("Preliminary CFD — Velocity Fields at Representative Frames")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "cfd_overview.png"), dpi=150)
        plt.close(fig)

    print(f"\nSaved CFD summary: {summary_path}")
    print(f"Processed {len(results)} frame(s)")


if __name__ == "__main__":
    main()
