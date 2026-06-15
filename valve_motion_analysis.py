import os
import json
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


def load_binary_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


def find_skeleton_endpoints(binary_mask):
    skel = skeletonize(binary_mask > 0).astype(np.uint8)
    h, w = skel.shape
    endpoints = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] == 0:
                continue
            neighborhood = skel[y - 1:y + 2, x - 1:x + 2]
            count = int(neighborhood.sum()) - 1
            if count == 1:
                endpoints.append((x, y))
    return skel, endpoints


def classify_keypoints(binary_mask):
    skel, endpoints = find_skeleton_endpoints(binary_mask)
    if len(endpoints) < 2:
        return None

    coords = np.array(endpoints, dtype=np.float32)
    centroid = coords.mean(axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)

    if len(endpoints) >= 4:
        tips_idx = np.argsort(distances)[-2:]
        base_idx = np.argsort(distances)[:2]
        tips = coords[tips_idx]
        bases = coords[base_idx]
    else:
        # Fallback: pick two tips farthest from centroid, and estimate bases near centroid
        tips_idx = np.argsort(distances)[-2:]
        tips = coords[tips_idx]

        # Estimate bases as the nearest contour points to centroid along skeleton
        base_candidates = coords[np.argsort(distances)[:2]]
        bases = base_candidates

    # Left/right ordering by x-coordinate
    tip_left, tip_right = tips[np.argsort(tips[:, 0])]
    base_left, base_right = bases[np.argsort(bases[:, 0])]

    keypoints = {
        "tip_left": tuple(tip_left),
        "tip_right": tuple(tip_right),
        "base_left": tuple(base_left),
        "base_right": tuple(base_right),
    }
    return keypoints


def match_points(prev_point, curr_point, max_jump):
    if prev_point is None:
        return curr_point
    if curr_point is None:
        return None
    dist = np.linalg.norm(np.array(prev_point) - np.array(curr_point))
    if dist > max_jump:
        return None
    return curr_point


def compute_angle(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    dot = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def smooth_series(values, window=5):
    if window <= 1:
        return values
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    half = window // 2
    for i in range(len(values)):
        chunk = values[max(0, i - half) : min(len(values), i + half + 1)]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk):
            out[i] = float(np.mean(chunk))
    return out


def analyze_masks(mask_dir, fps=30.0, max_jump=20, smooth_window=5, output_prefix="valve_metrics"):
    mask_files = sorted([
        f for f in os.listdir(mask_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")

    metrics = []
    prev = {k: None for k in ["tip_left", "tip_right", "base_left", "base_right"]}

    for i, fname in enumerate(mask_files):
        mask_path = os.path.join(mask_dir, fname)
        mask = load_binary_mask(mask_path)
        if mask is None:
            continue

        keypoints = classify_keypoints(mask)
        if keypoints is None:
            keypoints = {k: None for k in prev.keys()}

        # Apply continuity constraints
        tracked = {}
        for k in prev.keys():
            tracked[k] = match_points(prev[k], keypoints.get(k), max_jump=max_jump)
        prev = tracked

        if any(v is None for v in tracked.values()):
            angle = np.nan
            asymmetry = np.nan
            left_len = np.nan
            right_len = np.nan
        else:
            base_mid = (np.array(tracked["base_left"]) + np.array(tracked["base_right"])) / 2.0
            v_left = np.array(tracked["tip_left"]) - base_mid
            v_right = np.array(tracked["tip_right"]) - base_mid
            angle = compute_angle(v_left, v_right)
            left_len = np.linalg.norm(v_left)
            right_len = np.linalg.norm(v_right)
            mean_len = (left_len + right_len) / 2.0 if (left_len + right_len) > 0 else np.nan
            asymmetry = abs(left_len - right_len) / mean_len if mean_len > 0 else np.nan

        metrics.append({
            "frame": i,
            "time_s": i / fps if fps > 0 else np.nan,
            "angle_deg": angle,
            "leaflet_left_len": left_len,
            "leaflet_right_len": right_len,
            "asymmetry": asymmetry,
            "tip_left_x": tracked["tip_left"][0] if tracked["tip_left"] else np.nan,
            "tip_left_y": tracked["tip_left"][1] if tracked["tip_left"] else np.nan,
            "tip_right_x": tracked["tip_right"][0] if tracked["tip_right"] else np.nan,
            "tip_right_y": tracked["tip_right"][1] if tracked["tip_right"] else np.nan,
            "base_left_x": tracked["base_left"][0] if tracked["base_left"] else np.nan,
            "base_left_y": tracked["base_left"][1] if tracked["base_left"] else np.nan,
            "base_right_x": tracked["base_right"][0] if tracked["base_right"] else np.nan,
            "base_right_y": tracked["base_right"][1] if tracked["base_right"] else np.nan,
        })

    # Convert to arrays for smoothing and derivatives
    times = np.array([m["time_s"] for m in metrics], dtype=float)
    angles = np.array([m["angle_deg"] for m in metrics], dtype=float)
    asym = np.array([m["asymmetry"] for m in metrics], dtype=float)

    angles_s = smooth_series(angles, window=smooth_window)
    asym_s = smooth_series(asym, window=smooth_window)

    # Derivative: opening/closing speed (deg/s)
    dt = 1.0 / fps if fps > 0 else np.nan
    speed = np.gradient(angles_s, dt) if fps > 0 else np.full_like(angles_s, np.nan)

    # Determine open/close timing based on threshold of max angle
    max_angle = np.nanmax(angles_s)
    open_thresh = 0.2 * max_angle if np.isfinite(max_angle) else np.nan
    is_open = angles_s >= open_thresh if np.isfinite(open_thresh) else np.zeros_like(angles_s, dtype=bool)
    transitions = np.diff(is_open.astype(int))
    open_times = times[1:][transitions == 1]
    close_times = times[1:][transitions == -1]

    # Save CSV
    csv_path = f"{output_prefix}.csv"
    fieldnames = list(metrics[0].keys()) + ["angle_smooth_deg", "speed_deg_s", "asymmetry_smooth"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, m in enumerate(metrics):
            row = dict(m)
            row["angle_smooth_deg"] = angles_s[i] if i < len(angles_s) else np.nan
            row["speed_deg_s"] = speed[i] if i < len(speed) else np.nan
            row["asymmetry_smooth"] = asym_s[i] if i < len(asym_s) else np.nan
            writer.writerow(row)

    # Save summary metadata
    summary = {
        "fps": fps,
        "num_frames": len(metrics),
        "max_angle_deg": float(max_angle) if np.isfinite(max_angle) else None,
        "open_threshold_deg": float(open_thresh) if np.isfinite(open_thresh) else None,
        "open_times_s": open_times.tolist(),
        "close_times_s": close_times.tolist(),
    }
    with open(f"{output_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(times, angles_s, label="Opening Angle (smoothed)")
    plt.ylabel("Angle (deg)")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(times, speed, label="Opening/Closing Speed")
    plt.ylabel("deg/s")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(times, asym_s, label="Leaflet Asymmetry (smoothed)")
    plt.xlabel("Time (s)")
    plt.ylabel("Asymmetry")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_path = f"{output_prefix}_plots.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved metrics to '{csv_path}'")
    print(f"Saved plots to '{plot_path}'")
    print(f"Saved summary to '{output_prefix}_summary.json'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze valve motion from binary masks.")
    parser.add_argument("--masks", default="predicted_masks", help="Directory of binary masks")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    parser.add_argument("--max-jump", type=float, default=20.0, help="Max pixel jump per frame")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window (frames)")
    parser.add_argument("--out", default="valve_metrics", help="Output prefix for CSV/plots")
    args = parser.parse_args()

    analyze_masks(
        mask_dir=args.masks,
        fps=args.fps,
        max_jump=args.max_jump,
        smooth_window=args.smooth,
        output_prefix=args.out,
    )
