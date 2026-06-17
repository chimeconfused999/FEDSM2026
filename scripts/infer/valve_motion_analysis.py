import os
import json
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize


def load_binary_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


def preprocess_mask(mask, close_px=5):
    if close_px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return (closed > 0).astype(np.uint8)


def point_on_border(point, width, height, margin):
    if point is None:
        return True
    x, y = point
    return x < margin or y < margin or x >= width - margin or y >= height - margin


def find_skeleton_endpoints(binary_mask):
    skel = skeletonize(binary_mask > 0).astype(np.uint8)
    h, w = skel.shape
    endpoints = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] == 0:
                continue
            neighborhood = skel[y - 1 : y + 2, x - 1 : x + 2]
            count = int(neighborhood.sum()) - 1
            if count == 1:
                endpoints.append((x, y))
    return skel, endpoints


def _validate_keypoints(
    tip_left,
    tip_right,
    base_left,
    base_right,
    min_leaflet_len,
    max_leaflet_len,
    min_tip_sep,
    min_tip_above_base,
):
    base_mid = (base_left + base_right) / 2.0
    if np.linalg.norm(tip_left - tip_right) < min_tip_sep:
        return None
    if tip_left[1] > base_mid[1] - min_tip_above_base:
        return None
    if tip_right[1] > base_mid[1] - min_tip_above_base:
        return None

    v_left = tip_left - base_mid
    v_right = tip_right - base_mid
    left_len = float(np.linalg.norm(v_left))
    right_len = float(np.linalg.norm(v_right))
    if left_len < min_leaflet_len or right_len < min_leaflet_len:
        return None
    if left_len > max_leaflet_len or right_len > max_leaflet_len:
        return None

    return {
        "tip_left": (float(tip_left[0]), float(tip_left[1])),
        "tip_right": (float(tip_right[0]), float(tip_right[1])),
        "base_left": (float(base_left[0]), float(base_left[1])),
        "base_right": (float(base_right[0]), float(base_right[1])),
    }


def classify_keypoints_contour(
    binary_mask,
    edge_margin=12,
    min_foreground=100,
    min_leaflet_len=30,
    max_leaflet_len=350,
    min_tip_sep=40,
    min_tip_above_base=25,
):
    if int(binary_mask.sum()) < min_foreground:
        return None

    h, w = binary_mask.shape
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    pts = max(contours, key=cv2.contourArea).reshape(-1, 2)
    pts = pts[
        (pts[:, 0] >= edge_margin)
        & (pts[:, 0] < w - edge_margin)
        & (pts[:, 1] >= edge_margin)
        & (pts[:, 1] < h - edge_margin)
    ]
    if len(pts) < 20:
        return None

    cx = float(np.median(pts[:, 0]))
    left_pts = pts[pts[:, 0] < cx]
    right_pts = pts[pts[:, 0] >= cx]
    if len(left_pts) < 5 or len(right_pts) < 5:
        return None

    tip_left = left_pts[np.argmin(left_pts[:, 1])].astype(np.float32)
    tip_right = right_pts[np.argmin(right_pts[:, 1])].astype(np.float32)
    base_left = left_pts[np.argmax(left_pts[:, 1])].astype(np.float32)
    base_right = right_pts[np.argmax(right_pts[:, 1])].astype(np.float32)

    return _validate_keypoints(
        tip_left, tip_right, base_left, base_right,
        min_leaflet_len, max_leaflet_len, min_tip_sep, min_tip_above_base,
    )


def classify_keypoints_skeleton(
    binary_mask,
    edge_margin=12,
    min_foreground=100,
    min_leaflet_len=30,
    max_leaflet_len=350,
    min_tip_sep=40,
    min_tip_above_base=25,
):
    if int(binary_mask.sum()) < min_foreground:
        return None

    h, w = binary_mask.shape
    _, endpoints = find_skeleton_endpoints(binary_mask)
    endpoints = [e for e in endpoints if not point_on_border(e, w, h, edge_margin)]
    if len(endpoints) < 4:
        return None

    coords = np.array(endpoints, dtype=np.float32)
    centroid = coords.mean(axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)
    tips = coords[np.argsort(distances)[-2:]]
    bases = coords[np.argsort(distances)[:2]]
    tip_left, tip_right = tips[np.argsort(tips[:, 0])]
    base_left, base_right = bases[np.argsort(bases[:, 0])]

    return _validate_keypoints(
        tip_left, tip_right, base_left, base_right,
        min_leaflet_len, max_leaflet_len, min_tip_sep, min_tip_above_base,
    )


def classify_keypoints(binary_mask, **kwargs):
    """Contour-based detection for filled masks; skeleton fallback for thin edges."""
    return classify_keypoints_contour(binary_mask, **kwargs) or classify_keypoints_skeleton(
        binary_mask, **kwargs
    )


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


def reject_implausible_angle(angle, min_angle_deg, max_angle_deg):
    if not np.isfinite(angle):
        return np.nan
    if angle < min_angle_deg or angle > max_angle_deg:
        return np.nan
    return float(angle)


def smooth_series(values, window=5, method="median"):
    if window <= 1:
        return np.asarray(values, dtype=float)
    values = np.asarray(values, dtype=float)
    if method == "median":
        out = values.copy()
        finite = np.isfinite(values)
        if not finite.any():
            return out
        filled = values.copy()
        idx = np.arange(len(values))
        filled[~finite] = np.interp(idx[~finite], idx[finite], values[finite])
        smoothed = median_filter(filled, size=window, mode="nearest")
        smoothed[~finite] = np.nan
        for i in np.where(finite)[0]:
            half = window // 2
            chunk = values[max(0, i - half) : min(len(values), i + half + 1)]
            chunk = chunk[np.isfinite(chunk)]
            if len(chunk):
                smoothed[i] = float(np.median(chunk))
            else:
                smoothed[i] = np.nan
        return smoothed

    out = np.full_like(values, np.nan, dtype=float)
    half = window // 2
    for i in range(len(values)):
        chunk = values[max(0, i - half) : min(len(values), i + half + 1)]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk):
            out[i] = float(np.mean(chunk))
    return out


def reject_speed_outliers(values, fps, max_speed_deg_s):
    if fps <= 0 or max_speed_deg_s <= 0:
        return values
    out = np.asarray(values, dtype=float).copy()
    dt = 1.0 / fps
    speed = np.gradient(out, dt)
    out[np.abs(speed) > max_speed_deg_s] = np.nan
    return out


def fill_short_gaps(values, max_gap=2):
    out = np.asarray(values, dtype=float).copy()
    n = len(out)
    i = 0
    while i < n:
        if np.isfinite(out[i]):
            i += 1
            continue
        start = i
        while i < n and not np.isfinite(out[i]):
            i += 1
        gap_len = i - start
        left = out[start - 1] if start > 0 else np.nan
        right = out[i] if i < n else np.nan
        if gap_len <= max_gap and np.isfinite(left) and np.isfinite(right):
            out[start:i] = np.linspace(left, right, gap_len + 2)[1:-1]
    return out


def prepare_plot_series(values, fps, gap_fill_s=0.5, smooth_window=21):
    """Interpolate short gaps and smooth for display only (not validation)."""
    max_gap = max(3, int(gap_fill_s * fps))
    out = fill_short_gaps(np.asarray(values, dtype=float), max_gap=max_gap)
    out = smooth_series(out, window=smooth_window, method="median")
    return fill_short_gaps(out, max_gap=max_gap)


def compute_segment_speed(values, fps, max_speed_deg_s=80.0):
    """Speed from contiguous segments only — avoids spikes at gaps."""
    out = np.full(len(values), np.nan, dtype=float)
    if fps <= 0:
        return out
    dt = 1.0 / fps
    i = 0
    values = np.asarray(values, dtype=float)
    while i < len(values):
        if not np.isfinite(values[i]):
            i += 1
            continue
        j = i + 1
        while j < len(values) and np.isfinite(values[j]):
            j += 1
        seg = values[i:j]
        if len(seg) >= 2:
            sp = np.gradient(seg, dt)
            if max_speed_deg_s > 0:
                sp = np.where(np.abs(sp) <= max_speed_deg_s, sp, np.nan)
            out[i:j] = sp
        i = j if j > i else i + 1
    return out


def plot_broken_line(ax, times, values, fps, color="#2563eb", linewidth=1.8, label=None):
    """Plot continuous segments separately so gaps are not connected."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    max_dt = 2.5 / fps if fps > 0 else np.inf
    label_used = False
    start = 0
    for i in range(1, len(values) + 1):
        end_segment = i >= len(values)
        if not end_segment:
            if not np.isfinite(values[i]) or not np.isfinite(values[i - 1]):
                end_segment = True
            elif times[i] - times[i - 1] > max_dt:
                end_segment = True
        if end_segment:
            xs = times[start:i]
            ys = values[start:i]
            mask = np.isfinite(ys)
            if mask.sum() >= 2:
                ax.plot(
                    xs[mask], ys[mask], color=color, linewidth=linewidth,
                    label=label if not label_used else None,
                )
                label_used = True
            start = i
            while start < len(values) and not np.isfinite(values[start]):
                start += 1


def prepare_poster_series(values, vmin=None, vmax=None, savgol_window=51):
    """Fully interpolated + heavy Savitzky-Golay smooth for poster display only."""
    values = np.asarray(values, dtype=float)
    if vmin is not None or vmax is not None:
        lo = vmin if vmin is not None else -np.inf
        hi = vmax if vmax is not None else np.inf
        values = np.where((values >= lo) & (values <= hi), values, np.nan)

    idx = np.arange(len(values))
    finite = np.isfinite(values)
    if finite.sum() < 8:
        return values

    filled = np.interp(idx, idx[finite], values[finite])
    win = min(savgol_window, len(filled) // 6 * 2 + 1)
    win = max(win, 7)
    if win % 2 == 0:
        win += 1
    if win >= len(filled):
        win = len(filled) - 1 if len(filled) % 2 == 0 else len(filled)
        if win < 5:
            return filled

    smoothed = savgol_filter(filled, window_length=win, polyorder=3)
    if vmin is not None or vmax is not None:
        lo = vmin if vmin is not None else np.nanmin(smoothed)
        hi = vmax if vmax is not None else np.nanmax(smoothed)
        smoothed = np.clip(smoothed, lo, hi)
    return smoothed


def _style_poster_axes(ax, labelsize=15):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=labelsize, width=1.2)


def save_poster_motion_plot(times, angles_raw, asym_raw, output_path, dpi=300):
    """Single clean figure for poster: opening-angle proxy + leaflet asymmetry."""
    times = np.asarray(times, dtype=float)
    angle_poster = prepare_poster_series(angles_raw, vmin=0, vmax=70, savgol_window=61)
    asym_poster = prepare_poster_series(asym_raw, vmin=0, vmax=0.45, savgol_window=41)

    fig = plt.figure(figsize=(10, 6.2), dpi=dpi)
    fig.patch.set_facecolor("white")

    # Layout: title | plot A | plot B | x-label band | caption band
    gs = fig.add_gridspec(
        nrows=5,
        ncols=1,
        height_ratios=[0.40, 2.30, 1.10, 0.18, 0.52],
        hspace=0.42,
        left=0.12,
        right=0.94,
        top=0.96,
        bottom=0.08,
    )

    ax_title = fig.add_subplot(gs[0])
    ax_title.axis("off")
    ax_title.text(
        0.5,
        0.55,
        "Valve motion over cardiac cycle",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color="#111827",
        transform=ax_title.transAxes,
    )
    ax_title.text(
        0.5,
        0.05,
        "1,311 frames  ·  ~44 s  ·  single cadaveric B-mode sequence",
        ha="center",
        va="center",
        fontsize=12,
        color="#4b5563",
        transform=ax_title.transAxes,
    )

    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)

    ax1.fill_between(times, 0, angle_poster, color="#3b82f6", alpha=0.16, linewidth=0, zorder=1)
    ax1.plot(times, angle_poster, color="#1d4ed8", linewidth=2.6, zorder=3)
    ax1.set_ylabel("Opening-angle\nproxy (deg)", fontsize=14, fontweight="bold", labelpad=8)
    ax1.set_ylim(0, 78)
    ax1.set_xlim(times[0], times[-1])
    ax1.text(0.015, 0.97, "A", transform=ax1.transAxes, fontsize=15, fontweight="bold", va="top", ha="left")
    _style_poster_axes(ax1, labelsize=13)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.plot(times, asym_poster, color="#047857", linewidth=2.4, zorder=3)
    ax2.fill_between(times, 0, asym_poster, color="#10b981", alpha=0.14, linewidth=0, zorder=1)
    ax2.set_ylabel("Leaflet\nasymmetry", fontsize=14, fontweight="bold", labelpad=8)
    ax2.set_ylim(0, 0.44)
    ax2.set_xlim(times[0], times[-1])
    ax2.text(0.015, 0.97, "B", transform=ax2.transAxes, fontsize=15, fontweight="bold", va="top", ha="left")
    _style_poster_axes(ax2, labelsize=13)

    ax_xlabel = fig.add_subplot(gs[3])
    ax_xlabel.axis("off")
    ax_xlabel.text(
        0.5,
        0.5,
        "Time (s)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
        transform=ax_xlabel.transAxes,
    )

    ax_caption = fig.add_subplot(gs[4])
    ax_caption.axis("off")
    ax_caption.text(
        0.5,
        0.72,
        "Figure X.  Temporal metrics derived from automated valve segmentation.",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#111827",
        transform=ax_caption.transAxes,
    )
    ax_caption.text(
        0.5,
        0.22,
        "2D contour proxy from predicted masks — illustrative trend only, not clinically calibrated.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4b5563",
        style="italic",
        transform=ax_caption.transAxes,
    )

    fig.align_ylabels([ax1, ax2])
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def analyze_masks(
    mask_dir,
    fps=30.0,
    max_jump=8,
    smooth_window=15,
    output_prefix="valve_metrics",
    morph_close=5,
    edge_margin=12,
    min_angle_deg=0.5,
    max_angle_deg=75.0,
    max_speed_deg_s=120.0,
    open_thresh_pct=0.35,
    strict=True,
):
    if not strict:
        max_jump = max(max_jump, 20)
        smooth_window = min(smooth_window, 5)
        max_angle_deg = 180.0
        max_speed_deg_s = 0.0

    mask_files = sorted(
        f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")

    metrics = []
    prev = {k: None for k in ["tip_left", "tip_right", "base_left", "base_right"]}
    hold_streak = 0
    max_hold_frames = 4 if strict else 0
    min_hold_fg = 8000

    for i, fname in enumerate(mask_files):
        mask_path = os.path.join(mask_dir, fname)
        mask = load_binary_mask(mask_path)
        if mask is None:
            continue

        mask = preprocess_mask(mask, close_px=morph_close)
        fg_pixels = int(mask.sum())
        keypoints = classify_keypoints(mask, edge_margin=edge_margin)
        if keypoints is None and max_hold_frames > 0:
            if all(prev[k] is not None for k in prev) and fg_pixels >= min_hold_fg and hold_streak < max_hold_frames:
                keypoints = {k: prev[k] for k in prev}
                hold_streak += 1
            else:
                hold_streak = 0
                keypoints = {k: None for k in prev.keys()}
        elif keypoints is None:
            keypoints = {k: None for k in prev.keys()}
            hold_streak = 0
        else:
            hold_streak = 0

        tracked = {}
        for k in prev.keys():
            tracked[k] = match_points(prev[k], keypoints.get(k), max_jump=max_jump)

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
            angle = reject_implausible_angle(angle, min_angle_deg, max_angle_deg)
            left_len = np.linalg.norm(v_left)
            right_len = np.linalg.norm(v_right)
            mean_len = (left_len + right_len) / 2.0 if (left_len + right_len) > 0 else np.nan
            asymmetry = abs(left_len - right_len) / mean_len if mean_len > 0 else np.nan

        prev = tracked

        metrics.append(
            {
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
            }
        )

    times = np.array([m["time_s"] for m in metrics], dtype=float)
    angles = np.array([m["angle_deg"] for m in metrics], dtype=float)
    asym = np.array([m["asymmetry"] for m in metrics], dtype=float)

    angles_s = smooth_series(angles, window=smooth_window, method="median")
    angles_s = reject_speed_outliers(angles_s, fps, max_speed_deg_s)
    angles_s = smooth_series(angles_s, window=max(3, smooth_window // 3), method="median")
    angles_s = fill_short_gaps(angles_s, max_gap=2 if strict else 0)
    asym_s = smooth_series(asym, window=smooth_window, method="median")
    asym_plot = prepare_plot_series(asym_s, fps, gap_fill_s=0.5, smooth_window=21)

    angles_plot = prepare_plot_series(angles_s, fps, gap_fill_s=0.5, smooth_window=21)
    speed = compute_segment_speed(angles_plot, fps, max_speed_deg_s=80.0)

    valid_angles = angles_s[np.isfinite(angles_s)]
    if len(valid_angles):
        ref_angle = float(np.nanpercentile(valid_angles, 95))
        open_thresh = open_thresh_pct * ref_angle
        max_angle = float(np.nanmax(valid_angles))
    else:
        ref_angle = np.nan
        open_thresh = np.nan
        max_angle = np.nan

    is_open = angles_s >= open_thresh if np.isfinite(open_thresh) else np.zeros_like(angles_s, dtype=bool)
    transitions = np.diff(is_open.astype(int))
    open_times = times[1:][transitions == 1]
    close_times = times[1:][transitions == -1]

    csv_path = f"{output_prefix}.csv"
    fieldnames = list(metrics[0].keys()) + [
        "angle_smooth_deg", "angle_plot_deg", "speed_deg_s", "asymmetry_smooth", "asymmetry_plot",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, m in enumerate(metrics):
            row = dict(m)
            row["angle_smooth_deg"] = angles_s[i] if i < len(angles_s) else np.nan
            row["angle_plot_deg"] = angles_plot[i] if i < len(angles_plot) else np.nan
            row["speed_deg_s"] = speed[i] if i < len(speed) else np.nan
            row["asymmetry_smooth"] = asym_s[i] if i < len(asym_s) else np.nan
            row["asymmetry_plot"] = asym_plot[i] if i < len(asym_plot) else np.nan
            writer.writerow(row)

    summary = {
        "fps": fps,
        "num_frames": len(metrics),
        "strict_mode": strict,
        "max_angle_deg": float(max_angle) if np.isfinite(max_angle) else None,
        "p95_angle_deg": float(ref_angle) if np.isfinite(ref_angle) else None,
        "open_threshold_deg": float(open_thresh) if np.isfinite(open_thresh) else None,
        "valid_angle_frames": int(np.isfinite(angles_s).sum()),
        "open_times_s": open_times.tolist(),
        "close_times_s": close_times.tolist(),
    }
    with open(f"{output_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(10, 8))
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    ax1 = plt.subplot(3, 1, 1)
    plot_broken_line(ax1, times, angles_plot, fps, label="Opening angle (display)")
    if np.isfinite(max_angle_deg):
        ax1.set_ylim(0, min(max_angle_deg + 10, 85))
    ax1.set_ylabel("Angle (deg)")
    ax1.legend(loc="upper right")

    ax2 = plt.subplot(3, 1, 2)
    plot_broken_line(ax2, times, speed, fps, color="#dc2626", label="Opening/closing speed")
    ax2.set_ylim(-90, 90)
    ax2.set_ylabel("deg/s")
    ax2.legend(loc="upper right")

    ax3 = plt.subplot(3, 1, 3)
    plot_broken_line(ax3, times, asym_plot, fps, color="#059669", label="Leaflet asymmetry")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Asymmetry")
    ax3.set_ylim(0, max(0.5, float(np.nanpercentile(asym_plot, 99)) * 1.1))
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plot_path = f"{output_prefix}_plots.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    poster_path = f"{output_prefix}_poster.png"
    save_poster_motion_plot(times, angles_s, asym_s, poster_path, dpi=300)

    print(f"Saved metrics to '{csv_path}'")
    print(f"Saved plots to '{plot_path}'")
    print(f"Saved poster figure to '{poster_path}'")
    print(f"Saved summary to '{output_prefix}_summary.json'")
    if strict:
        max_txt = f"{summary['max_angle_deg']:.1f}" if summary["max_angle_deg"] is not None else "n/a"
        p95_txt = f"{summary['p95_angle_deg']:.1f}" if summary["p95_angle_deg"] is not None else "n/a"
        print(
            f"Strict mode: {summary['valid_angle_frames']}/{len(metrics)} frames with valid angle, "
            f"max={max_txt} deg, p95={p95_txt} deg"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze valve motion from binary masks.")
    parser.add_argument("--masks", default="predicted_masks", help="Directory of binary masks")
    parser.add_argument("--fps", type=float, default=29.97, help="Frames per second")
    parser.add_argument("--max-jump", type=float, default=8.0, help="Max pixel jump per frame")
    parser.add_argument("--smooth", type=int, default=15, help="Median smoothing window (frames)")
    parser.add_argument("--morph-close", type=int, default=5, help="Morphological close radius (px)")
    parser.add_argument("--edge-margin", type=int, default=12, help="Reject keypoints near image border")
    parser.add_argument("--max-angle", type=float, default=75.0, help="Reject angles above this (deg)")
    parser.add_argument("--max-speed", type=float, default=120.0, help="Reject angle changes faster than this (deg/s)")
    parser.add_argument("--no-strict", action="store_true", help="Use legacy loose tracking")
    parser.add_argument("--out", default="valve_metrics", help="Output prefix for CSV/plots")
    args = parser.parse_args()

    analyze_masks(
        mask_dir=args.masks,
        fps=args.fps,
        max_jump=args.max_jump,
        smooth_window=args.smooth,
        output_prefix=args.out,
        morph_close=args.morph_close,
        edge_margin=args.edge_margin,
        max_angle_deg=args.max_angle,
        max_speed_deg_s=args.max_speed,
        strict=not args.no_strict,
    )
