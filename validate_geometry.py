"""
Geometric validation: compare automated vs manual measurements.
Reports MAE, mean % difference, Pearson r, and Bland-Altman plots.
"""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from geometry_utils import extract_frame_geometry, load_binary_mask
from pipeline_config import DEFAULT_MODEL, VALIDATION_GEOMETRY_OUT
from validate_segmentation import collect_manual_pairs, load_model, predict_mask


METRIC_KEYS = [
    "opening_angle_deg",
    "valve_length_px",
    "sinus_height_px",
    "leaflet_curvature_mean",
    "lumen_area_px",
]


def geometry_from_path(path):
    mask = load_binary_mask(path)
    if mask is None:
        return None
    return extract_frame_geometry(mask)


def compare_frame(auto_geom, manual_geom):
    row = {}
    for key in METRIC_KEYS:
        a = auto_geom.get(key, np.nan)
        m = manual_geom.get(key, np.nan)
        row[f"auto_{key}"] = a
        row[f"manual_{key}"] = m
        if np.isfinite(a) and np.isfinite(m) and m != 0:
            row[f"pct_diff_{key}"] = 100.0 * abs(a - m) / abs(m)
        else:
            row[f"pct_diff_{key}"] = np.nan
        row[f"abs_err_{key}"] = abs(a - m) if np.isfinite(a) and np.isfinite(m) else np.nan
    return row


def aggregate_metric(rows, key):
    auto = [r[f"auto_{key}"] for r in rows if np.isfinite(r.get(f"auto_{key}", np.nan))]
    manual = [r[f"manual_{key}"] for r in rows if np.isfinite(r.get(f"manual_{key}", np.nan))]
    paired = [
        (r[f"auto_{key}"], r[f"manual_{key}"])
        for r in rows
        if np.isfinite(r.get(f"auto_{key}", np.nan)) and np.isfinite(r.get(f"manual_{key}", np.nan))
    ]
    if not paired:
        return None

    auto_arr = np.array([p[0] for p in paired])
    manual_arr = np.array([p[1] for p in paired])
    abs_err = np.abs(auto_arr - manual_arr)
    pct = np.where(np.abs(manual_arr) > 1e-6, 100.0 * abs_err / np.abs(manual_arr), np.nan)

    r_val, p_val = stats.pearsonr(auto_arr, manual_arr) if len(paired) >= 3 else (np.nan, np.nan)
    bias = float(np.mean(auto_arr - manual_arr))
    loa = 1.96 * float(np.std(auto_arr - manual_arr))

    return {
        "n_pairs": len(paired),
        "mae": float(np.mean(abs_err)),
        "mean_pct_diff": float(np.nanmean(pct)),
        "pearson_r": float(r_val),
        "pearson_p": float(p_val),
        "bland_altman_bias": bias,
        "bland_altman_loa_lower": bias - loa,
        "bland_altman_loa_upper": bias + loa,
    }


def plot_bland_altman(rows, key, out_path, title):
    paired = [
        (r[f"auto_{key}"], r[f"manual_{key}"])
        for r in rows
        if np.isfinite(r.get(f"auto_{key}", np.nan)) and np.isfinite(r.get(f"manual_{key}", np.nan))
    ]
    if len(paired) < 2:
        return False

    auto = np.array([p[0] for p in paired])
    manual = np.array([p[1] for p in paired])
    mean = (auto + manual) / 2.0
    diff = auto - manual
    bias = np.mean(diff)
    loa = 1.96 * np.std(diff)

    plt.figure(figsize=(6, 5))
    plt.scatter(mean, diff, alpha=0.75, c="#2dd4bf", edgecolors="white")
    plt.axhline(bias, color="red", linestyle="-", label=f"Bias={bias:.3f}")
    plt.axhline(bias + loa, color="gray", linestyle="--", label=f"+1.96 SD")
    plt.axhline(bias - loa, color="gray", linestyle="--", label=f"-1.96 SD")
    plt.xlabel("Mean of auto & manual")
    plt.ylabel("Auto − Manual")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate geometric metrics vs manual masks.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--images", default="images")
    parser.add_argument("--manual-gt", default="masks_binary2", help="Ground-truth masks (training format)")
    parser.add_argument("--annotated-dir", default="masks_annotated", help="Limit to hand-annotated frames")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default=VALIDATION_GEOMETRY_OUT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    model = load_model(args.model, device)
    all_pairs = collect_manual_pairs(args.images, args.manual_gt)
    if os.path.isdir(args.annotated_dir):
        expert_names = set(os.listdir(args.annotated_dir))
        pairs = [p for p in all_pairs if p[0] in expert_names]
    else:
        pairs = all_pairs
    print(f"Geometric validation on {len(pairs)} expert-annotated frames")
    rows = []

    for fname, img_path, gt_path in pairs:
        manual_geom = geometry_from_path(gt_path)
        if manual_geom is None:
            continue

        pred_mask, _ = predict_mask(model, img_path, device, threshold=args.threshold)
        auto_geom = extract_frame_geometry(pred_mask)

        row = {"filename": fname}
        row.update(compare_frame(auto_geom, manual_geom))
        rows.append(row)

    csv_path = os.path.join(args.out, "geometry_comparison.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    metric_summary = {}
    for key in METRIC_KEYS:
        agg = aggregate_metric(rows, key)
        if agg:
            metric_summary[key] = agg
            plot_bland_altman(
                rows, key,
                os.path.join(args.out, f"bland_altman_{key}.png"),
                title=key.replace("_", " ").title(),
            )

    summary = {
        "model_path": args.model,
        "manual_gt_dir": args.manual_gt,
        "num_frames": len(rows),
        "metrics": metric_summary,
        "note": (
            "Automated geometry from model-predicted masks; "
            "manual geometry from expert-traced binary masks."
        ),
    }
    summary_path = os.path.join(args.out, "geometry_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Geometric Validation ({len(rows)} frames) ===")
    for key, agg in metric_summary.items():
        print(f"\n{key}:")
        print(f"  MAE: {agg['mae']:.4f}")
        print(f"  Mean % diff: {agg['mean_pct_diff']:.2f}%")
        print(f"  Pearson r: {agg['pearson_r']:.4f}")
        print(f"  Bland-Altman bias: {agg['bland_altman_bias']:.4f}")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
