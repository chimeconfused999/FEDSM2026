"""Poster flowcharts: technical (filenames) and overview (project description)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLACK = "#000000"
EDGE = "#1e3a5f"


def box_size(lines, min_w=1.65, char_w=0.108, line_h=0.31):
    max_len = max(len(l) for l in lines)
    w = max(min_w, max_len * char_w + 0.38)
    h = 0.42 + len(lines) * line_h
    return w, h


def draw_box(ax, cx, cy, w, h, lines, facecolor="#eef2ff"):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.05",
            linewidth=1.6, edgecolor=EDGE, facecolor=facecolor, zorder=2,
        )
    )
    pad = 0.10
    step = (h - 2 * pad) / len(lines)
    y0 = cy + h / 2 - pad - step / 2
    for i, line in enumerate(lines):
        fs = 11 if i == 0 else 9.5
        ax.text(
            cx, y0 - i * step, line,
            ha="center", va="center",
            fontsize=fs, fontweight="bold" if i == 0 else "normal",
            color=BLACK, zorder=3,
        )


def h_arrow(ax, x1, x2, y, gap=0.05):
    left = x1 + gap
    right = x2 - gap
    if right <= left:
        return
    ax.add_patch(
        FancyArrowPatch(
            (left, y), (right, y),
            arrowstyle="-|>", mutation_scale=15, linewidth=1.9,
            color=EDGE, zorder=1, shrinkA=0, shrinkB=0,
        )
    )


def v_arrow(ax, x, y1, y2, gap=0.05):
    top = max(y1, y2) - gap
    bot = min(y1, y2) + gap
    if top <= bot:
        return
    ax.add_patch(
        FancyArrowPatch(
            (x, top), (x, bot),
            arrowstyle="-|>", mutation_scale=15, linewidth=1.9,
            color=EDGE, zorder=1, shrinkA=0, shrinkB=0,
        )
    )


def elbow_arrow(ax, x1, y1, x2, y2, mid_y, gap=0.05):
    """Down from (x1,y1), across at mid_y, down into (x2,y2)."""
    y1_out = y1 - gap
    y2_in = y2 + gap
    ax.plot([x1, x1], [y1_out, mid_y], color=EDGE, linewidth=1.9, solid_capstyle="round", zorder=1)
    ax.plot([x1, x2], [mid_y, mid_y], color=EDGE, linewidth=1.9, solid_capstyle="round", zorder=1)
    ax.add_patch(
        FancyArrowPatch(
            (x2, mid_y), (x2, y2_in),
            arrowstyle="-|>", mutation_scale=15, linewidth=1.9,
            color=EDGE, zorder=1, shrinkA=0, shrinkB=0,
        )
    )


def layout_row(ax, boxes, row_y, col_cx, col_ws, arrow_gap):
    """Place up to 3 boxes on one row; return list of (cx, w, h, top, bot)."""
    placed = []
    for i, (color, lines) in enumerate(boxes):
        w, h = box_size(lines)
        w = max(w, col_ws[i])
        cx = col_cx[i]
        draw_box(ax, cx, row_y, w, h, lines, facecolor=color)
        top = row_y + h / 2
        bot = row_y - h / 2
        placed.append((cx, w, h, top, bot))
    for i in range(len(placed) - 1):
        cx1, w1, _, _, _ = placed[i]
        cx2, w2, _, _, _ = placed[i + 1]
        h_arrow(ax, cx1 + w1 / 2, cx2 - w2 / 2, row_y)
    return placed


def column_layout(all_sizes, arrow_gap, margin=0.35):
    """Three aligned columns from two rows of box sizes."""
    col_ws = [max(all_sizes[i][0], all_sizes[i + 3][0]) for i in range(3)]
    total_w = sum(col_ws) + arrow_gap * 2 + margin * 2
    x = margin
    col_cx = []
    for cw in col_ws:
        col_cx.append(x + cw / 2)
        x += cw + arrow_gap
    return col_cx, col_ws, total_w


def make_technical_flowchart():
    out = Path(__file__).resolve().parents[2] / "docs" / "poster" / "pipeline_flowchart.png"

    boxes = [
        ("#dbeafe", ["Input video", "Ultrasound_Venous_Valve.avi", "extractframes.py", "images/"]),
        ("#e0e7ff", ["Expert labels", "masks_annotated/", "binarymasks2.py", "masks_binary2/"]),
        ("#ede9fe", ["Train U-Net", "train_venous.py", "trained_valve_model.pth", "training_history.png"]),
        ("#d1fae5", ["Inference", "predict_masks.py", "predicted_masks/", "metadata.json"]),
        ("#ccfbf1", ["Motion metrics", "valve_motion_analysis.py", "valve_metrics.csv", "valve_metrics_poster.png"]),
        ("#fef3c7", [
            "Validation", "validate_segmentation.py", "validate_geometry.py",
            "validation_segmentation/", "validation_geometry/",
        ]),
    ]

    sizes = [box_size(b[1]) for b in boxes]
    arrow_gap = 0.40
    total_w = sum(s[0] for s in sizes) + arrow_gap * (len(boxes) - 1) + 0.45

    fig_h = 2.55
    fig, ax = plt.subplots(figsize=(max(14.5, total_w * 0.88), fig_h), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    cy = 1.05
    x = 0.22
    centers = []
    for (color, lines), (w, h) in zip(boxes, sizes):
        cx = x + w / 2
        centers.append((cx, w, h))
        draw_box(ax, cx, cy, w, h, lines, facecolor=color)
        x += w + arrow_gap

    for i in range(len(centers) - 1):
        cx1, w1, _ = centers[i]
        cx2, w2, _ = centers[i + 1]
        h_arrow(ax, cx1 + w1 / 2, cx2 - w2 / 2, cy)

    ax.text(
        total_w / 2, fig_h - 0.22,
        "Venous valve analysis pipeline (repository scripts and outputs)",
        ha="center", va="top", fontsize=11.5, fontweight="bold", color=BLACK,
    )

    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {out}")


def make_overview_flowchart():
    out = Path(__file__).resolve().parents[2] / "docs" / "poster" / "pipeline_overview_flowchart.png"

    row1 = [
        ("#dbeafe", [
            "Ultrasound acquisition",
            "Records 2D B-mode ultrasound of a",
            "cadaveric venous valve during flow.",
        ]),
        ("#e0e7ff", [
            "Frame extraction",
            "Splits the cine into individual frames",
            "so each moment can be analyzed.",
        ]),
        ("#ede9fe", [
            "Expert annotation",
            "A clinician outlines valve leaflets on",
            "selected frames to create training labels.",
        ]),
    ]
    row2 = [
        ("#d1fae5", [
            "U-Net segmentation",
            "A deep network learns leaflet shapes",
            "from expert masks using Dice + BCE loss.",
        ]),
        ("#ccfbf1", [
            "Full-sequence inference",
            "The trained model segments the valve",
            "in every frame across the full cine.",
        ]),
        ("#fef3c7", [
            "Quantitative validation",
            "Predictions are compared to experts for",
            "segmentation accuracy, length, and motion.",
        ]),
    ]

    all_boxes = row1 + row2
    sizes = [box_size(b[1], min_w=2.95, char_w=0.088, line_h=0.33) for b in all_boxes]
    arrow_gap = 0.55
    col_cx, col_ws, total_w = column_layout(sizes, arrow_gap)

    row1_h = max(sizes[i][1] for i in range(3))
    row2_h = max(sizes[i][1] for i in range(3, 6))
    row_gap = 0.78
    fig_h = row1_h + row2_h + row_gap + 1.05

    fig, ax = plt.subplots(figsize=(max(11, total_w * 1.05), fig_h), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    row2_y = 0.55 + row2_h / 2
    row1_y = row2_y + row2_h / 2 + row_gap + row1_h / 2

    top_row = layout_row(ax, row1, row1_y, col_cx, col_ws, arrow_gap)
    bot_row = layout_row(ax, row2, row2_y, col_cx, col_ws, arrow_gap)

    cx_from, _, _, _, bot_from = top_row[2]
    cx_to, _, _, top_to, _ = bot_row[0]
    mid_y = (bot_from + top_to) / 2
    elbow_arrow(ax, cx_from, bot_from, cx_to, top_to, mid_y)

    ax.text(
        total_w / 2, fig_h - 0.22,
        "Automated Venous Valve Analysis Pipeline",
        ha="center", va="top", fontsize=12.5, fontweight="bold", color=BLACK,
    )

    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    make_technical_flowchart()
    make_overview_flowchart()


if __name__ == "__main__":
    main()
