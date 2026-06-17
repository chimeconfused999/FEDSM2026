"""Generate poster Table 3 — dataset & pipeline summary (horizontal, white background)."""

from pathlib import Path

import matplotlib.pyplot as plt


def main():
    out = Path(__file__).resolve().parents[2] / "docs" / "poster" / "table3_dataset_summary.png"

    headers = [
        "Specimen",
        "Video source",
        "Frames",
        "Duration",
        "Resolution",
        "Frame rate",
        "Expert labels",
        "Training masks",
        "Holdout set",
        "Architecture",
        "Loss",
        "Train input",
        "Epochs",
        "Val. threshold",
        "Expert Dice",
        "Holdout Dice",
        "Valve length err.",
    ]

    values = [
        "Cadaveric\nvenous valve",
        "Ultrasound_Venous\nValve.avi",
        "1,311",
        "~43.7 s",
        "688 × 464 px",
        "29.97 fps",
        "47 frames",
        "1,311\n(masks_binary2)",
        "262 frames\n(20%, seed 42)",
        "U-Net",
        "Dice + BCE",
        "256 × 256",
        "50",
        "0.4",
        "0.66 ± 0.02",
        "0.83 ± 0.01",
        "~7.3%",
    ]

    n = len(headers)
    fig_w = max(14, n * 0.95)
    fig, ax = plt.subplots(figsize=(fig_w, 2.6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    ax.set_title(
        "Table 3.  Dataset and pipeline summary",
        fontsize=13,
        fontweight="bold",
        color="#111827",
        pad=14,
        loc="left",
        x=0.02,
    )

    table = ax.table(
        cellText=[values],
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.08, 0.96, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 2.15)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_text_props(fontweight="bold", color="#0f172a")
            cell.set_facecolor("white")
        else:
            cell.set_text_props(color="#1f2937")

    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
