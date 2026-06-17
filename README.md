# CineValve

**Open-source deep learning pipeline for venous valve segmentation and motion analysis in 2D B-mode ultrasound cine loops.**

CineValve turns ultrasound video into per-frame valve segmentations, temporal motion metrics, and quantitative validation against expert annotations. It targets researchers in vascular imaging, venous hemodynamics, and ultrasound-based biomechanics — not a single conference or institution.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-U--Net-ee4c2c.svg)](https://pytorch.org/)

**Repository:** [github.com/chimeconfused999/CineValve](https://github.com/chimeconfused999/CineValve)

---

## Why CineValve?

| Problem | What CineValve does |
|---------|---------------------|
| Manual valve tracing is slow | U-Net segments leaflets across an entire cine automatically |
| Frame-by-frame analysis is tedious | One model inference pass over all frames |
| Hard to compare methods | Standard validation outputs (Dice, IoU, geometry) in one place |
| Mixed datasets are risky | Built-in guards keep venous and carotid data separate |

---

## Features

- **Segmentation** — U-Net with Dice + BCE loss on expert-filled masks
- **Full-cine inference** — per-frame binary masks for every timestep
- **Motion analysis** — contour-based temporal metrics from predicted masks
- **Validation** — segmentation and geometry checks vs expert labels
- **Optional carotid pretraining** — separate [`carotid/`](carotid/README.md) reference dataset
- **Single CLI** — `python app.py` for extract, train, predict, validate, and more

---

## Quick start

```bash
git clone https://github.com/chimeconfused999/CineValve.git
cd CineValve
pip install -r requirements.txt
```

Add your ultrasound video (e.g. `data/videos/input.avi`) or pass `--video` to any command.

```bash
python app.py extract        # video → frames
python app.py prepare-masks  # annotations → training masks
python app.py train          # train U-Net
python app.py full           # predict → motion → validate
```

Optional editable install:

```bash
pip install -e .
cinevalve full
```

---

## Commands

| Command | Description |
|---------|-------------|
| `python app.py full` | End-to-end: predict → motion → validation → optional CFD export |
| `python app.py train` | Train venous U-Net on `data/masks/training/` |
| `python app.py predict` | Segment all frames from a video |
| `python app.py extract` | Extract frames to `data/images/` |
| `python app.py prepare-masks` | Build training masks from annotations |
| `python app.py motion` | Temporal valve metrics from predictions |
| `python app.py validate` | Segmentation + geometry validation |
| `python app.py check` | Verify venous/carotid path separation |
| `python app.py carotid` | Optional carotid workflow ([details](carotid/README.md)) |

---

## Project structure

```
├── app.py                      # Unified CLI
├── cinevalve/                  # Core library (model, training, config)
├── scripts/                    # Pipeline scripts (data, train, infer, validate)
├── data/
│   ├── images/                 # Extracted frames
│   ├── masks/annotated/        # Expert tracings
│   ├── masks/training/         # Binary training masks
│   └── videos/                 # Your input video (add locally)
├── models/venous/              # Trained checkpoint
├── outputs/
│   ├── predictions/            # Per-frame segmentations
│   ├── validation/             # Segmentation + geometry results
│   ├── motion/                 # Temporal metrics
│   └── cfd/                    # Optional geometry export
└── carotid/                    # Optional reference dataset
```

---

## Pipeline

1. **Extract** frames from a B-mode cine → `data/images/`
2. **Prepare** expert annotations → `data/masks/training/`
3. **Train** a U-Net → `models/venous/trained_valve_model.pth`
4. **Predict** on every frame → `outputs/predictions/`
5. **Analyze** motion over time → `outputs/motion/`
6. **Validate** vs experts → `outputs/validation/`

Validation includes Dice, IoU, precision, recall, and geometry comparisons. Segmentation agreement with expert-filled masks is the most reliable benchmark; some geometry proxies are exploratory and depend on image quality and annotation style.

---

## Carotid reference dataset

The [`carotid/`](carotid/README.md) folder holds an **optional** common-carotid-artery ultrasound dataset (~1,100 image–mask pairs) for pretraining experiments. It is fully isolated from the venous workflow:

- Masks label the **lumen cavity**, not the vessel wall
- Carotid scripts cannot overwrite venous `data/` or `models/venous/`
- Transfer learning example:

```bash
python app.py train --pretrain carotid/models/trained_carotid_model.pth
```

---

## Requirements

- Python 3.9+
- PyTorch, torchvision, OpenCV, Pillow, NumPy, Matplotlib, scikit-image, pandas, tqdm

See [`requirements.txt`](requirements.txt). For GPU training, install the CUDA PyTorch build from [pytorch.org](https://pytorch.org/).

---

## Dataset safety

`cinevalve/safety.py` prevents accidental cross-writes between venous and carotid paths:

```bash
python app.py check
```

Refreshing protected prediction folders requires `--confirm-overwrite-venous`.

---

## Outputs

| Location | Contents |
|----------|----------|
| `outputs/predictions/` | Per-frame masks + `metadata.json` |
| `outputs/validation/segmentation/` | Dice/IoU summaries, overlays, CSV |
| `outputs/validation/geometry/` | Bland–Altman plots, geometry JSON |
| `outputs/motion/` | `valve_metrics.csv`, plots, summary JSON |
| `outputs/cfd/` | Optional profile CSVs and flow-proxy maps |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Model not found` | Run `python app.py train` or place weights in `models/venous/` |
| `Video not found` | Add `data/videos/input.avi` or pass `--video` |
| `REFUSED: protected folder` | Use `--confirm-overwrite-venous` when replacing predictions |
| `python` opens an editor (Windows) | Use `py app.py full` |

---

## Citation

If you use this software in published work, please cite the repository:

```bibtex
@software{cinevalve,
  title   = {CineValve: Ultrasound Venous Valve Segmentation and Analysis},
  author  = {Kumar, Vyom},
  year    = {2026},
  url     = {https://github.com/chimeconfused999/CineValve}
}
```

---

## License

MIT — ultrasound data is not redistributed; add your own videos and annotations for training.
