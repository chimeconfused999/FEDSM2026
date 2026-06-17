# CineValve

**Automated venous valve segmentation and motion analysis from 2D B-mode ultrasound.**

CineValve is an open-source PyTorch pipeline that segments venous valve leaflets in ultrasound cine loops, runs full-sequence inference, and evaluates predictions against expert annotations. It is designed for research workflows in venous hemodynamics and valve biomechanics.

**Repository:** [github.com/chimeconfused999/FEDSM2026](https://github.com/chimeconfused999/FEDSM2026)  
**Author:** [Vyom Kumar](https://github.com/chimeconfused999)

---

## Features

- U-Net segmentation (Dice + BCE) on expert-filled masks
- Full-cine inference with per-frame binary masks
- Temporal motion metrics from mask geometry
- Combined validation outputs (segmentation + geometry)
- Optional carotid pretraining dataset in [`carotid/`](carotid/README.md) (separate from venous workflow)
- Dataset safety guards to prevent cross-contamination between venous and carotid data

---

## Quick start

```powershell
git clone https://github.com/chimeconfused999/FEDSM2026.git
cd FEDSM2026
pip install -r requirements.txt
```

Place your ultrasound video at `data/videos/input.avi` (or pass `--video`).

```powershell
python app.py extract
python app.py prepare-masks
python app.py train
python app.py full
```

Or run the full pipeline after training:

```powershell
python app.py full
```

Install as a CLI (optional):

```powershell
pip install -e .
cinevalve full
```

---

## Application commands

| Command | Description |
|---------|-------------|
| `python app.py full` | Predict → motion → validation → optional CFD export |
| `python app.py train` | Train venous U-Net |
| `python app.py predict` | Segment video frames |
| `python app.py extract` | Video → `data/images/` |
| `python app.py prepare-masks` | Annotations → `data/masks/training/` |
| `python app.py motion` | Temporal metrics |
| `python app.py validate` | Segmentation + geometry validation |
| `python app.py check` | Dataset path safety check |
| `python app.py carotid` | Optional carotid workflow ([details](carotid/README.md)) |

---

## Repository layout

```
CineValve/
├── app.py                 # Unified CLI
├── run_all.py             # Full venous pipeline
├── cinevalve/             # Core library (model, training, config, safety)
├── scripts/
│   ├── data/              # Frame extraction, mask preparation
│   ├── train/             # Venous training
│   ├── infer/             # Prediction and motion analysis
│   ├── validate/          # Venous validation
│   ├── utils/             # Safety and data checks
│   └── optional/          # CFD geometry export
├── data/
│   ├── images/            # Extracted frames
│   ├── masks/             # Annotated + training masks
│   └── videos/            # Your input video (add locally)
├── models/venous/         # Venous checkpoint
├── outputs/
│   ├── predictions/       # Per-frame segmentations
│   ├── validation/        # All validation results
│   ├── motion/            # Temporal metrics
│   └── cfd/               # Optional geometry export
└── carotid/               # Optional reference dataset (see carotid/README.md)
```

---

## Pipeline overview

1. **Extract frames** from an ultrasound cine → `data/images/`
2. **Prepare masks** from expert annotations → `data/masks/training/`
3. **Train** a U-Net → `models/venous/trained_valve_model.pth`
4. **Predict** masks for every frame → `outputs/predictions/`
5. **Analyze motion** → `outputs/motion/`
6. **Validate** against experts → `outputs/validation/`

Validation reports Dice, IoU, precision, recall, and geometry comparisons (valve length, opening-angle proxy, etc.). Geometry metrics vary in reliability; segmentation metrics against expert-filled masks are the primary benchmark.

---

## Carotid reference dataset

The [`carotid/`](carotid/README.md) folder contains an **optional** common-carotid-artery ultrasound dataset (~1,100 image–mask pairs) used for pretraining experiments. It is isolated from venous data:

- Carotid masks = **filled lumen**, not vessel wall
- Carotid scripts cannot write to `data/` or `models/venous/`
- Useful for transfer learning: `python app.py train --pretrain carotid/models/trained_carotid_model.pth`

This is a **reference workflow**, not the main venous-valve use case.

---

## Requirements

- Python 3.9+
- PyTorch + torchvision
- OpenCV, Pillow, NumPy, Matplotlib, scikit-image, pandas, tqdm

See [`requirements.txt`](requirements.txt). For GPU training on Windows, install the CUDA PyTorch build from [pytorch.org](https://pytorch.org/).

---

## Dataset safety

`cinevalve/safety.py` blocks accidental overwrites between venous and carotid paths. Run before training:

```powershell
python app.py check
```

Protected venous folders require `--confirm-overwrite-venous` to replace existing predictions.

---

## Outputs

All generated results live under `outputs/`:

| Subfolder | Contents |
|-----------|----------|
| `predictions/` | `frame_XXXX.png` masks + `metadata.json` |
| `validation/segmentation/` | Dice/IoU summaries, overlays, per-frame CSV |
| `validation/geometry/` | Bland–Altman plots, geometry JSON |
| `motion/` | `valve_metrics.csv`, plots, summary JSON |
| `cfd/` | Optional profile CSVs and flow-proxy maps |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Model not found` | Run `python app.py train` or download/place weights in `models/venous/` |
| `Video not found` | Add `data/videos/input.avi` or use `--video path/to/file.avi` |
| `REFUSED: protected folder` | Pass `--confirm-overwrite-venous` when refreshing predictions |
| `python` opens an editor | Use `py app.py full` on Windows |

---

## License

MIT — see repository for details. Ultrasound data is for research use; add your own videos locally.

---

## Citation

If you use CineValve in your work, please cite the repository and credit the author. Suggested name: **CineValve** (cine ultrasound valve analysis).
