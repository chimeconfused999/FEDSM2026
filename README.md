# FEDSM 2026 — Automated Venous Valve Analysis from 2D B-Mode Ultrasound

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-U--Net-ee4c2c.svg)](https://pytorch.org/)

Research codebase for an **ASAIO / FEDSM 2026 poster**: automatically segment venous valve leaflets in cine ultrasound, quantify motion over the cardiac cycle, and validate predictions against expert tracings.

**Repository:** [github.com/chimeconfused999/FEDSM2026](https://github.com/chimeconfused999/FEDSM2026)

**Author:** Vyom Kumar ([chimeconfused999](https://github.com/chimeconfused999))

---

## Table of contents

1. [Overview](#overview)
2. [What this project does](#what-this-project-does)
3. [Quick start](#quick-start)
4. [Unified application (`app.py`)](#unified-application-apppy)
5. [Repository layout](#repository-layout)
6. [Dataset](#dataset)
7. [Pipeline stages (detailed)](#pipeline-stages-detailed)
8. [Validation results](#validation-results)
9. [Training](#training)
10. [Carotid workflow (optional)](#carotid-workflow-optional)
11. [Poster assets](#poster-assets)
12. [Docker](#docker)
13. [Dataset safety](#dataset-safety)
14. [Poster guidance](#poster-guidance)
15. [Troubleshooting](#troubleshooting)
16. [Citation](#citation)

---

## Overview

Deep vein thrombosis (DVT) and chronic venous disease depend on competent **venous valves**. This project tests whether a **2D B-mode ultrasound cine** of a cadaveric venous valve can be analyzed end-to-end with deep learning:

- **Segment** valve leaflets in every frame (U-Net)
- **Track** opening/closing proxies over time
- **Validate** against 47 expert-annotated frames and a held-out subset
- **Measure** valve length geometry where the pipeline is reliable

The main video is `data/videos/Ultrasound_Venous_Valve.avi` (~1,311 frames, 688×464 px, ~30 fps, ~44 s).

---

## What this project does

```
Ultrasound video  →  frames  →  expert masks  →  train U-Net
       →  predict all frames  →  motion metrics  →  validation
```

| Stage | What happens |
|-------|----------------|
| **Acquisition** | 2D B-mode cine of cadaveric venous valve during pump-driven flow |
| **Frame extraction** | Video split into `images/` for per-frame analysis |
| **Expert annotation** | Clinician outlines leaflets; filled masks in `masks_binary2/` |
| **Segmentation** | U-Net (Dice + BCE) learns leaflet shapes at 256×256 |
| **Inference** | Model segments every frame → `predicted_masks/` |
| **Motion** | Contour-based proxies for opening angle and timing |
| **Validation** | Dice/IoU vs experts; valve length Bland–Altman |

See `docs/poster/pipeline_overview_flowchart.png` for a visual summary.

---

## Quick start

### 1. Clone and install

```powershell
git clone https://github.com/chimeconfused999/FEDSM2026.git
cd FEDSM2026
pip install -r requirements.txt
# Optional editable install (adds `fedsm` CLI):
pip install -e .
```

On Windows with GPU, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/) if needed.

### 2. Run the full venous pipeline

```powershell
python app.py full
```

Equivalent:

```powershell
python run_all.py
```

This runs: **predict masks → motion analysis → segmentation validation → geometry validation → CFD geometry export**.

### 3. Train from scratch (if no checkpoint)

```powershell
python app.py extract
python app.py prepare-masks
python app.py train
python app.py full
```

---

## Unified application (`app.py`)

All workflows are exposed through one CLI:

| Command | Description |
|---------|-------------|
| `python app.py full` | End-to-end venous pipeline |
| `python app.py train` | Train U-Net on `masks_binary2/` |
| `python app.py predict` | Segment video → `predicted_masks/` |
| `python app.py extract` | `data/videos/*.avi` → `images/` |
| `python app.py prepare-masks` | `masks_annotated/` → `masks_binary2/` |
| `python app.py motion` | Temporal metrics from predicted masks |
| `python app.py validate` | Segmentation + geometry validation |
| `python app.py poster` | Regenerate flowchart and dataset table PNGs |
| `python app.py check` | Verify venous/carotid path separation |
| `python app.py carotid` | Optional carotid train + validate |
| `python app.py cfd` | Optional preliminary geometry export |

After `pip install -e .`, you can also run:

```powershell
fedsm full
```

Useful flags:

```powershell
python app.py full --threshold 0.4 --skip-cfd
python app.py train --pretrain models/trained_carotid_model.pth --epochs 50
python app.py predict --video data/videos/Ultrasound_Venous_Valve.avi
```

---

## Repository layout

```
FEDSM2026/
├── app.py                      # Unified CLI entry point
├── run_all.py                  # Full venous pipeline orchestrator
├── pyproject.toml              # Package metadata (`pip install -e .`)
├── requirements.txt
├── fedsm/                      # Core library
│   ├── config.py               # Paths, thresholds, defaults
│   ├── safety.py               # Venous vs carotid guard rails
│   ├── model.py                # U-Net architecture
│   ├── geometry.py             # Mask geometry + metrics
│   └── training.py             # Training loop + dataset
├── scripts/
│   ├── data/                   # extractframes.py, binarymasks2.py
│   ├── train/                  # train_venous.py, train_carotid.py
│   ├── infer/                  # predict_masks.py, valve_motion_analysis.py
│   ├── validate/               # validate_segmentation.py, validate_geometry.py
│   ├── utils/                  # check_dataset_safety.py
│   ├── poster/                 # make_pipeline_flowchart.py, make_dataset_table.py
│   └── optional/               # cfd_preliminary.py, run_carotid.py
├── data/
│   ├── videos/                 # Source ultrasound cine
│   └── README.md               # Data folder guide
├── images/                     # Extracted frames (~1,311)
├── masks_annotated/            # Expert line tracings (47 frames)
├── masks_binary2/              # Filled training masks (all frames)
├── models/                     # trained_valve_model.pth, carotid checkpoint
├── predicted_masks/            # Model predictions per frame
├── validation_segmentation/    # Dice/IoU summaries + overlays
├── validation_geometry/        # Bland–Altman, geometry JSON
├── outputs/                    # Training curves, valve_metrics_*, poster PNGs
├── cfd_output/                 # Optional geometry / flow-proxy export
├── docs/                       # Guides + poster figures
└── website/                    # Project website assets
```

---

## Dataset

| Item | Value |
|------|-------|
| Video | `data/videos/Ultrasound_Venous_Valve.avi` |
| Frames | 1,311 @ 688×464 px, ~30 fps |
| Expert-reviewed annotations | **47** frames in `masks_annotated/` |
| Training labels | Filled regions in `masks_binary2/` (green threshold from annotations) |
| Model input size | 256×256 (resized) |
| Default inference threshold | 0.5 (try 0.4 for higher recall) |

**Important:** Training uses **filled leaflet regions** (`masks_binary2/`), not thin white strokes in `masks_annotated/`. Raw stroke Dice (~0.15) is not a meaningful headline metric.

---

## Pipeline stages (detailed)

### 1. Frame extraction (`scripts/data/extractframes.py`)

Reads the source AVI and writes numbered PNGs to `images/`. Respects dataset safety rules before overwriting protected folders.

### 2. Mask preparation (`scripts/data/binarymasks2.py`)

Converts green-filled expert annotations into binary masks in `masks_binary2/` used for U-Net training.

### 3. Training (`scripts/train/train_venous.py` → `fedsm/training.py`)

- Architecture: `UNetEdgeDetector` (encoder–decoder U-Net, 1-channel sigmoid output)
- Loss: **Dice + BCE**
- Augmentation: flips, rotation, brightness (50% probability)
- Saves: `models/trained_valve_model.pth`, `outputs/training_history.png`

### 4. Inference (`scripts/infer/predict_masks.py`)

Runs the trained model on every video frame. Writes `predicted_masks/frame_XXXX.png` plus `metadata.json` (fps, frame count).

### 5. Motion analysis (`scripts/infer/valve_motion_analysis.py`)

Extracts contour-based keypoints from binary masks and computes temporal proxies (opening angle, valve length over time). Outputs under `outputs/valve_metrics_*`.

### 6. Validation

- **`validate_segmentation.py`** — Dice, IoU, precision, recall on expert subset + 20% holdout
- **`validate_geometry.py`** — Valve length, opening angle, sinus height vs manual (see [Validation results](#validation-results) for what to trust)

### 7. Optional CFD export (`scripts/optional/cfd_preliminary.py`)

Exports valve profile CSVs and a **distance-transform flow proxy** — not Navier–Stokes CFD. Exploratory only.

---

## Validation results

Model: `models/trained_valve_model.pth`, threshold **0.4** (latest segmentation validation).

### Expert-annotated subset (47 frames)

| Metric | Value |
|--------|-------|
| Dice | **0.66** ± 0.02 |
| IoU | **0.49** |
| Precision | **0.89** |
| Recall | **0.52** |

High precision, moderate recall — the model is conservative (under-segments vs expert fills).

### 20% holdout (262 frames, seed 42)

| Metric | Value |
|--------|-------|
| Dice | **0.83** ± 0.01 |
| IoU | **0.71** |

Holdout frames share the training distribution but lack line-by-line expert review; **expert Dice is the stricter poster benchmark**.

### Geometry (expert frames)

| Metric | Trust for poster? | Notes |
|--------|-------------------|-------|
| **Valve length** | **Yes** | ~7–8% mean error |
| Opening angle | No | Large MAE; 2D contour proxy only |
| Sinus height | No | Unreliable |
| Lumen area | No | Systematic bias |

Outputs: `validation_segmentation/segmentation_summary.json`, `validation_geometry/geometry_summary.json`, overlay PNGs.

---

## Training

```powershell
# Venous (default)
python app.py train

# With carotid pretrain
python app.py train --pretrain models/trained_carotid_model.pth

# Direct script
python scripts/train/train_venous.py --epochs 50 --batch-size 4
```

Check data before training:

```powershell
python app.py check
python scripts/utils/check_training_data.py
```

---

## Carotid workflow (optional)

Separate dataset: `Common Carotid Artery Ultrasound Images/` (~1,100 US image / expert lumen mask pairs). Useful for **pretraining**, not the primary venous poster story.

```powershell
python app.py carotid
```

Saves `models/trained_carotid_model.pth`. Carotid masks are **filled lumen** regions, not artery walls. `fedsm/safety.py` blocks carotid scripts from overwriting venous folders.

---

## Poster assets

Regenerate figures:

```powershell
python app.py poster
```

| File | Description |
|------|-------------|
| `docs/poster/pipeline_overview_flowchart.png` | Conceptual 2×3 pipeline (sentences) |
| `docs/poster/pipeline_flowchart.png` | Technical flowchart with script filenames |
| `docs/poster/table3_dataset_summary.png` | Dataset summary table |
| `outputs/training_history.png` | Train/val loss curves |
| `validation_segmentation/segmentation_histograms.png` | Dice/IoU distributions |

---

## Docker

CPU-only image (no CUDA):

```bash
docker build -t fedsm2026 .
docker run --rm -v "$(pwd):/app" fedsm2026 python app.py carotid
```

See `docker-compose.yml` for a compose-based workflow.

---

## Dataset safety

`fedsm/safety.py` enforces:

- Carotid training **cannot** read/write `images/`, `masks_*`, `predicted_masks/`, or `models/trained_valve_model.pth`
- Venous training **cannot** use carotid folders
- Overwriting protected venous directories requires `--confirm-overwrite-venous`

Always run `python app.py check` after cloning on a new machine.

---

## Poster guidance

**Lead with:**

- Segmentation: expert Dice **~0.66**, holdout **~0.83**
- Valve **length** agreement (~7–8% error)
- Example overlay (e.g. `validation_segmentation/overlays/overlay_frame_0062.png`)
- Clear limitations (47 expert frames, 2D B-mode, geometry proxies)

**Do not lead with:**

- CFD velocity fields (flow proxy only)
- Opening-angle Bland–Altman as a primary result
- Carotid segmentation as the main claim
- Holdout Dice alone without expert subset context

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `python` opens VS Code | Use `py app.py full` or full path to `python.exe` |
| `Model not found` | Train first, or place weights in `models/trained_valve_model.pth` |
| CUDA not available | Install GPU PyTorch; training still works on CPU (slow) |
| `REFUSED: protected venous folder` | Pass `--confirm-overwrite-venous` intentionally |
| `git pull` conflicts on `.pth` | `git stash` local weights, pull, restore |

---

## Citation

If you use this code or dataset description, please cite the ASAIO / FEDSM 2026 poster (Vyom Kumar, UTEP) and link this repository.

---

## Additional documentation

- [`data/README.md`](data/README.md) — data folder reference
- [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) — training notes (if present)
- [`docs/CREATE_VIDEO_GUIDE.md`](docs/CREATE_VIDEO_GUIDE.md) — video creation notes (if present)

---

## License

MIT — see repository for details. Ultrasound data is for research use as described in the poster context.
