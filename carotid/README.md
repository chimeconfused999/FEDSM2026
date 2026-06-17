# Optional carotid artery dataset and training (separate from venous valve workflow)

This folder is **self-contained**. The main CineValve pipeline targets **venous valves**; carotid is included only as an optional pretraining reference on a public-style ultrasound dataset.

## Layout

| Path | Description |
|------|-------------|
| `carotid/dataset/US images/` | B-mode ultrasound frames |
| `carotid/dataset/Expert mask images/` | Expert lumen masks (filled interior, not vessel wall) |
| `carotid/models/` | Carotid U-Net checkpoint |
| `carotid/validation/` | Holdout metrics and overlays |
| `carotid/scripts/` | Train, validate, and run scripts |

## Usage

From the repository root:

```powershell
python app.py carotid
```

Or directly:

```powershell
python carotid/scripts/run_carotid.py
```

Fine-tune on venous data afterward:

```powershell
python app.py train --pretrain carotid/models/trained_carotid_model.pth
```

## Note

Carotid masks segment the **lumen cavity**, not the artery wall. Do not mix carotid folders with `data/images` or `data/masks` — `cinevalve/safety.py` enforces separation.
