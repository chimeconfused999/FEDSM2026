# Data layout

| Path | Description |
|------|-------------|
| `data/videos/` | Place your ultrasound `.avi` here (not shipped in the public repo) |
| `data/images/` | Extracted frames |
| `data/masks/annotated/` | Manual clinician tracings |
| `data/masks/training/` | Binary training masks (filled leaflet regions) |
| `models/venous/` | Venous U-Net weights |
| `outputs/predictions/` | Per-frame segmentations |
| `outputs/validation/` | Segmentation and geometry validation results |
| `outputs/motion/` | Temporal valve metrics |
| `outputs/cfd/` | Optional geometry export |

See `carotid/README.md` for the optional carotid reference dataset.
