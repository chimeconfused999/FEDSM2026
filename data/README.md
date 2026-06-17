# Data layout

| Path | Description |
|------|-------------|
| `data/videos/` | Source ultrasound cine (`Ultrasound_Venous_Valve.avi`) |
| `images/` | Extracted frames (~1,311) from the venous video |
| `masks_annotated/` | Manual clinician tracings (47 expert-reviewed frames) |
| `masks_binary2/` | Filled binary training masks derived from annotations |
| `Common Carotid Artery Ultrasound Images/` | Separate carotid dataset (~1,100 image–mask pairs) |
| `models/` | Trained U-Net checkpoints (`.pth`) |
| `predicted_masks/` | Per-frame model segmentations |
| `validation_segmentation/` | Dice / IoU validation outputs |
| `validation_geometry/` | Geometry validation and Bland–Altman plots |
| `outputs/` | Training curves, motion metrics CSV/JSON, poster figures |
| `cfd_output/` | Optional preliminary geometry / flow-proxy export |

Run `python scripts/utils/check_dataset_safety.py` before training to confirm venous and carotid paths are not mixed.
