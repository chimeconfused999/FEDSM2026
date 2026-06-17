"""Project root and standard directory layout."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data (inputs)
DATA_DIR = ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
IMAGES_DIR = ROOT / "images"
MASKS_ANNOTATED_DIR = ROOT / "masks_annotated"
MASKS_BINARY2_DIR = ROOT / "masks_binary2"
CAROTID_ROOT = ROOT / "Common Carotid Artery Ultrasound Images"

# Models
MODELS_DIR = ROOT / "models"

# Generated outputs
OUTPUTS_DIR = ROOT / "outputs"
PREDICTED_MASKS_DIR = ROOT / "predicted_masks"
VALIDATION_SEGMENTATION_DIR = ROOT / "validation_segmentation"
VALIDATION_GEOMETRY_DIR = ROOT / "validation_geometry"
CFD_OUTPUT_DIR = ROOT / "cfd_output"

# Docs / poster assets
DOCS_DIR = ROOT / "docs"
POSTER_DIR = DOCS_DIR / "poster"

# Scripts
SCRIPTS_DIR = ROOT / "scripts"
