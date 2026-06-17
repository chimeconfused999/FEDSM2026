"""Project root and directory layout."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
CAROTID_DIR = ROOT / "carotid"
