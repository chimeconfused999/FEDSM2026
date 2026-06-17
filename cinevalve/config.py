"""Shared defaults for the venous valve analysis pipeline."""

from cinevalve.safety import (
    CAROTID_HISTORY,
    CAROTID_IMAGE_DIR,
    CAROTID_MASK_DIR,
    CAROTID_MODEL,
    VENOUS_HISTORY,
    VENOUS_IMAGE_DIR,
    VENOUS_MASK_DIR,
    VENOUS_MODEL,
)

DEFAULT_MODEL = VENOUS_MODEL
DEFAULT_VIDEO = "data/videos/input.avi"
DEFAULT_MASKS_DIR = "outputs/predictions"
DEFAULT_THRESHOLD = 0.5
DEFAULT_IMG_SIZE = (256, 256)
DEFAULT_FPS = 30.0
DEFAULT_PIXEL_TO_MM = 0.1

VALIDATION_OUT = "outputs/validation"
VALIDATION_SEGMENTATION_OUT = "outputs/validation/segmentation"
VALIDATION_GEOMETRY_OUT = "outputs/validation/geometry"
CFD_OUTPUT_DIR = "outputs/cfd"
VALVE_METRICS_PREFIX = "outputs/motion/valve_metrics"
