"""Shared defaults for the venous valve analysis pipeline."""

from fedsm.safety import (
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
DEFAULT_VIDEO = "data/videos/Ultrasound_Venous_Valve.avi"
DEFAULT_MASKS_DIR = "predicted_masks"
DEFAULT_THRESHOLD = 0.5
DEFAULT_IMG_SIZE = (256, 256)
DEFAULT_FPS = 30.0
DEFAULT_PIXEL_TO_MM = 0.1

VALIDATION_SEGMENTATION_OUT = "validation_segmentation"
VALIDATION_GEOMETRY_OUT = "validation_geometry"
CFD_OUTPUT_DIR = "cfd_output"
VALVE_METRICS_PREFIX = "outputs/valve_metrics"
