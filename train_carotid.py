"""
Train U-Net on the Common Carotid Artery dataset only.

This script never reads from or writes to venous valve folders (images/,
masks_binary2/, trained_valve_model.pth, etc.). All paths are locked here.
"""

from dataset_safety import (
    CAROTID_HISTORY,
    CAROTID_IMAGE_DIR,
    CAROTID_MASK_DIR,
    CAROTID_MODEL,
)
from train_valve_features import run_training

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Train U-Net on carotid dataset only.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("CAROTID_EPOCHS", "50")),
        help="Training epochs (env CAROTID_EPOCHS also works)",
    )
    args = parser.parse_args()

    run_training(
        image_dir=CAROTID_IMAGE_DIR,
        mask_dir=CAROTID_MASK_DIR,
        model_save_path=CAROTID_MODEL,
        history_path=CAROTID_HISTORY,
        dataset="carotid",
        epochs=args.epochs,
    )
