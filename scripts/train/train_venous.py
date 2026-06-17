"""
Train U-Net on venous valve data (images/ + masks_binary2/).

Uses dataset safety checks so carotid paths/models are never touched.
"""

import argparse

from fedsm.safety import VENOUS_HISTORY, VENOUS_IMAGE_DIR, VENOUS_MASK_DIR, VENOUS_MODEL
from fedsm.training import run_training


def main():
    parser = argparse.ArgumentParser(description="Train on venous valve ultrasound data.")
    parser.add_argument("--images", default=VENOUS_IMAGE_DIR)
    parser.add_argument("--masks", default=VENOUS_MASK_DIR)
    parser.add_argument("--model", default=VENOUS_MODEL)
    parser.add_argument("--history", default=VENOUS_HISTORY)
    parser.add_argument("--pretrain", default=None, help="Optional carotid checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-5, help="LR when --pretrain is set")
    args = parser.parse_args()

    lr = args.finetune_lr if args.pretrain else args.lr
    run_training(
        image_dir=args.images,
        mask_dir=args.masks,
        model_save_path=args.model,
        history_path=args.history,
        dataset="venous",
        pretrain_path=args.pretrain,
        epochs=args.epochs,
        learning_rate=lr,
    )


if __name__ == "__main__":
    main()
