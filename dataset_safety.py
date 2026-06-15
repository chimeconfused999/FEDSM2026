"""
Guard rails to keep venous valve datasets separate from carotid / other training.

Venous folders are read-only for carotid workflows and require explicit confirmation
before any script overwrites them.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Venous valve project data (do not overwrite from carotid workflows)
# ---------------------------------------------------------------------------
PROTECTED_VENOUS_DIRS = frozenset({
    "images",
    "images2",
    "masks_annotated",
    "masks_binary",
    "masks_binary2",
    "predicted_masks",
})

PROTECTED_VENOUS_MODELS = frozenset({
    "trained_valve_model.pth",
})

VENOUS_IMAGE_DIR = "images"
VENOUS_MASK_DIR = "masks_binary2"
VENOUS_MODEL = "trained_valve_model.pth"
VENOUS_HISTORY = "training_history.png"

# ---------------------------------------------------------------------------
# Carotid dataset (separate tree — never copy into venous folders)
# ---------------------------------------------------------------------------
CAROTID_ROOT = "Common Carotid Artery Ultrasound Images"
CAROTID_IMAGE_DIR = os.path.join(CAROTID_ROOT, "US images")
CAROTID_MASK_DIR = os.path.join(CAROTID_ROOT, "Expert mask images")
CAROTID_MODEL = "trained_carotid_model.pth"
CAROTID_HISTORY = "training_history_carotid.png"


class DatasetSafetyError(SystemExit):
    """Raised when an operation would corrupt venous valve data."""


def norm_rel(path: str | Path) -> str:
    """Normalize a path to a forward-slash relative string when under project root."""
    p = Path(path)
    if not p.is_absolute():
        return str(p).replace("\\", "/")
    try:
        return str(p.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(p.resolve()).replace("\\", "/")


def is_protected_venous_dir(path: str | Path) -> bool:
    rel = norm_rel(path).lower().rstrip("/")
    for protected in PROTECTED_VENOUS_DIRS:
        p = protected.lower()
        if rel == p or rel.startswith(p + "/"):
            return True
    return False


def is_carotid_dir(path: str | Path) -> bool:
    rel = norm_rel(path).lower()
    return "common carotid artery ultrasound images" in rel


def count_files_in_dir(path: str | Path) -> int:
    p = Path(path)
    if not p.is_dir():
        return 0
    return sum(1 for f in p.iterdir() if f.is_file())


def assert_training_config(
    image_dir: str,
    mask_dir: str,
    model_path: str,
    history_path: str,
    dataset: str,
) -> None:
    """
    Validate training paths before any epoch runs.

    dataset: 'venous' | 'carotid'
    """
    img = norm_rel(image_dir)
    mask = norm_rel(mask_dir)
    model_name = Path(model_path).name
    history_name = Path(history_path).name

    if dataset == "carotid":
        if not is_carotid_dir(image_dir):
            raise DatasetSafetyError(
                f"REFUSED: carotid training must read US images from '{CAROTID_IMAGE_DIR}', "
                f"got '{img}'."
            )
        if not is_carotid_dir(mask_dir):
            raise DatasetSafetyError(
                f"REFUSED: carotid training must read masks from '{CAROTID_MASK_DIR}', "
                f"got '{mask}'."
            )
        if model_name in PROTECTED_VENOUS_MODELS:
            raise DatasetSafetyError(
                f"REFUSED: carotid training cannot overwrite venous model '{model_name}'. "
                f"Use '{CAROTID_MODEL}'."
            )
        if model_name != CAROTID_MODEL:
            raise DatasetSafetyError(
                f"REFUSED: carotid training must save to '{CAROTID_MODEL}', got '{model_name}'."
            )
        if history_name == VENOUS_HISTORY:
            raise DatasetSafetyError(
                f"REFUSED: carotid training cannot overwrite '{VENOUS_HISTORY}'. "
                f"Use '{CAROTID_HISTORY}'."
            )
        print("[safety] Carotid training: read-only on carotid folders; venous data untouched.")

    elif dataset == "venous":
        if is_carotid_dir(image_dir) or is_carotid_dir(mask_dir):
            raise DatasetSafetyError(
                "REFUSED: venous training cannot use carotid folders. "
                f"Use --images {VENOUS_IMAGE_DIR} --masks {VENOUS_MASK_DIR}."
            )
        if model_name == CAROTID_MODEL:
            raise DatasetSafetyError(
                f"REFUSED: venous training cannot overwrite carotid model '{CAROTID_MODEL}'."
            )
        print("[safety] Venous training: read-only on image/mask folders; only model/history written.")

    else:
        raise DatasetSafetyError(f"Unknown dataset '{dataset}'. Use 'venous' or 'carotid'.")

    if not Path(image_dir).is_dir():
        raise DatasetSafetyError(f"Image directory not found: {image_dir}")
    if not Path(mask_dir).is_dir():
        raise DatasetSafetyError(f"Mask directory not found: {mask_dir}")


def assert_safe_output_dir(
    output_dir: str | Path,
    confirm_overwrite_venous: bool = False,
    purpose: str = "write files",
) -> None:
    """Block writes into protected venous folders unless explicitly confirmed."""
    if not is_protected_venous_dir(output_dir):
        return

    n = count_files_in_dir(output_dir)
    if not confirm_overwrite_venous:
        raise DatasetSafetyError(
            f"REFUSED: cannot {purpose} in protected venous folder "
            f"'{norm_rel(output_dir)}' ({n} existing files). "
            "Pass --confirm-overwrite-venous only if you intend to replace venous data."
        )
    print(
        f"[safety] WARNING: overwriting protected venous folder "
        f"'{norm_rel(output_dir)}' ({n} files) — confirmed by flag."
    )


def assert_safe_model_output(model_path: str, dataset: str) -> None:
    """Ensure inference / export does not clobber the wrong checkpoint."""
    name = Path(model_path).name
    if dataset == "carotid" and name in PROTECTED_VENOUS_MODELS:
        raise DatasetSafetyError(
            f"REFUSED: carotid workflow cannot write to venous model '{name}'."
        )
