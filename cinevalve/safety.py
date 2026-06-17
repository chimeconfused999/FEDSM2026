"""
Guard rails to keep venous valve datasets separate from carotid workflows.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROTECTED_VENOUS_DIRS = frozenset({
    "data/images",
    "data/masks",
    "outputs/predictions",
})

PROTECTED_VENOUS_MODELS = frozenset({
    "trained_valve_model.pth",
})

VENOUS_IMAGE_DIR = "data/images"
VENOUS_MASK_DIR = "data/masks/training"
VENOUS_MODEL = "models/venous/trained_valve_model.pth"
VENOUS_HISTORY = "outputs/training_history.png"

CAROTID_ROOT = "carotid/dataset"
CAROTID_IMAGE_DIR = os.path.join(CAROTID_ROOT, "US images")
CAROTID_MASK_DIR = os.path.join(CAROTID_ROOT, "Expert mask images")
CAROTID_MODEL = "carotid/models/trained_carotid_model.pth"
CAROTID_HISTORY = "carotid/outputs/training_history.png"


class DatasetSafetyError(SystemExit):
    """Raised when an operation would corrupt venous valve data."""


def norm_rel(path: str | Path) -> str:
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
    return rel.startswith("carotid/")


def count_files_in_dir(path: str | Path) -> int:
    p = Path(path)
    if not p.is_dir():
        return 0
    return sum(1 for f in p.iterdir() if f.is_file())


def _model_basename(model_path: str) -> str:
    return Path(model_path).name


def assert_training_config(
    image_dir: str,
    mask_dir: str,
    model_path: str,
    history_path: str,
    dataset: str,
) -> None:
    img = norm_rel(image_dir)
    mask = norm_rel(mask_dir)
    model_name = _model_basename(model_path)
    history_name = Path(history_path).name

    if dataset == "carotid":
        if not is_carotid_dir(image_dir):
            raise DatasetSafetyError(
                f"REFUSED: carotid training must read from '{CAROTID_IMAGE_DIR}', got '{img}'."
            )
        if not is_carotid_dir(mask_dir):
            raise DatasetSafetyError(
                f"REFUSED: carotid training must read from '{CAROTID_MASK_DIR}', got '{mask}'."
            )
        if model_name in PROTECTED_VENOUS_MODELS:
            raise DatasetSafetyError(
                f"REFUSED: carotid training cannot overwrite venous model '{model_name}'."
            )
        if norm_rel(model_path) != norm_rel(CAROTID_MODEL):
            raise DatasetSafetyError(
                f"REFUSED: carotid training must save to '{CAROTID_MODEL}', got '{model_path}'."
            )
        print("[safety] Carotid training: venous data untouched.")

    elif dataset == "venous":
        if is_carotid_dir(image_dir) or is_carotid_dir(mask_dir):
            raise DatasetSafetyError(
                "REFUSED: venous training cannot use carotid folders. "
                f"Use --images {VENOUS_IMAGE_DIR} --masks {VENOUS_MASK_DIR}."
            )
        if _model_basename(CAROTID_MODEL) == model_name and "carotid" in norm_rel(model_path):
            raise DatasetSafetyError("REFUSED: venous training cannot overwrite carotid model.")
        print("[safety] Venous training: image/mask folders read-only; model/history written.")

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
    if not is_protected_venous_dir(output_dir):
        return
    n = count_files_in_dir(output_dir)
    if not confirm_overwrite_venous:
        raise DatasetSafetyError(
            f"REFUSED: cannot {purpose} in protected venous folder "
            f"'{norm_rel(output_dir)}' ({n} existing files). "
            "Pass --confirm-overwrite-venous only if you intend to replace venous data."
        )
    print(f"[safety] WARNING: overwriting '{norm_rel(output_dir)}' ({n} files).")


def assert_safe_model_output(model_path: str, dataset: str) -> None:
    name = _model_basename(model_path)
    if dataset == "carotid" and name in PROTECTED_VENOUS_MODELS:
        raise DatasetSafetyError(
            f"REFUSED: carotid workflow cannot write to venous model '{name}'."
        )
