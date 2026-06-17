"""One-time import patcher for reorganized script layout."""

from pathlib import Path

REPLACEMENTS = [
    ("from model import", "from fedsm.model import"),
    ("from dataset_safety import", "from fedsm.safety import"),
    ("from pipeline_config import", "from fedsm.config import"),
    ("from geometry_utils import", "from fedsm.geometry import"),
    ("from train_valve_features import", "from fedsm.training import"),
    ("from valve_motion_analysis import", "from scripts.infer.valve_motion_analysis import"),
]

ROOT = Path(__file__).resolve().parents[1]

for path in list(ROOT.rglob("*.py")):
    if "archive" in path.parts or path.name == "patch_imports.py":
        continue
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"Patched: {path.relative_to(ROOT)}")
