from __future__ import annotations

import os
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = V2_ROOT.parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


DEFAULT_DATA_ROOT = _path_from_env(
    "SLEEP_TWIN_DATA_ROOT",
    REPO_ROOT / "motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0",
)
DEFAULT_FEATURE_DIR = _path_from_env("SLEEP_TWIN_V2_FEATURE_DIR", V2_ROOT / "artifacts" / "features")
DEFAULT_MODEL_DIR = _path_from_env("SLEEP_TWIN_V2_MODEL_DIR", V2_ROOT / "artifacts" / "models")
DEFAULT_REPORT_DIR = _path_from_env("SLEEP_TWIN_V2_REPORT_DIR", V2_ROOT / "artifacts" / "reports")
