"""Path resolution for the Sleep Twin project.

All paths can be overridden via environment variables. The dataset path also
has a sensible fallback to the original location on this machine, so the
project works out of the box without any env setup.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# Dataset: prefer SLEEP_TWIN_DATA_ROOT env var. Otherwise look in (a) the
# project root, (b) a sibling folder, (c) the original AOT Project location
# on this machine. The historical path is hard-coded as a final fallback so
# the project keeps working after being moved out of the AOT Project folder.
_HISTORICAL_DATA_ROOT = Path(
    "C:/Users/user/OneDrive/Desktop/AOT Project/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"
)
_SIBLING_DATA_ROOT = PROJECT_ROOT.parent / "motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"
_LOCAL_DATA_ROOT = PROJECT_ROOT / "motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"


def _resolve_data_root() -> Path:
    env = os.environ.get("SLEEP_TWIN_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    for candidate in (_LOCAL_DATA_ROOT, _SIBLING_DATA_ROOT, _HISTORICAL_DATA_ROOT):
        if candidate.exists():
            return candidate
    return _HISTORICAL_DATA_ROOT


DEFAULT_DATA_ROOT = _resolve_data_root()
DEFAULT_FEATURE_DIR = _path_from_env("SLEEP_TWIN_FEATURE_DIR", PROJECT_ROOT / "artifacts" / "features")
DEFAULT_MODEL_DIR = _path_from_env("SLEEP_TWIN_MODEL_DIR", PROJECT_ROOT / "artifacts" / "models")
DEFAULT_REPORT_DIR = _path_from_env("SLEEP_TWIN_REPORT_DIR", PROJECT_ROOT / "artifacts" / "reports")
