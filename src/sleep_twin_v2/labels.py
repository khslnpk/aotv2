from __future__ import annotations

import numpy as np


LABEL_NAMES_3CLASS = ["wake", "nrem", "rem"]


def map_sleep_labels_3class(raw_labels: np.ndarray) -> np.ndarray:
    """Map PSG labels to {wake=0, nrem=1, rem=2}, -1 for unmapped."""
    mapped = np.full(raw_labels.shape, -1, dtype=np.int64)
    mapped[raw_labels == 0] = 0
    mapped[np.isin(raw_labels, [1, 2, 3, 4])] = 1
    mapped[raw_labels == 5] = 2
    return mapped
