from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def make_group_holdout_split(
    groups: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(outer.split(indices, groups=groups))
    val_fraction = val_size / max(1.0 - test_size, 1e-6)
    val_fraction = float(np.clip(val_fraction, 0.05, 0.4))
    inner = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state + 1)
    inner_train, inner_val = next(inner.split(train_val_idx, groups=groups[train_val_idx]))
    return train_val_idx[inner_train], train_val_idx[inner_val], test_idx


def make_group_kfold(groups: np.ndarray, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    kf = GroupKFold(n_splits=n_splits)
    return [(np.asarray(tr), np.asarray(va)) for tr, va in kf.split(np.zeros(len(groups)), groups=groups)]


def split_subject_summary(groups: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    return {
        "train_subjects": sorted(set(groups[train_idx].tolist())),
        "val_subjects": sorted(set(groups[val_idx].tolist())),
        "test_subjects": sorted(set(groups[test_idx].tolist())),
    }
