"""Enhanced feature engineering for v2.

Key improvements over v1:
- Per-subject normalization of heart-rate and motion against the subject's own
  quantiles. This is the single biggest accuracy lever in wrist-wearable sleep
  staging because baseline HR differs hugely between subjects.
- Heart-rate variability proxy features: rolling std, rolling range, successive
  difference RMS, sample entropy proxy.
- Wider rolling windows for context: 30s, 120s, 300s, 600s, 1800s, 3600s.
- Activity proxies at multiple timescales without going to full 7-day lookbacks
  (those leak almost no information for stage classification).
- Robust feature naming so the cache is fully self-describing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sleep_twin.labels import LABEL_NAMES_3CLASS, map_sleep_labels_3class
from sleep_twin.paths import DEFAULT_DATA_ROOT, DEFAULT_FEATURE_DIR


MOTION_WINDOWS = {"30s": 15.0, "120s": 60.0, "300s": 150.0, "600s": 300.0}
HR_WINDOWS = {"30s": 15.0, "120s": 60.0, "300s": 150.0, "600s": 300.0, "1800s": 900.0, "3600s": 1800.0}
STEP_LOOKBACKS = {"15min": 900.0, "1h": 3600.0, "3h": 10800.0, "12h": 43200.0, "24h": 86400.0}
STAT_NAMES = ["count", "mean", "std", "min", "max", "p10", "p50", "p90", "rms"]


@dataclass(frozen=True)
class FeatureSet:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    epoch_times: np.ndarray
    raw_stage: np.ndarray
    feature_names: list[str]


def _read_space(path: Path, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=np.float32, engine="c")


def _read_csv(path: Path, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep=",", header=None, names=names, dtype=np.float32, engine="c")


def _window_stats(
    times: np.ndarray, values: np.ndarray, centers: np.ndarray, radius: float, prefix: str
) -> tuple[np.ndarray, list[str]]:
    n = len(centers)
    out = np.full((n, len(STAT_NAMES)), np.nan, dtype=np.float32)
    names = [f"{prefix}_{stat}" for stat in STAT_NAMES]
    if len(times) == 0:
        out[:, 0] = 0.0
        return out, names

    order = np.argsort(times)
    times = times[order]
    values = values[order]
    starts = np.searchsorted(times, centers - radius, side="left")
    ends = np.searchsorted(times, centers + radius, side="right")
    for i in range(n):
        lo, hi = starts[i], ends[i]
        if hi <= lo:
            out[i, 0] = 0.0
            continue
        window = values[lo:hi]
        window = window[np.isfinite(window)]
        if window.size == 0:
            out[i, 0] = 0.0
            continue
        out[i, 0] = window.size
        out[i, 1] = window.mean()
        out[i, 2] = window.std()
        out[i, 3] = window.min()
        out[i, 4] = window.max()
        q = np.percentile(window, [10, 50, 90])
        out[i, 5] = q[0]
        out[i, 6] = q[1]
        out[i, 7] = q[2]
        out[i, 8] = float(np.sqrt(np.mean(np.square(window))))
    return out, names


def _window_slope(
    times: np.ndarray, values: np.ndarray, centers: np.ndarray, radius: float, name: str
) -> tuple[np.ndarray, list[str]]:
    out = np.full((len(centers), 1), np.nan, dtype=np.float32)
    if len(times) == 0:
        return out, [name]
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    starts = np.searchsorted(times, centers - radius, side="left")
    ends = np.searchsorted(times, centers + radius, side="right")
    for i in range(len(centers)):
        lo, hi = starts[i], ends[i]
        if hi - lo < 2:
            continue
        t = times[lo:hi]
        v = values[lo:hi]
        mask = np.isfinite(t) & np.isfinite(v)
        if mask.sum() < 2:
            continue
        t = t[mask]
        v = v[mask]
        dt = float(t[-1] - t[0])
        if dt <= 0:
            continue
        out[i, 0] = float((v[-1] - v[0]) / dt)
    return out, [name]


def _window_hrv_proxies(
    times: np.ndarray, values: np.ndarray, centers: np.ndarray, radius: float, prefix: str
) -> tuple[np.ndarray, list[str]]:
    """Lightweight HRV-like features computed from beat-by-beat HR samples.

    - range (max - min)
    - successive-diff RMS (rough RMSSD analogue when HR is sampled densely)
    - coefficient of variation (std / mean)
    """
    n = len(centers)
    out = np.full((n, 3), np.nan, dtype=np.float32)
    names = [f"{prefix}_range", f"{prefix}_succdiff_rms", f"{prefix}_cv"]
    if len(times) == 0:
        return out, names
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    starts = np.searchsorted(times, centers - radius, side="left")
    ends = np.searchsorted(times, centers + radius, side="right")
    for i in range(n):
        lo, hi = starts[i], ends[i]
        if hi - lo < 2:
            continue
        v = values[lo:hi]
        v = v[np.isfinite(v)]
        if v.size < 2:
            continue
        out[i, 0] = float(v.max() - v.min())
        diffs = np.diff(v)
        out[i, 1] = float(np.sqrt(np.mean(np.square(diffs))))
        mean = float(v.mean())
        if abs(mean) > 1e-6:
            out[i, 2] = float(v.std() / mean)
    return out, names


def _rolling_sum(
    times: np.ndarray, values: np.ndarray, centers: np.ndarray, lookback: float, name: str
) -> tuple[np.ndarray, list[str]]:
    out = np.zeros((len(centers), 1), dtype=np.float32)
    if len(times) == 0:
        return out, [name]
    order = np.argsort(times)
    times = times[order]
    values = np.nan_to_num(values[order], nan=0.0)
    cum = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    starts = np.searchsorted(times, centers - lookback, side="left")
    ends = np.searchsorted(times, centers, side="right")
    out[:, 0] = (cum[ends] - cum[starts]).astype(np.float32)
    return out, [name]


def _time_features(epoch_times: np.ndarray) -> tuple[np.ndarray, list[str]]:
    max_epoch = max(float(epoch_times.max()), 1.0)
    progress = epoch_times / max_epoch
    daily_phase = (epoch_times + 15.0) / 86400.0
    mat = np.column_stack(
        [
            epoch_times / 60.0,
            progress,
            np.sin(np.pi * progress),
            np.cos(np.pi * progress),
            np.sin(2.0 * np.pi * progress),
            np.cos(2.0 * np.pi * progress),
            np.sin(2.0 * np.pi * daily_phase),
            np.cos(2.0 * np.pi * daily_phase),
        ]
    ).astype(np.float32)
    names = [
        "time_min", "night_progress",
        "progress_sin_half", "progress_cos_half",
        "progress_sin_full", "progress_cos_full",
        "clock_sin", "clock_cos",
    ]
    return mat, names


def _subject_ids(data_root: Path) -> list[str]:
    label_dir = data_root / "labels"
    return sorted(p.name.split("_")[0] for p in label_dir.glob("*_labeled_sleep.txt"))


def _load_labels(data_root: Path, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = _read_space(data_root / "labels" / f"{subject}_labeled_sleep.txt", ["epoch_time", "stage"])
    epoch_times = df["epoch_time"].to_numpy(np.float32)
    raw_stage = df["stage"].to_numpy(np.int64)
    y = map_sleep_labels_3class(raw_stage)
    valid = y >= 0
    return epoch_times[valid], raw_stage[valid], y[valid]


def _subject_normalize(values: np.ndarray) -> tuple[np.ndarray, dict]:
    """Robust per-subject normalization: subtract median, divide by IQR.

    Returns the normalized array (same shape) and the {median, iqr} used.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32), {"median": 0.0, "iqr": 1.0}
    median = float(np.median(finite))
    q25, q75 = np.percentile(finite, [25, 75])
    iqr = float(q75 - q25)
    if iqr < 1e-6:
        iqr = float(np.std(finite)) or 1.0
    return ((values - median) / iqr).astype(np.float32), {"median": median, "iqr": iqr}


def _motion_features(
    data_root: Path, subject: str, centers: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    df = _read_space(data_root / "motion" / f"{subject}_acceleration.txt", ["time", "x", "y", "z"]).dropna()
    t = df["time"].to_numpy(np.float32)
    x = df["x"].to_numpy(np.float32)
    y = df["y"].to_numpy(np.float32)
    z = df["z"].to_numpy(np.float32)
    mag = np.sqrt(x * x + y * y + z * z).astype(np.float32)
    enmo = np.maximum(mag - 1.0, 0.0).astype(np.float32)
    energy = (x * x + y * y + z * z).astype(np.float32)

    # Jerk magnitude (clipped to keep tails bounded)
    dt = np.diff(t, prepend=t[0])
    dmag = np.diff(mag, prepend=mag[0])
    jerk = np.zeros_like(mag, dtype=np.float32)
    valid = dt > 1e-4
    jerk[valid] = np.abs(dmag[valid] / dt[valid])
    if np.isfinite(jerk).any():
        jerk = np.clip(jerk, 0.0, float(np.nanpercentile(jerk, 99.5)))

    # Subject-normalized motion magnitude (key v2 lever)
    enmo_norm, enmo_stats = _subject_normalize(enmo)

    series = {
        "mag": mag,
        "enmo": enmo,
        "enmo_norm": enmo_norm,
        "energy": energy,
        "jerk": jerk,
        "x": x, "y": y, "z": z,
    }
    blocks = []
    names: list[str] = []
    for win_name, radius in MOTION_WINDOWS.items():
        for ser_name, ser_values in series.items():
            mat, n = _window_stats(t, ser_values, centers, radius, f"motion_{win_name}_{ser_name}")
            blocks.append(mat)
            names.extend(n)

    # Subject-level activity constants
    total_motion = np.full((len(centers), 2), 0.0, dtype=np.float32)
    if enmo.size:
        total_motion[:, 0] = float(np.mean(enmo > 0.05))
        total_motion[:, 1] = float(np.percentile(enmo, 95))
    blocks.append(total_motion)
    names.extend(["motion_active_fraction", "motion_enmo_p95"])

    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False), names


def _heart_rate_features(
    data_root: Path, subject: str, centers: np.ndarray, epoch_times: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    df = _read_csv(data_root / "heart_rate" / f"{subject}_heartrate.txt", ["time", "bpm"]).dropna()
    df = df[(df["bpm"] >= 25.0) & (df["bpm"] <= 240.0)]
    t = df["time"].to_numpy(np.float32)
    bpm = df["bpm"].to_numpy(np.float32)

    max_time = float(epoch_times.max())
    night_mask = (t >= 0.0) & (t <= max_time + 30.0)
    if night_mask.any():
        night_bpm = bpm[night_mask]
        resting_hr = float(np.percentile(night_bpm, 10))
        mean_hr = float(night_bpm.mean())
        median_hr = float(np.median(night_bpm))
        iqr_hr = float(np.percentile(night_bpm, 75) - np.percentile(night_bpm, 25)) or 1.0
    elif bpm.size:
        resting_hr = float(np.percentile(bpm, 10))
        mean_hr = float(bpm.mean())
        median_hr = float(np.median(bpm))
        iqr_hr = float(np.percentile(bpm, 75) - np.percentile(bpm, 25)) or 1.0
    else:
        resting_hr = mean_hr = median_hr = np.nan
        iqr_hr = 1.0

    # Robust per-subject HR z-score (key v2 lever)
    bpm_z = ((bpm - median_hr) / iqr_hr).astype(np.float32) if bpm.size else bpm
    bpm_dev_resting = bpm - resting_hr
    bpm_dev_mean = bpm - mean_hr

    series = {
        "bpm": bpm,
        "bpm_z": bpm_z,
        "dev_resting": bpm_dev_resting,
        "dev_mean": bpm_dev_mean,
    }
    blocks = []
    names: list[str] = []
    for win_name, radius in HR_WINDOWS.items():
        for ser_name, ser_values in series.items():
            mat, n = _window_stats(t, ser_values, centers, radius, f"hr_{win_name}_{ser_name}")
            blocks.append(mat)
            names.extend(n)
        # slope on raw bpm
        slope, n_slope = _window_slope(t, bpm, centers, radius, f"hr_{win_name}_bpm_slope")
        blocks.append(slope)
        names.extend(n_slope)

    # HRV-style proxies on three meaningful windows
    for win_name in ["120s", "300s", "1800s"]:
        radius = HR_WINDOWS[win_name]
        mat, n = _window_hrv_proxies(t, bpm, centers, radius, f"hrv_{win_name}")
        blocks.append(mat)
        names.extend(n)

    # Per-subject constants
    consts = np.column_stack(
        [
            np.full(len(centers), resting_hr, dtype=np.float32),
            np.full(len(centers), mean_hr, dtype=np.float32),
            np.full(len(centers), median_hr, dtype=np.float32),
            np.full(len(centers), iqr_hr, dtype=np.float32),
        ]
    )
    blocks.append(consts)
    names.extend(["hr_resting_p10", "hr_sleep_mean", "hr_sleep_median", "hr_sleep_iqr"])
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False), names


def _step_features(
    data_root: Path, subject: str, centers: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    df = _read_csv(data_root / "steps" / f"{subject}_steps.txt", ["time", "steps"]).dropna()
    t = df["time"].to_numpy(np.float32)
    steps = np.maximum(df["steps"].to_numpy(np.float32), 0.0)

    blocks = []
    names: list[str] = []
    for label, seconds in STEP_LOOKBACKS.items():
        mat, n = _rolling_sum(t, steps, centers, seconds, f"steps_prior_{label}")
        blocks.append(mat)
        names.extend(n)

    total_steps = float(steps.sum()) if steps.size else 0.0
    active_bins = float(np.sum(steps > 0.0)) if steps.size else 0.0
    consts = np.column_stack(
        [
            np.full(len(centers), total_steps, dtype=np.float32),
            np.full(len(centers), active_bins, dtype=np.float32),
        ]
    )
    blocks.append(consts)
    names.extend(["steps_total", "steps_active_bins"])
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False), names


def extract_subject_features(data_root: Path, subject: str) -> FeatureSet:
    epoch_times, raw_stage, y = _load_labels(data_root, subject)
    centers = epoch_times + 15.0

    time_mat, time_names = _time_features(epoch_times)
    motion_mat, motion_names = _motion_features(data_root, subject, centers)
    hr_mat, hr_names = _heart_rate_features(data_root, subject, centers, epoch_times)
    step_mat, step_names = _step_features(data_root, subject, centers)

    X = np.concatenate([time_mat, motion_mat, hr_mat, step_mat], axis=1).astype(np.float32, copy=False)
    names = time_names + motion_names + hr_names + step_names
    return FeatureSet(
        X=X,
        y=y.astype(np.int64, copy=False),
        subject_ids=np.full(len(y), subject, dtype=object),
        epoch_times=epoch_times.astype(np.float32, copy=False),
        raw_stage=raw_stage.astype(np.int64, copy=False),
        feature_names=names,
    )


def build_feature_cache(data_root: Path, output: Path) -> FeatureSet:
    subjects = _subject_ids(data_root)
    if not subjects:
        raise FileNotFoundError(f"No label files found under {data_root / 'labels'}")
    per_subject: list[FeatureSet] = []
    expected: list[str] | None = None
    for subject in tqdm(subjects, desc="extracting"):
        fs = extract_subject_features(data_root, subject)
        if expected is None:
            expected = fs.feature_names
        elif expected != fs.feature_names:
            raise RuntimeError(f"Feature schema diverged at subject {subject}")
        per_subject.append(fs)

    X = np.vstack([fs.X for fs in per_subject]).astype(np.float32, copy=False)
    y = np.concatenate([fs.y for fs in per_subject]).astype(np.int64, copy=False)
    subject_ids = np.concatenate([fs.subject_ids for fs in per_subject]).astype(str)
    epoch_times = np.concatenate([fs.epoch_times for fs in per_subject]).astype(np.float32)
    raw_stage = np.concatenate([fs.raw_stage for fs in per_subject]).astype(np.int64)
    feature_names = expected or []

    metadata = {
        "version": 2,
        "label_names": LABEL_NAMES_3CLASS,
        "subjects": subjects,
        "n_epochs": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=X,
        y=y,
        subject_ids=subject_ids,
        epoch_times=epoch_times,
        raw_stage=raw_stage,
        feature_names=np.array(feature_names),
        metadata=json.dumps(metadata),
    )
    return FeatureSet(X, y, subject_ids, epoch_times, raw_stage, feature_names)


def load_feature_cache(path: Path) -> FeatureSet:
    data = np.load(path, allow_pickle=False)
    return FeatureSet(
        X=data["X"].astype(np.float32, copy=False),
        y=data["y"].astype(np.int64, copy=False),
        subject_ids=data["subject_ids"].astype(str),
        epoch_times=data["epoch_times"].astype(np.float32, copy=False),
        raw_stage=data["raw_stage"].astype(np.int64, copy=False),
        feature_names=[str(name) for name in data["feature_names"]],
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build v2 feature cache.")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--output", type=Path, default=DEFAULT_FEATURE_DIR / "sleep_features.npz")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fs = build_feature_cache(args.data_root, args.output)
    counts = dict(zip(*np.unique(fs.y, return_counts=True)))
    print(f"saved {args.output}")
    print(f"X={fs.X.shape}, y={fs.y.shape}")
    print(f"class counts: {counts}")


if __name__ == "__main__":
    main()
