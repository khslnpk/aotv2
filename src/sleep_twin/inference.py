"""End-to-end inference service for the Sleep Twin app.

Loads the cached per-base-learner probabilities and the trained ensemble bundle
once at startup, then serves per-subject analyses on demand: predicted hypnogram
(geometric-mean ensemble + HMM smoothing — the project's headline configuration),
sleep architecture summary, physiological summary, and the six Digital Twin
scores with their full per-component breakdown.

The "predict" path here does **not** re-run XGBoost/LightGBM/CatBoost/BiLSTM —
their probabilities are already persisted in artifacts/models/. We just slice
them by subject, combine, and HMM-smooth. That makes the API server boot in
about a second.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sleep_twin.digital_twin import (
    HumanState,
    PhysioSummary,
    SleepSummary,
    estimate_human_state,
    simulate_sleep_less,
    simulate_train_harder,
    summarize_3class_sleep,
    summarize_physio,
)
from sleep_twin.features import load_feature_cache
from sleep_twin.labels import LABEL_NAMES_3CLASS
from sleep_twin.paths import DEFAULT_DATA_ROOT, DEFAULT_FEATURE_DIR, DEFAULT_MODEL_DIR
from sleep_twin.temporal_smoothing import smooth_subject_sequences


_DEFAULT_SOURCES = ("xgboost_cuda", "lightgbm", "catboost", "catboost_tuned")


class SleepTwinService:
    """In-memory analysis service. One instance per process."""

    def __init__(
        self,
        feature_path: Path = DEFAULT_FEATURE_DIR / "sleep_features.npz",
        booster_dir: Path = DEFAULT_MODEL_DIR / "boosters",
        sequence_dir: Path = DEFAULT_MODEL_DIR / "sequence",
        ensemble_dir: Path = DEFAULT_MODEL_DIR / "ensemble",
        data_root: Path = DEFAULT_DATA_ROOT,
    ) -> None:
        self.data_root = data_root
        self.fs = load_feature_cache(feature_path)

        ensemble_bundle = joblib.load(ensemble_dir / "ensemble.joblib")
        self.transitions = np.asarray(ensemble_bundle["transitions"], dtype=np.float64)
        self.initial = np.asarray(ensemble_bundle["initial"], dtype=np.float64)
        self.log_T = np.log(self.transitions + 1e-12)
        self.log_I = np.log(self.initial + 1e-12)

        # Load every available per-source probability across all 26 773 epochs
        boost_npz = np.load(booster_dir / "boosting_probabilities.npz")
        self.probs: dict[str, np.ndarray] = {}
        for src in _DEFAULT_SOURCES:
            key = f"{src}__all"
            if key in boost_npz.files:
                self.probs[src] = boost_npz[key].astype(np.float32)

        seq_path = sequence_dir / "sequence_probabilities.npz"
        if seq_path.exists():
            seq = np.load(seq_path)
            if "all_probs" in seq.files:
                self.probs["bilstm_attention"] = seq["all_probs"].astype(np.float32)

        # Which subjects landed in which split
        split_file = booster_dir / "split_subjects.json"
        if split_file.exists():
            self.split = json.loads(split_file.read_text())
        else:
            self.split = {"train_subjects": [], "val_subjects": [], "test_subjects": []}

        # Headline leaderboard metrics (for the UI badge)
        metrics_file = ensemble_dir / "ensemble_metrics.json"
        self.leaderboard_rows: list[dict] = []
        if metrics_file.exists():
            self.leaderboard_rows = json.loads(metrics_file.read_text()).get("rows", [])

    # ------------------------------------------------------------------
    # Subject catalog
    # ------------------------------------------------------------------

    def list_subjects(self) -> list[dict[str, Any]]:
        subjects: list[dict[str, Any]] = []
        for subject in sorted(set(self.fs.subject_ids.tolist())):
            mask = self.fs.subject_ids == subject
            n_epochs = int(mask.sum())
            split = self._split_of(subject)
            subjects.append(
                {
                    "id": subject,
                    "n_epochs": n_epochs,
                    "duration_hours": round(n_epochs * 0.5 / 60.0, 2),
                    "split": split,
                }
            )
        return subjects

    def _split_of(self, subject: str) -> str:
        if subject in self.split.get("test_subjects", []):
            return "test"
        if subject in self.split.get("val_subjects", []):
            return "val"
        if subject in self.split.get("train_subjects", []):
            return "train"
        return "unknown"

    # ------------------------------------------------------------------
    # Hypnogram prediction (geometric ensemble + HMM)
    # ------------------------------------------------------------------

    def _ensemble_proba(self, mask: np.ndarray) -> np.ndarray:
        stack = np.stack([p[mask] for p in self.probs.values()], axis=0)
        # Geometric mean: average in log space, re-normalize
        geo = np.exp(np.mean(np.log(stack + 1e-9), axis=0))
        return geo / geo.sum(axis=1, keepdims=True)

    def predict_subject(self, subject_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (epoch_times_sec, predicted_stages, true_stages) chronologically ordered."""
        mask = self.fs.subject_ids == subject_id
        if not mask.any():
            raise KeyError(f"unknown subject: {subject_id}")
        idx = np.where(mask)[0]
        order = idx[np.argsort(self.fs.epoch_times[idx])]
        proba = self._ensemble_proba(np.zeros(len(self.fs.subject_ids), dtype=bool))
        # rebuild with ordered slice
        ordered_proba = self._ensemble_proba(mask)[np.argsort(self.fs.epoch_times[idx])]
        smoothed = smooth_subject_sequences(
            proba=ordered_proba,
            subject_ids=np.full(len(ordered_proba), subject_id),
            epoch_times=self.fs.epoch_times[order],
            log_transitions=self.log_T,
            log_initial=self.log_I,
        )
        return (
            self.fs.epoch_times[order].astype(np.float64),
            smoothed.astype(np.int64),
            self.fs.y[order].astype(np.int64),
        )

    # ------------------------------------------------------------------
    # Raw signals for visualization
    # ------------------------------------------------------------------

    def _load_hr(self, subject_id: str) -> tuple[np.ndarray, np.ndarray]:
        path = self.data_root / "heart_rate" / f"{subject_id}_heartrate.txt"
        if not path.exists():
            return np.zeros(0), np.zeros(0)
        df = pd.read_csv(path, sep=",", header=None, names=["time", "bpm"], dtype=np.float64)
        df = df.dropna()
        df = df[(df["bpm"] >= 25.0) & (df["bpm"] <= 240.0)]
        return df["time"].to_numpy(), df["bpm"].to_numpy()

    def _load_motion_enmo(self, subject_id: str) -> tuple[np.ndarray, np.ndarray]:
        path = self.data_root / "motion" / f"{subject_id}_acceleration.txt"
        if not path.exists():
            return np.zeros(0), np.zeros(0)
        df = pd.read_csv(path, sep=r"\s+", header=None, names=["time", "x", "y", "z"], dtype=np.float64, engine="c")
        df = df.dropna()
        mag = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2).to_numpy()
        enmo = np.maximum(mag - 1.0, 0.0)
        return df["time"].to_numpy(), enmo

    def _load_steps(self, subject_id: str) -> float:
        path = self.data_root / "steps" / f"{subject_id}_steps.txt"
        if not path.exists():
            return 0.0
        df = pd.read_csv(path, sep=",", header=None, names=["time", "steps"], dtype=np.float64)
        df = df.dropna()
        return float(np.maximum(df["steps"].to_numpy(), 0.0).sum())

    @staticmethod
    def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
        if x.size <= max_points or x.size == 0:
            return x, y
        # bucket-mean downsampling, preserving min/max would be nicer but mean is cheap and smooth
        bins = np.linspace(x[0], x[-1], max_points + 1)
        idx = np.clip(np.searchsorted(bins, x) - 1, 0, max_points - 1)
        sums = np.bincount(idx, weights=y, minlength=max_points)
        counts = np.bincount(idx, minlength=max_points)
        with np.errstate(invalid="ignore"):
            means = np.where(counts > 0, sums / counts, np.nan)
        centers = 0.5 * (bins[:-1] + bins[1:])
        return centers, means

    # ------------------------------------------------------------------
    # Headline analysis
    # ------------------------------------------------------------------

    def analyze(self, subject_id: str) -> dict[str, Any]:
        epoch_times, predicted, truth = self.predict_subject(subject_id)
        hr_t, hr_bpm = self._load_hr(subject_id)
        m_t, m_enmo = self._load_motion_enmo(subject_id)
        steps_total = self._load_steps(subject_id)

        sleep = summarize_3class_sleep(predicted)
        physio = summarize_physio(predicted, epoch_times, hr_t, hr_bpm, m_t, m_enmo)
        state = estimate_human_state(sleep, physio, prior_day_steps=steps_total)

        # Per-epoch confidence: max class probability after the ensemble
        mask = self.fs.subject_ids == subject_id
        idx = np.where(mask)[0]
        order = idx[np.argsort(self.fs.epoch_times[idx])]
        ordered_proba = self._ensemble_proba(mask)[np.argsort(self.fs.epoch_times[idx])]
        confidence = ordered_proba.max(axis=1)

        hr_x, hr_y = self._downsample_xy(hr_t, hr_bpm, max_points=600)
        m_x, m_y = self._downsample_xy(m_t, m_enmo, max_points=600)

        agreement = float((predicted == truth).mean()) if truth.size else 0.0

        return {
            "subject_id": subject_id,
            "split": self._split_of(subject_id),
            "n_epochs": int(predicted.size),
            "duration_hours": round(predicted.size * 0.5 / 60.0, 2),
            "model_agreement_with_psg": agreement,
            "epoch_times_sec": epoch_times.tolist(),
            "predicted_stages": predicted.tolist(),
            "true_stages": truth.tolist(),
            "epoch_confidence": confidence.tolist(),
            "ensemble_probabilities": ordered_proba.tolist(),
            "label_names": LABEL_NAMES_3CLASS,
            "hr_timeline": {
                "time_sec": [None if not np.isfinite(v) else float(v) for v in hr_x],
                "bpm": [None if not np.isfinite(v) else float(v) for v in hr_y],
            },
            "motion_timeline": {
                "time_sec": [None if not np.isfinite(v) else float(v) for v in m_x],
                "enmo": [None if not np.isfinite(v) else float(v) for v in m_y],
            },
            "steps_total": steps_total,
            "sleep_summary": asdict(sleep),
            "physio_summary": asdict(physio),
            "state": _human_state_to_dict(state),
            "baseline_inputs": {
                "prior_day_steps": steps_total,
            },
        }

    def whatif(
        self,
        subject_id: str,
        lost_minutes: float = 0.0,
        extra_steps: float = 0.0,
    ) -> dict[str, Any]:
        """Re-score a subject under a hypothetical perturbation."""
        epoch_times, predicted, _ = self.predict_subject(subject_id)
        hr_t, hr_bpm = self._load_hr(subject_id)
        m_t, m_enmo = self._load_motion_enmo(subject_id)
        baseline_steps = self._load_steps(subject_id)

        sleep = summarize_3class_sleep(predicted)
        physio = summarize_physio(predicted, epoch_times, hr_t, hr_bpm, m_t, m_enmo)
        base = estimate_human_state(sleep, physio, prior_day_steps=baseline_steps)

        modified = base
        if lost_minutes > 0:
            modified = simulate_sleep_less(modified, lost_minutes)
        if extra_steps > 0:
            modified = simulate_train_harder(modified, extra_steps)

        return {
            "subject_id": subject_id,
            "baseline": _human_state_to_dict(base),
            "scenario": _human_state_to_dict(modified),
            "delta": {
                k: _human_state_to_dict(modified)[k] - _human_state_to_dict(base)[k]
                for k in (
                    "sleep_quality_score",
                    "sleep_debt_score",
                    "recovery_score",
                    "stress_index",
                    "fatigue_score",
                    "energy_level",
                )
            },
        }

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def leaderboard(self) -> list[dict[str, Any]]:
        return self.leaderboard_rows


def _human_state_to_dict(state: HumanState) -> dict[str, Any]:
    return {
        "sleep_quality_score": float(state.sleep_quality_score),
        "sleep_debt_score": float(state.sleep_debt_score),
        "recovery_score": float(state.recovery_score),
        "stress_index": float(state.stress_index),
        "fatigue_score": float(state.fatigue_score),
        "energy_level": float(state.energy_level),
        "components": state.components,
    }
