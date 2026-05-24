"""Stacked ensemble + HMM smoothing.

Takes per-model class probabilities saved by the boosting and sequence trainers,
fits a softmax/logreg meta-learner on the validation split, applies it to test,
then runs HMM/Viterbi smoothing on the final probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from sleep_twin_v2.evaluation import evaluate_predictions, format_metrics_table, metrics_row, write_json
from sleep_twin_v2.features import load_feature_cache
from sleep_twin_v2.labels import LABEL_NAMES_3CLASS
from sleep_twin_v2.paths import DEFAULT_FEATURE_DIR, DEFAULT_MODEL_DIR
from sleep_twin_v2.temporal_smoothing import (
    fit_initial_probs,
    fit_transition_matrix,
    smooth_subject_sequences,
)


MODEL_KEYS = ("xgboost_cuda", "xgboost_cpu", "lightgbm", "catboost", "bilstm_attention_v2")


def _load_proba(path: Path, keys: list[str], split: str) -> tuple[np.ndarray, list[str]]:
    data = np.load(path)
    cols = []
    used: list[str] = []
    for key in keys:
        full = f"{key}__{split}"
        if full in data.files:
            cols.append(data[full])
            used.append(key)
    if not cols:
        raise RuntimeError(f"No probability columns found in {path} for split={split}")
    stacked = np.concatenate(cols, axis=1)
    return stacked.astype(np.float32), used


def _load_sequence_probs(path: Path, split: str) -> np.ndarray:
    data = np.load(path)
    key = {"train": "train_probs", "val": "val_probs", "test": "test_probs", "all": "all_probs"}[split]
    return data[key].astype(np.float32)


def _concat_proba_sources(
    boost_path: Path, seq_path: Path | None, split: str, boost_keys: list[str]
) -> tuple[np.ndarray, list[str]]:
    matrices: list[np.ndarray] = []
    sources: list[str] = []
    boost, used = _load_proba(boost_path, boost_keys, split)
    matrices.append(boost)
    sources.extend(used)
    if seq_path and seq_path.exists():
        seq = _load_sequence_probs(seq_path, split)
        matrices.append(seq)
        sources.append("bilstm_attention_v2")
    return np.concatenate(matrices, axis=1), sources


def stack_and_smooth(
    feature_path: Path,
    booster_dir: Path,
    sequence_dir: Path | None,
    output_dir: Path,
) -> dict:
    fs = load_feature_cache(feature_path)
    labels = LABEL_NAMES_3CLASS
    num_classes = len(labels)
    output_dir.mkdir(parents=True, exist_ok=True)

    boost_proba_path = booster_dir / "boosting_probabilities.npz"
    seq_proba_path = sequence_dir / "sequence_probabilities.npz" if sequence_dir else None
    data = np.load(boost_proba_path)
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    boost_keys = [k for k in MODEL_KEYS if k != "bilstm_attention_v2"]

    X_train_stack, sources = _concat_proba_sources(boost_proba_path, seq_proba_path, "train", boost_keys)
    X_val_stack, _ = _concat_proba_sources(boost_proba_path, seq_proba_path, "val", boost_keys)
    X_test_stack, _ = _concat_proba_sources(boost_proba_path, seq_proba_path, "test", boost_keys)

    meta = LogisticRegression(
        max_iter=2000,
        C=4.0,
        class_weight="balanced",
        solver="lbfgs",
    )
    meta.fit(X_val_stack, y_val)

    rows: list[dict] = []

    def _eval(name: str, probs: np.ndarray, y_ref: np.ndarray) -> tuple[dict, np.ndarray]:
        preds = np.argmax(probs, axis=1)
        m = evaluate_predictions(y_ref, preds, labels)
        rows.append(metrics_row(name, m))
        return m, preds

    # Individual base models on test (for reference)
    cursor = 0
    base_probas_test: dict[str, np.ndarray] = {}
    seg_size = num_classes
    for src in sources:
        seg = X_test_stack[:, cursor : cursor + seg_size]
        base_probas_test[src] = seg
        _eval(f"base_{src}", seg, y_test)
        cursor += seg_size

    # Ensemble averaging (equal-weight, for ablation)
    avg_test = np.mean([base_probas_test[s] for s in sources], axis=0)
    _eval("ensemble_mean", avg_test, y_test)

    # Stacked meta-learner predictions
    stacked_test_probs = meta.predict_proba(X_test_stack)
    _eval("ensemble_stacked", stacked_test_probs, y_test)

    # Sequence-aware HMM smoothing
    train_sequences: list[np.ndarray] = []
    for subject in np.unique(fs.subject_ids[train_idx]):
        mask = fs.subject_ids == subject
        idx = np.where(mask & np.isin(np.arange(len(fs.y)), train_idx))[0]
        order = idx[np.argsort(fs.epoch_times[idx])]
        train_sequences.append(fs.y[order])

    transitions = fit_transition_matrix(train_sequences, num_classes)
    initial = fit_initial_probs(train_sequences, num_classes)
    log_T = np.log(transitions + 1e-12)
    log_I = np.log(initial + 1e-12)

    smoothed_preds = smooth_subject_sequences(
        proba=stacked_test_probs,
        subject_ids=fs.subject_ids[test_idx],
        epoch_times=fs.epoch_times[test_idx],
        log_transitions=log_T,
        log_initial=log_I,
    )
    smoothed_metrics = evaluate_predictions(y_test, smoothed_preds, labels)
    rows.append(metrics_row("ensemble_stacked_hmm", smoothed_metrics))

    # Also apply HMM smoothing to each base model alone for ablation
    for src, probs in base_probas_test.items():
        preds = smooth_subject_sequences(
            proba=probs,
            subject_ids=fs.subject_ids[test_idx],
            epoch_times=fs.epoch_times[test_idx],
            log_transitions=log_T,
            log_initial=log_I,
        )
        m = evaluate_predictions(y_test, preds, labels)
        rows.append(metrics_row(f"base_{src}_hmm", m))

    sorted_rows = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)

    joblib.dump(
        {
            "meta_learner": meta,
            "transitions": transitions,
            "initial": initial,
            "sources": sources,
            "label_names": labels,
        },
        output_dir / "ensemble.joblib",
    )
    write_json(
        output_dir / "ensemble_metrics.json",
        {
            "stacked": smoothed_metrics,
            "rows": sorted_rows,
            "transitions": transitions.tolist(),
            "initial": initial.tolist(),
            "sources": sources,
        },
    )

    leaderboard_md = output_dir / "leaderboard.md"
    leaderboard_md.write_text("# v2 Ensemble Leaderboard\n\n" + format_metrics_table(sorted_rows) + "\n")

    print("\n=== v2 leaderboard (sorted by macro F1) ===")
    print(format_metrics_table(sorted_rows))
    return {"rows": sorted_rows, "metrics": smoothed_metrics}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run stacked ensemble + HMM smoothing.")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURE_DIR / "sleep_features_v2.npz")
    ap.add_argument("--booster-dir", type=Path, default=DEFAULT_MODEL_DIR / "boosters")
    ap.add_argument("--sequence-dir", type=Path, default=DEFAULT_MODEL_DIR / "sequence")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR / "ensemble")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    seq_dir = args.sequence_dir if (args.sequence_dir / "sequence_probabilities.npz").exists() else None
    stack_and_smooth(
        feature_path=args.features,
        booster_dir=args.booster_dir,
        sequence_dir=seq_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
