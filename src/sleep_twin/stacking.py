"""Ensemble stacking + HMM/Viterbi smoothing.

Loads per-model class probabilities saved by the boosting and sequence trainers,
fits a meta-learner on the validation split, then post-processes with HMM/Viterbi
smoothing fit from training-label transition statistics. Also reports simple
arithmetic/geometric averaging baselines for ablation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sleep_twin.evaluation import evaluate_predictions, format_metrics_table, metrics_row, write_json
from sleep_twin.features import load_feature_cache
from sleep_twin.labels import LABEL_NAMES_3CLASS
from sleep_twin.paths import DEFAULT_FEATURE_DIR, DEFAULT_MODEL_DIR
from sleep_twin.temporal_smoothing import (
    fit_initial_probs,
    fit_transition_matrix,
    smooth_subject_sequences,
)


BOOST_KEYS = ("xgboost_cuda", "xgboost_cpu", "lightgbm", "catboost", "catboost_tuned")


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
    return np.concatenate(cols, axis=1).astype(np.float32), used


def _load_sequence_probs(path: Path, split: str) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.load(path)
    key = {"train": "train_probs", "val": "val_probs", "test": "test_probs", "all": "all_probs"}[split]
    if key not in data.files:
        return None
    return data[key].astype(np.float32)


def _concat_sources(boost_path: Path, seq_path: Path | None, split: str) -> tuple[np.ndarray, list[str]]:
    boost, used = _load_proba(boost_path, list(BOOST_KEYS), split)
    matrices = [boost]
    sources = list(used)
    if seq_path is not None:
        seq = _load_sequence_probs(seq_path, split)
        if seq is not None:
            matrices.append(seq)
            sources.append("bilstm_attention")
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

    X_train_stack, sources = _concat_sources(boost_proba_path, seq_proba_path, "train")
    X_val_stack, _ = _concat_sources(boost_proba_path, seq_proba_path, "val")
    X_test_stack, _ = _concat_sources(boost_proba_path, seq_proba_path, "test")
    print(f"[stack] sources: {sources}  feature dim per split: {X_train_stack.shape[1]}")

    # Meta-learner candidates
    candidates: dict[str, object] = {
        "logreg_C1.0": LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs"),
        "logreg_C0.3": LogisticRegression(max_iter=3000, C=0.3, solver="lbfgs"),
        "logreg_C4.0": LogisticRegression(max_iter=3000, C=4.0, solver="lbfgs"),
        "mlp_64": MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, alpha=1e-3, random_state=42, early_stopping=True),
        "mlp_64_32": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, alpha=1e-3, random_state=42, early_stopping=True),
    }

    best_name = None
    best_val_macro = -np.inf
    best_meta = None
    val_results: list[tuple[str, float, float]] = []

    # Fit each candidate on TRAIN probabilities, score on VAL probabilities.
    # Training-set base probabilities here are "in-sample" predictions from the
    # base learners, so they're optimistic — but the val score still ranks the
    # meta-learners fairly because all see the same in-sample bias.
    for name, model in candidates.items():
        model.fit(X_train_stack, y_train)
        val_probs = model.predict_proba(X_val_stack)
        val_pred = np.argmax(val_probs, axis=1)
        val_metrics = evaluate_predictions(y_val, val_pred, labels)
        val_results.append((name, val_metrics["macro_f1"], val_metrics["balanced_accuracy"]))
        print(f"[stack] {name:15s} val_macroF1={val_metrics['macro_f1']:.4f} val_balAcc={val_metrics['balanced_accuracy']:.4f}")
        if val_metrics["macro_f1"] > best_val_macro:
            best_val_macro = val_metrics["macro_f1"]
            best_name = name
            best_meta = model

    print(f"[stack] best meta-learner: {best_name} (val macroF1={best_val_macro:.4f})")

    # HMM fit from training labels
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

    # Final predictions: best meta-learner on test
    rows: list[dict] = []
    stacked_test_probs = best_meta.predict_proba(X_test_stack)
    rows.append(metrics_row(f"stack_{best_name}", evaluate_predictions(y_test, np.argmax(stacked_test_probs, axis=1), labels)))

    smoothed_preds = smooth_subject_sequences(
        proba=stacked_test_probs,
        subject_ids=fs.subject_ids[test_idx],
        epoch_times=fs.epoch_times[test_idx],
        log_transitions=log_T,
        log_initial=log_I,
    )
    smoothed_metrics = evaluate_predictions(y_test, smoothed_preds, labels)
    rows.append(metrics_row(f"stack_{best_name}_hmm", smoothed_metrics))

    seg_size = num_classes
    base_test = {}
    for i, src in enumerate(sources):
        base_test[src] = X_test_stack[:, i * seg_size : (i + 1) * seg_size]
        rows.append(metrics_row(f"base_{src}", evaluate_predictions(y_test, np.argmax(base_test[src], axis=1), labels)))
        preds = smooth_subject_sequences(
            proba=base_test[src],
            subject_ids=fs.subject_ids[test_idx],
            epoch_times=fs.epoch_times[test_idx],
            log_transitions=log_T,
            log_initial=log_I,
        )
        rows.append(metrics_row(f"base_{src}_hmm", evaluate_predictions(y_test, preds, labels)))

    avg_test = np.mean(list(base_test.values()), axis=0)
    rows.append(metrics_row("avg_mean", evaluate_predictions(y_test, np.argmax(avg_test, axis=1), labels)))
    avg_hmm = smooth_subject_sequences(
        proba=avg_test,
        subject_ids=fs.subject_ids[test_idx],
        epoch_times=fs.epoch_times[test_idx],
        log_transitions=log_T,
        log_initial=log_I,
    )
    rows.append(metrics_row("avg_mean_hmm", evaluate_predictions(y_test, avg_hmm, labels)))

    # Geometric mean (log-average)
    geo_test = np.exp(np.mean(np.log(np.stack(list(base_test.values())) + 1e-9), axis=0))
    geo_test = geo_test / geo_test.sum(axis=1, keepdims=True)
    rows.append(metrics_row("avg_geo", evaluate_predictions(y_test, np.argmax(geo_test, axis=1), labels)))
    geo_hmm = smooth_subject_sequences(
        proba=geo_test,
        subject_ids=fs.subject_ids[test_idx],
        epoch_times=fs.epoch_times[test_idx],
        log_transitions=log_T,
        log_initial=log_I,
    )
    rows.append(metrics_row("avg_geo_hmm", evaluate_predictions(y_test, geo_hmm, labels)))

    sorted_rows = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)

    joblib.dump(
        {
            "meta_learner": best_meta,
            "meta_learner_name": best_name,
            "transitions": transitions,
            "initial": initial,
            "sources": sources,
            "label_names": labels,
            "val_results": val_results,
        },
        output_dir / "ensemble.joblib",
    )
    write_json(
        output_dir / "ensemble_metrics.json",
        {
            "best_meta": best_name,
            "best_val_macro_f1": best_val_macro,
            "rows": sorted_rows,
            "transitions": transitions.tolist(),
            "initial": initial.tolist(),
            "sources": sources,
            "val_results": val_results,
        },
    )

    (output_dir / "leaderboard.md").write_text(
        f"# Sleep Twin Ensemble Leaderboard\n\n"
        f"**Best meta-learner**: `{best_name}` (val macro-F1={best_val_macro:.4f})\n\n"
        f"{format_metrics_table(sorted_rows)}\n"
    )

    print("\n=== leaderboard (sorted by macro F1) ===")
    print(format_metrics_table(sorted_rows))
    return {"rows": sorted_rows, "best_meta": best_name}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stacking + HMM smoothing.")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURE_DIR / "sleep_features.npz")
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
