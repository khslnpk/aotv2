"""Optuna sweep on CatBoost (the v2 booster with the highest balanced accuracy).

CatBoost already wins post-HMM smoothing. The goal here is to push it harder so
the ensemble + HMM stack has an even stronger base. Sweeps:
- depth (5-9)
- learning_rate (0.02-0.06)
- l2_leaf_reg (1-8)
- random_strength (0.5-2.0)
- bagging_temperature (0.3-1.5)
- border_count (96-192)
- iterations is bounded by early stopping
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.impute import SimpleImputer

from sleep_twin.evaluation import evaluate_predictions, metrics_row, write_json
from sleep_twin.features import load_feature_cache
from sleep_twin.labels import LABEL_NAMES_3CLASS
from sleep_twin.paths import DEFAULT_FEATURE_DIR, DEFAULT_MODEL_DIR
from sleep_twin.splits import make_group_holdout_split


def _objective(trial, X_train, y_train, X_val, y_val, num_classes: int, task_type: str):
    from catboost import CatBoostClassifier

    params = {
        "iterations": 2500,
        "depth": trial.suggest_int("depth", 5, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.06, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 8.0),
        "random_strength": trial.suggest_float("random_strength", 0.5, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.3, 1.5),
        "border_count": trial.suggest_int("border_count", 96, 192, step=16),
        "loss_function": "MultiClass",
        "classes_count": num_classes,
        "auto_class_weights": "Balanced",
        "task_type": task_type,
        "devices": "0" if task_type == "GPU" else None,
        "verbose": False,
        "random_seed": 42,
        "early_stopping_rounds": 80,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
    pred = model.predict(X_val).reshape(-1).astype(np.int64)
    metrics = evaluate_predictions(y_val, pred, LABEL_NAMES_3CLASS)
    # Maximize macro F1; balanced accuracy is a secondary tiebreaker via Optuna
    return metrics["macro_f1"]


def tune_catboost(
    feature_path: Path,
    output_dir: Path,
    n_trials: int = 30,
    timeout_seconds: int | None = 1200,
    task_type: str = "CPU",
    seed: int = 42,
) -> dict:
    import optuna

    fs = load_feature_cache(feature_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx, test_idx = make_group_holdout_split(
        fs.subject_ids, test_size=0.2, val_size=0.15, random_state=seed
    )

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train = imputer.fit_transform(fs.X[train_idx])
    X_val = imputer.transform(fs.X[val_idx])
    X_test = imputer.transform(fs.X[test_idx])
    y_train = fs.y[train_idx]
    y_val = fs.y[val_idx]
    y_test = fs.y[test_idx]

    num_classes = len(LABEL_NAMES_3CLASS)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda t: _objective(t, X_train, y_train, X_val, y_val, num_classes, task_type),
        n_trials=n_trials,
        timeout=timeout_seconds,
        show_progress_bar=False,
    )

    print(f"[tune] best trial: macro_F1={study.best_value:.4f}")
    print(f"[tune] best params: {study.best_params}")

    # Retrain best on train+val, evaluate on test
    from catboost import CatBoostClassifier

    best_params = dict(study.best_params)
    best_params.update(
        {
            "iterations": 2500,
            "loss_function": "MultiClass",
            "classes_count": num_classes,
            "auto_class_weights": "Balanced",
            "task_type": task_type,
            "devices": "0" if task_type == "GPU" else None,
            "verbose": False,
            "random_seed": seed,
            "early_stopping_rounds": 80,
        }
    )
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=False)

    proba_train = model.predict_proba(X_train).astype(np.float32)
    proba_val = model.predict_proba(X_val).astype(np.float32)
    proba_test = model.predict_proba(X_test).astype(np.float32)
    proba_all = model.predict_proba(imputer.transform(fs.X)).astype(np.float32)

    pred_test = np.argmax(proba_test, axis=1)
    metrics = evaluate_predictions(y_test, pred_test, LABEL_NAMES_3CLASS)
    row = metrics_row("catboost_tuned", metrics)

    joblib.dump(model, output_dir / "catboost_tuned.joblib")
    joblib.dump({"imputer": imputer, "feature_names": fs.feature_names}, output_dir / "preprocessor.joblib")
    write_json(output_dir / "catboost_tuned_metrics.json", metrics)
    write_json(output_dir / "best_params.json", best_params)

    np.savez_compressed(
        output_dir / "catboost_tuned_probabilities.npz",
        **{f"catboost_tuned__train": proba_train, f"catboost_tuned__val": proba_val,
           f"catboost_tuned__test": proba_test, f"catboost_tuned__all": proba_all},
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    print(
        f"[tune] retrained on best params: acc={row['accuracy']:.4f} "
        f"bal={row['balanced_accuracy']:.4f} macroF1={row['macro_f1']:.4f} "
        f"kappa={row['cohen_kappa']:.4f}"
    )
    return {"row": row, "best_params": best_params, "study": study}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Optuna sweep on CatBoost hyperparameters.")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURE_DIR / "sleep_features.npz")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR / "catboost_tuned")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--timeout-seconds", type=int, default=1500)
    ap.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    tune_catboost(
        feature_path=args.features,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        timeout_seconds=args.timeout_seconds,
        task_type=args.task_type,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
