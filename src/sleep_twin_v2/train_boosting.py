"""Train boosting models (XGBoost + LightGBM + CatBoost) on v2 features.

Each model is fit on the same imputed training set with class-balanced sample
weights and tuned hyperparameters. Probabilities are persisted so downstream
HMM smoothing and stacking can consume them.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight

from sleep_twin_v2.evaluation import evaluate_predictions, metrics_row, write_json
from sleep_twin_v2.features import load_feature_cache
from sleep_twin_v2.labels import LABEL_NAMES_3CLASS
from sleep_twin_v2.paths import DEFAULT_FEATURE_DIR, DEFAULT_MODEL_DIR
from sleep_twin_v2.splits import make_group_holdout_split, split_subject_summary


def _xgb(num_classes: int, device: str) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.80,
        colsample_bylevel=0.80,
        min_child_weight=2.0,
        reg_lambda=1.5,
        reg_alpha=0.1,
        gamma=0.05,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        device=device,
        n_jobs=max(os.cpu_count() or 1, 1),
        random_state=42,
        early_stopping_rounds=80,
    )


def _lgbm(num_classes: int) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=2000,
        max_depth=-1,
        num_leaves=95,
        learning_rate=0.025,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        min_child_samples=20,
        reg_lambda=1.5,
        reg_alpha=0.1,
        objective="multiclass",
        num_class=num_classes,
        class_weight="balanced",
        n_jobs=max(os.cpu_count() or 1, 1),
        random_state=42,
        verbose=-1,
    )


def _catboost(num_classes: int, device: str) -> Any:
    from catboost import CatBoostClassifier

    task_type = "GPU" if device == "cuda" else "CPU"
    return CatBoostClassifier(
        iterations=2000,
        depth=7,
        learning_rate=0.04,
        l2_leaf_reg=4.0,
        random_strength=1.0,
        bagging_temperature=0.8,
        border_count=128,
        loss_function="MultiClass",
        classes_count=num_classes,
        auto_class_weights="Balanced",
        task_type=task_type,
        devices="0" if task_type == "GPU" else None,
        verbose=False,
        random_seed=42,
        early_stopping_rounds=100,
    )


def train_boosters(
    feature_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    val_size: float = 0.15,
    seed: int = 42,
    xgb_device: str = "auto",
    catboost_device: str = "cpu",
) -> dict:
    fs = load_feature_cache(feature_path)
    labels = LABEL_NAMES_3CLASS
    num_classes = len(labels)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx, test_idx = make_group_holdout_split(
        fs.subject_ids, test_size=test_size, val_size=val_size, random_state=seed
    )
    write_json(output_dir / "split_subjects.json", split_subject_summary(fs.subject_ids, train_idx, val_idx, test_idx))

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train = imputer.fit_transform(fs.X[train_idx])
    X_val = imputer.transform(fs.X[val_idx])
    X_test = imputer.transform(fs.X[test_idx])
    X_all = imputer.transform(fs.X)
    joblib.dump({"imputer": imputer, "feature_names": fs.feature_names}, output_dir / "preprocessor.joblib")

    y_train = fs.y[train_idx]
    y_val = fs.y[val_idx]
    y_test = fs.y[test_idx]
    sample_weight = compute_sample_weight("balanced", y_train)

    rows: list[dict] = []
    proba_store: dict[str, dict[str, np.ndarray]] = {}

    requested_xgb = "cuda" if xgb_device == "auto" else xgb_device

    def _fit_and_score(name: str, model, fit_kwargs: dict) -> None:
        start = time.time()
        print(f"[boost] training {name}...")
        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except Exception as exc:
            if name.startswith("xgboost") and requested_xgb == "cuda":
                print(f"  XGBoost CUDA failed ({exc}); retrying on CPU.")
                model = _xgb(num_classes, "cpu")
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight=sample_weight, verbose=False)
            else:
                raise

        proba_train = model.predict_proba(X_train)
        proba_val = model.predict_proba(X_val)
        proba_test = model.predict_proba(X_test)
        proba_all = model.predict_proba(X_all)
        proba_store[name] = {
            "train": proba_train.astype(np.float32),
            "val": proba_val.astype(np.float32),
            "test": proba_test.astype(np.float32),
            "all": proba_all.astype(np.float32),
        }

        y_pred = np.argmax(proba_test, axis=1)
        metrics = evaluate_predictions(y_test, y_pred, labels)
        row = metrics_row(name, metrics)
        rows.append(row)
        joblib.dump(model, output_dir / f"{name}.joblib")
        write_json(output_dir / f"{name}_metrics.json", metrics)
        elapsed = time.time() - start
        print(
            f"  {name}: acc={row['accuracy']:.4f}, bal={row['balanced_accuracy']:.4f}, "
            f"macroF1={row['macro_f1']:.4f}, kappa={row['cohen_kappa']:.4f} ({elapsed:.1f}s)"
        )

    # XGBoost
    xgb_model = _xgb(num_classes, requested_xgb)
    _fit_and_score(
        f"xgboost_{requested_xgb}",
        xgb_model,
        {"eval_set": [(X_val, y_val)], "sample_weight": sample_weight, "verbose": False},
    )

    # LightGBM
    lgbm_model = _lgbm(num_classes)
    from lightgbm import early_stopping, log_evaluation

    _fit_and_score(
        "lightgbm",
        lgbm_model,
        {
            "eval_set": [(X_val, y_val)],
            "callbacks": [early_stopping(stopping_rounds=80, verbose=False), log_evaluation(0)],
        },
    )

    # CatBoost
    cat_model = _catboost(num_classes, catboost_device)
    _fit_and_score(
        "catboost",
        cat_model,
        {"eval_set": (X_val, y_val), "use_best_model": True, "verbose": False},
    )

    np.savez_compressed(
        output_dir / "boosting_probabilities.npz",
        **{f"{name}__{split}": arr for name, splits in proba_store.items() for split, arr in splits.items()},
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    leaderboard = output_dir / "leaderboard.csv"
    with leaderboard.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "cohen_kappa"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["macro_f1"], reverse=True))
    return {"rows": rows, "proba_path": output_dir / "boosting_probabilities.npz"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train boosting ensemble for sleep staging.")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURE_DIR / "sleep_features_v2.npz")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR / "boosters")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--xgb-device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--catboost-device", choices=["cpu", "cuda"], default="cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    train_boosters(
        feature_path=args.features,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        xgb_device=args.xgb_device,
        catboost_device=args.catboost_device,
    )


if __name__ == "__main__":
    main()
