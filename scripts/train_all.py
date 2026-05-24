"""End-to-end training pipeline.

Stages (executed in order):
  1. Build the feature cache (skipped if it already exists; --force to rebuild)
  2. Train XGBoost + LightGBM + CatBoost boosters
  3. (optional) Run Optuna sweep on CatBoost
  4. (optional) Train BiLSTM sequence model  -- needs torch + numpy
  5. Merge per-source probabilities into one .npz
  6. Stack + HMM smoothing -> final leaderboard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleep_twin.features import build_feature_cache  # noqa: E402
from sleep_twin.paths import DEFAULT_DATA_ROOT, DEFAULT_FEATURE_DIR, DEFAULT_MODEL_DIR  # noqa: E402
from sleep_twin.stacking import stack_and_smooth  # noqa: E402
from sleep_twin.train_boosting import train_boosters  # noqa: E402


def _merge_probabilities(booster_dir: Path, tuned_dir: Path | None) -> None:
    """Combine the booster probability file with the tuned-catboost file in place."""
    boost_path = booster_dir / "boosting_probabilities.npz"
    if not boost_path.exists():
        raise FileNotFoundError(f"missing {boost_path}")
    payload = {k: np.load(boost_path)[k] for k in np.load(boost_path).files}
    if tuned_dir and (tuned_dir / "catboost_tuned_probabilities.npz").exists():
        tuned = np.load(tuned_dir / "catboost_tuned_probabilities.npz")
        for k in tuned.files:
            if k.startswith("catboost_tuned__"):
                payload[k] = tuned[k]
        print(f"[merge] added tuned catboost probabilities")
    np.savez_compressed(boost_path, **payload)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the full Sleep Twin pipeline.")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURE_DIR / "sleep_features.npz")
    ap.add_argument("--force-features", action="store_true")
    ap.add_argument("--skip-tune", action="store_true", help="Skip Optuna CatBoost sweep.")
    ap.add_argument("--skip-sequence", action="store_true", help="Skip BiLSTM training.")
    ap.add_argument("--xgb-device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--catboost-device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--tune-trials", type=int, default=25)
    ap.add_argument("--tune-timeout", type=int, default=1500)
    ap.add_argument("--sequence-radius", type=int, default=7)
    ap.add_argument("--sequence-hidden", type=int, default=64)
    ap.add_argument("--sequence-layers", type=int, default=1)
    ap.add_argument("--sequence-epochs", type=int, default=20)
    ap.add_argument("--sequence-batch-size", type=int, default=512)
    ap.add_argument("--sequence-device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.force_features or not args.features.exists():
        print(f"[features] building cache at {args.features}")
        build_feature_cache(args.data_root, args.features)
    else:
        print(f"[features] using existing cache {args.features}")

    booster_dir = DEFAULT_MODEL_DIR / "boosters"
    print("\n[boosting] training XGBoost + LightGBM + CatBoost")
    train_boosters(
        feature_path=args.features,
        output_dir=booster_dir,
        seed=args.seed,
        xgb_device=args.xgb_device,
        catboost_device=args.catboost_device,
    )

    tuned_dir = DEFAULT_MODEL_DIR / "catboost_tuned"
    if not args.skip_tune:
        from sleep_twin.tune_catboost import tune_catboost

        print("\n[tune] Optuna sweep on CatBoost")
        tune_catboost(
            feature_path=args.features,
            output_dir=tuned_dir,
            n_trials=args.tune_trials,
            timeout_seconds=args.tune_timeout,
            task_type="GPU" if args.catboost_device == "cuda" else "CPU",
            seed=args.seed,
        )
    else:
        tuned_dir = None

    sequence_dir = DEFAULT_MODEL_DIR / "sequence"
    if not args.skip_sequence:
        try:
            from sleep_twin.train_sequence import train_sequence_model

            print("\n[sequence] training BiLSTM")
            train_sequence_model(
                feature_path=args.features,
                output_dir=sequence_dir,
                seed=args.seed,
                sequence_radius=args.sequence_radius,
                batch_size=args.sequence_batch_size,
                epochs=args.sequence_epochs,
                device_name=args.sequence_device,
                hidden_size=args.sequence_hidden,
                num_layers=args.sequence_layers,
            )
        except ImportError as exc:
            print(f"[sequence] skipped (torch not available): {exc}")
            sequence_dir = None
    else:
        sequence_dir = None

    print("\n[merge] combining base probabilities")
    _merge_probabilities(booster_dir, tuned_dir)

    print("\n[stack] meta-learner + HMM smoothing")
    stack_and_smooth(
        feature_path=args.features,
        booster_dir=booster_dir,
        sequence_dir=sequence_dir,
        output_dir=DEFAULT_MODEL_DIR / "ensemble",
    )


if __name__ == "__main__":
    main()
