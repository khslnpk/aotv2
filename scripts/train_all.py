"""End-to-end training pipeline for v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleep_twin_v2.features import build_feature_cache  # noqa: E402
from sleep_twin_v2.paths import DEFAULT_DATA_ROOT, DEFAULT_FEATURE_DIR, DEFAULT_MODEL_DIR  # noqa: E402
from sleep_twin_v2.stacking import stack_and_smooth  # noqa: E402
from sleep_twin_v2.train_boosting import train_boosters  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train v2 pipeline.")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURE_DIR / "sleep_features_v2.npz")
    ap.add_argument("--force-features", action="store_true")
    ap.add_argument("--skip-sequence", action="store_true", help="Skip BiLSTM training (requires torch).")
    ap.add_argument("--xgb-device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--catboost-device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--sequence-radius", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
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

    sequence_dir = DEFAULT_MODEL_DIR / "sequence"
    if not args.skip_sequence:
        try:
            from sleep_twin_v2.train_sequence import train_sequence_model

            print("\n[sequence] training BiLSTM with attention + focal loss")
            train_sequence_model(
                feature_path=args.features,
                output_dir=sequence_dir,
                seed=args.seed,
                sequence_radius=args.sequence_radius,
                batch_size=args.batch_size,
                epochs=args.epochs,
            )
        except ImportError as exc:
            print(f"[sequence] skipped (torch not available): {exc}")
            sequence_dir = None
    else:
        sequence_dir = None

    print("\n[ensemble] stacking + HMM smoothing")
    stack_and_smooth(
        feature_path=args.features,
        booster_dir=booster_dir,
        sequence_dir=sequence_dir,
        output_dir=DEFAULT_MODEL_DIR / "ensemble",
    )


if __name__ == "__main__":
    main()
