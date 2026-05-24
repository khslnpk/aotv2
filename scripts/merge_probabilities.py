"""Merge per-source probability files (booster trio + tuned catboost + sequence)
into a single combined .npz that stacking_v2.py can consume.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--booster", type=Path, required=True, help="boosting_probabilities.npz")
    ap.add_argument("--tuned-catboost", type=Path, default=None, help="catboost_tuned_probabilities.npz")
    ap.add_argument("--sequence", type=Path, default=None, help="sequence_probabilities.npz")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if not args.booster.exists():
        sys.exit(f"missing booster file: {args.booster}")

    boost = np.load(args.booster)
    payload: dict[str, np.ndarray] = {k: boost[k] for k in boost.files}

    if args.tuned_catboost and args.tuned_catboost.exists():
        tuned = np.load(args.tuned_catboost)
        for k in tuned.files:
            if k.startswith("catboost_tuned__"):
                payload[k] = tuned[k]
        print(f"  merged tuned catboost: {[k for k in tuned.files if k.startswith('catboost_tuned__')]}")

    # We intentionally don't merge sequence_probabilities into the same file,
    # because stacking_v2 already auto-loads it from a separate sequence_dir.

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"wrote {args.output}")
    print(f"  keys: {sorted(k for k in payload if '__' in k)}")


if __name__ == "__main__":
    main()
