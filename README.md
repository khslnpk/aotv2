# Sleep Digital Twin v2

Successor project to the v1 prototype in the parent directory. Goal: keep the same idea (wearable sleep-stage classification → Digital Twin scores) but train a model with better accuracy/macro-F1/kappa than v1 and refactor the application layer for a faster runtime.

## What is different from v1

| Layer | v1 | v2 |
|---|---|---|
| Features | 318 (motion + HR + steps + time) | ~430 with per-subject HR/motion normalization, HRV proxies (range, succ-diff RMS, CV), wider rolling windows |
| Models | ExtraTrees, HistGB, XGBoost, MLP, Temporal CNN, BiLSTM | XGBoost + LightGBM + CatBoost + BiLSTM-attention (focal loss, wider context) + stacked logreg meta-learner |
| Smoothing | none | HMM/Viterbi post-processor on probability sequences |
| Class balance | sample-weight | sample-weight + focal loss (γ=1.5) |
| App | Streamlit | (planned) FastAPI + lightweight web client |

## Layout

```
sleep_twin_v2/
├── pyproject.toml
├── requirements.txt              # base
├── requirements-cuda.txt         # torch + cu128
├── src/sleep_twin_v2/
│   ├── paths.py
│   ├── labels.py                 # 3-class mapping
│   ├── splits.py                 # group-aware splits
│   ├── features.py               # enhanced feature builder
│   ├── evaluation.py
│   ├── temporal_smoothing.py     # HMM/Viterbi
│   ├── train_boosting.py         # XGBoost + LightGBM + CatBoost
│   ├── train_sequence.py         # BiLSTM-attn + focal loss
│   ├── stacking.py               # meta-learner + HMM combine
│   └── digital_twin.py
├── scripts/train_all.py
└── artifacts/{features,models,reports}/
```

## Quick start

```bash
# 1. Build feature cache
python -m sleep_twin_v2.features

# 2. Train everything in one shot
python scripts/train_all.py

# Or train individually
python -m sleep_twin_v2.train_boosting
python -m sleep_twin_v2.train_sequence
python -m sleep_twin_v2.stacking
```

Outputs land under `artifacts/models/{boosters,sequence,ensemble}/`. The final ensemble leaderboard is written to `artifacts/models/ensemble/leaderboard.md`.

## Dataset

Same Walch et al. wrist-wearable + PSG dataset as v1. Path resolves to `../motion-and-heart-rate-...` by default; override with `SLEEP_TWIN_DATA_ROOT`.
