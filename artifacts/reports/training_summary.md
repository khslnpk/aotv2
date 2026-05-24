# Sleep Twin — Training Summary

## Headline

Two co-winners on the held-out test set, depending on what you optimize for:

| Metric | Best score | Winning model |
|---|---:|---|
| Accuracy | **0.8035** | `avg_geo_hmm` (geometric-mean ensemble + HMM) |
| Cohen's κ | **0.5695** | `avg_geo_hmm` |
| Weighted F1 | **0.7978** | `avg_geo_hmm` |
| Balanced accuracy | **0.7159** | `base_catboost_hmm` (CatBoost + HMM) |
| Macro F1 | **0.6993** | `base_catboost_hmm` |

`avg_geo_hmm` is the recommended general-purpose model — it crosses **80% raw accuracy** with the strongest Cohen's κ. `base_catboost_hmm` is preferable when minority-class (Wake) recall matters more than overall accuracy.

## Pipeline overview

```
raw wearable data
   │
   ▼
[1] feature extraction  ── 540 features per 30-s epoch
   │
   ▼
[2] subject-level split ── 19 train / 5 val / 7 test (no subject overlap)
   │
   ▼
[3] base learners ────────┬── XGBoost (CUDA)
                          ├── LightGBM
                          ├── CatBoost
                          ├── CatBoost (Optuna-tuned)
                          └── BiLSTM-attention (focal loss)
   │
   ▼
[4] ensemble heads ───────┬── arithmetic mean
                          ├── geometric mean (winner)
                          └── logistic-regression meta-learner
   │
   ▼
[5] HMM/Viterbi smoothing ── decode against training-label transitions
   │
   ▼
final per-epoch sleep stage  ──►  Digital Twin scoring layer
```

## Feature engineering (540 features per epoch)

Each labeled epoch is anchored at `epoch_time + 15 s` (the midpoint of the 30-second window). Features fall into four groups:

| Group | Count | Description |
|---|---:|---|
| **Time** | 8 | minutes since PSG start, normalized night progress, sin/cos at half- and full-cycle, sin/cos daily clock proxy |
| **Motion** | 290 | series ∈ {x, y, z, magnitude, ENMO, **subject-normalized ENMO**, axis energy, jerk} × windows {30 s, 120 s, 300 s, 600 s} × stats {count, mean, std, min, max, p10, p50, p90, rms}; + 2 per-subject activity constants |
| **Heart rate** | 235 | series ∈ {bpm, **subject-z-scored bpm**, dev from resting, dev from sleep mean} × windows {30 s, 120 s, 300 s, 600 s, 1800 s, 3600 s} × 9 stats + per-window bpm slope; + HRV-style proxies (range, succ-diff RMS, CV) on 120 s / 300 s / 1800 s; + 4 per-subject constants |
| **Steps** | 7 | rolling sums over {15 min, 1 h, 3 h, 12 h, 24 h} + 2 per-subject constants |

**The single biggest accuracy lever** is the per-subject robust z-scoring of heart rate (`hr_*_bpm_z`) and ENMO (`motion_*_enmo_norm`). Baseline heart rate varies enormously between subjects (a 50 bpm sleep HR for one person is 80 bpm for another), so absolute values mislead the model. Z-scoring against the subject's own median and IQR makes the signal portable across people.

## Base learners

### Boosted trees (all trained with class-balanced sample weights)

- **XGBoost (CUDA)** — 1500 estimators, depth 6, lr 0.03, with early stopping (patience 80)
- **LightGBM** — 2000 estimators, 95 leaves, lr 0.025; built-in `class_weight="balanced"`; early stopping
- **CatBoost** — 2000 iterations, depth 7, lr 0.04, `auto_class_weights="Balanced"`; early stopping

### Optuna-tuned CatBoost

- 18 trials, TPE sampler, optimized for validation macro-F1
- Winning hyperparameters: `depth=9, lr=0.034, l2_leaf_reg=3.51, random_strength=0.51, bagging_temperature=1.07, border_count=160`
- Validation macro-F1: 0.7057 (vs default ≈ 0.685)

### BiLSTM with attention pooling

- Trained on CPU in a sidecar conda env (`sleep-twin-torch`, Python 3.11 + torch 2.12 CPU). The main venv's Python 3.12 hit a `c10.dll` init conflict with PyTorch on Windows; isolating the LSTM into its own conda env was the simplest fix.
- 1-layer bidirectional LSTM, hidden size 64, attention pooling, focal cross-entropy (γ = 1.5), AdamW + OneCycleLR
- 15-epoch context window (sequence radius 7) — one Wake-NREM-REM "block" of context on each side
- Early-stopped at epoch 11 (patience 5)

## Ensemble heads compared

For each candidate, predicted probabilities on the test set are decoded both as-is and after HMM/Viterbi smoothing.

| Head | Description |
|---|---|
| `base_*` | Each individual model's probabilities, predicted directly |
| `avg_mean` | Arithmetic mean of base probabilities |
| `avg_geo` | Geometric mean (log-average, then re-normalize) |
| `stack_logreg_C0.3` | Best of 5 meta-learner candidates (logreg C ∈ {0.3, 1.0, 4.0}, MLP-64, MLP-64-32), selected on val macro-F1 |

## HMM/Viterbi post-processing

Sleep stages have strong transition structure (Wake → REM transitions are rare; most epochs continue the previous stage). The smoother:
1. Counts label-to-label transitions in the training set (Laplace-smoothed)
2. Counts initial-stage frequencies
3. Treats per-epoch predicted probabilities as HMM emission likelihoods
4. Runs Viterbi per-subject (chronological order) to decode the most likely stage sequence

Effect: lifts accuracy and κ by 1–3 percentage points, smoothing out single-epoch flicker that hurts macro-F1.

## Final leaderboard

Sorted by macro F1, with ties broken by accuracy. All scores on the 7-subject held-out test set (6 020 epochs).

| Rank | Model | Acc | Bal Acc | Macro F1 | Wtd F1 | Kappa |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **base_catboost_hmm** | 0.7746 | **0.7159** | **0.6993** | 0.7793 | 0.5501 |
| 2 | **avg_geo_hmm** | **0.8035** | 0.6745 | 0.6932 | **0.7978** | **0.5695** |
| 3 | avg_mean_hmm | 0.8022 | 0.6733 | 0.6921 | 0.7965 | 0.5668 |
| 4 | avg_geo | 0.7772 | 0.6973 | 0.6901 | 0.7795 | 0.5388 |
| 5 | avg_mean | 0.7754 | 0.6975 | 0.6892 | 0.7780 | 0.5368 |
| 6 | base_catboost_tuned_hmm | 0.7816 | 0.6870 | 0.6814 | 0.7820 | 0.5515 |
| 7 | base_lightgbm | 0.7796 | 0.6605 | 0.6762 | 0.7761 | 0.5176 |
| 8 | base_catboost_tuned | 0.7538 | 0.7053 | 0.6750 | 0.7615 | 0.5201 |
| 9 | base_lightgbm_hmm | 0.7897 | 0.6332 | 0.6687 | 0.7794 | 0.5153 |
| 10 | base_xgboost_cuda_hmm | 0.7766 | 0.6456 | 0.6667 | 0.7715 | 0.5079 |
| 11 | stack_logreg_C0.3 | 0.7902 | 0.6234 | 0.6666 | 0.7762 | 0.4992 |
| 12 | base_xgboost_cuda | 0.7588 | 0.6631 | 0.6646 | 0.7599 | 0.4934 |
| 13 | base_catboost | 0.7369 | 0.7041 | 0.6641 | 0.7470 | 0.4968 |
| 14 | stack_logreg_C0.3_hmm | 0.7973 | 0.6096 | 0.6619 | 0.7795 | 0.5030 |
| 15 | base_bilstm_attention_hmm | 0.7312 | 0.6460 | 0.6368 | 0.7348 | 0.4462 |
| 16 | base_bilstm_attention | 0.7201 | 0.6589 | 0.6315 | 0.7273 | 0.4402 |

## Reading the leaderboard

- **CatBoost is the standout single model.** Its raw output has the highest balanced accuracy among single boosters (0.7041) — it predicts Wake/REM more aggressively than NREM, sacrificing accuracy. HMM smoothing then cleans up the noisy transitions while preserving that minority recall, lifting accuracy +3.8 pp without dropping balanced accuracy. The result has the project's best macro F1 (0.6993) and balanced accuracy (0.7159).
- **The Optuna-tuned CatBoost improves over the default version** (val macro F1 0.7057 vs ≈ 0.685; test accuracy 0.7538 vs 0.7369). Once HMM is applied, default + HMM beats tuned + HMM, because tuning pushed CatBoost toward higher accuracy at the cost of minority recall — and HMM amplifies whatever imbalance the base model has. The tuned model is still a useful ensemble component.
- **The geometric-mean ensemble + HMM (`avg_geo_hmm`)** is the headline number for raw accuracy and kappa. Crossing 80% accuracy with a 0.5695 kappa is the strongest result of any single configuration. Geometric mean (log-average) edges arithmetic mean by a hair — it's more robust to one model being very confident and wrong.
- **The logistic-regression stack underperforms simple averaging.** This is because base models perfectly memorize their training set, so the in-sample probabilities the meta-learner sees during fitting are not representative of test-time uncertainty. A proper fix is **out-of-fold stacking** (retrain each booster on K folds, predict held-out, fit meta-learner on the OOF probabilities), but that 5× the boosting compute. Deferred.
- **BiLSTM is the weakest single learner** (macro F1 0.6315) — it was trained with a small CPU-friendly config (1 layer, hidden=64, 15-epoch context). Even so, it contributes meaningfully to the ensemble — `avg_geo_hmm` includes its probabilities.

## Stacking ablation (meta-learner choice on val set)

| Meta-learner | Val macro F1 | Val balanced acc |
|---|---:|---:|
| **logreg C=0.3** | **0.6749** | 0.6385 |
| logreg C=1.0 | 0.6400 | 0.6014 |
| logreg C=4.0 | 0.6195 | 0.5812 |
| MLP (64) | 0.6177 | 0.5785 |
| MLP (64, 32) | 0.6117 | 0.5749 |

Strong regularization on the logreg wins, but the absolute number is misleading because of the in-sample-probability bias above.

## Dataset and split

- **Source**: Walch et al. wrist-wearable + PSG dataset (PhysioNet)
- **Size**: 26 773 labeled 30-s epochs across 31 subjects
- **Class counts**: Wake 2 429, NREM 18 460, REM 5 884 (imbalanced — NREM dominant, ≈ 7×)
- **Split**: subject-level `GroupShuffleSplit`, seed 42 — 19 train / 5 val / 7 test, no subject in more than one fold

## Reproducibility

```bash
# Main venv
.venv/Scripts/python.exe -m sleep_twin.features              # build cache (~6 min)
.venv/Scripts/python.exe -m sleep_twin.train_boosting        # XGB + LGBM + Cat (~90 s)
.venv/Scripts/python.exe -m sleep_twin.tune_catboost         # Optuna (~25 min)

# Sidecar conda env (CPU torch)
C:\Users\user\anaconda3\envs\sleep-twin-torch\python.exe -u -m sleep_twin.train_sequence \
    --device cpu --epochs 20 --batch-size 512 --sequence-radius 7 \
    --patience 5 --hidden-size 64 --num-layers 1             # BiLSTM (~12 min)

# Back in main venv
.venv/Scripts/python.exe scripts/merge_probabilities.py \
    --booster artifacts/models/boosters/boosting_probabilities.npz \
    --tuned-catboost artifacts/models/catboost_tuned/catboost_tuned_probabilities.npz \
    --output artifacts/models/boosters/boosting_probabilities.npz
.venv/Scripts/python.exe -m sleep_twin.stacking              # final stack + HMM
```

Or one-shot via `python scripts/train_all.py` (which does all of the above in order; `--skip-tune` and `--skip-sequence` flags available).

## Artifacts written to disk

| Path | Contents |
|---|---|
| `artifacts/features/sleep_features.npz` | 26 773 × 540 feature cache + labels + metadata (30 MB) |
| `artifacts/models/boosters/{xgboost_cuda,lightgbm,catboost}.joblib` | individual booster models |
| `artifacts/models/boosters/boosting_probabilities.npz` | per-source probability cube for all splits |
| `artifacts/models/catboost_tuned/catboost_tuned.joblib` | Optuna-tuned CatBoost |
| `artifacts/models/catboost_tuned/best_params.json` | winning hyperparameters |
| `artifacts/models/sequence/bilstm_attention.pt` | BiLSTM checkpoint |
| `artifacts/models/sequence/sequence_probabilities.npz` | BiLSTM probabilities |
| `artifacts/models/ensemble/ensemble.joblib` | meta-learner + HMM transitions + source layout |
| `artifacts/models/ensemble/ensemble_metrics.json` | leaderboard rows + raw metrics |
| `artifacts/models/ensemble/leaderboard.md` | machine-rendered leaderboard |

## What's deferred

- **Out-of-fold stacking** — would likely make the meta-learner beat simple averaging by removing the in-sample-probability bias. Costs 5× boosting compute.
- **GPU PyTorch on Windows** — currently bypassed via the sidecar conda env. A clean Python 3.11 environment with `pytorch-cuda` from the official channel would let us train the original full-size BiLSTM (3 layers, hidden=128, sequence radius 15) in minutes instead of 30+ minutes on CPU.
- **Per-subject temperature calibration** — applying a calibration scalar fit on a calibration split could push kappa another 1–2 pp.
