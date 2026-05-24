# Sleep Twin v2 — Training Summary

## Headline

The v2 pipeline beats the v1 best model (`xgboost_cuda`, macro-F1 = 0.6678) by a substantial margin on every metric.

| Metric | v1 best | v2 best | Delta |
|---|---:|---:|---:|
| Accuracy | 0.7646 | **0.7746** | **+1.00 pp** |
| Balanced accuracy | 0.6659 | **0.7159** | **+5.00 pp** |
| Macro F1 | 0.6678 | **0.6993** | **+3.15 pp** |
| Cohen's κ | 0.5050 | **0.5501** | **+4.52 pp** |
| Weighted F1 | 0.7651 | 0.7793 | +1.42 pp |

The strongest single model in v2 is **CatBoost smoothed with an HMM/Viterbi pass** trained from a transition matrix estimated on the training labels.

## What changed vs v1

1. **Feature engineering went from 318 → 540 features.** Key additions:
   - Per-subject robust z-scoring of heart-rate (`hr_*_bpm_z`) — strongest single lever, since baseline HR varies enormously between subjects
   - Subject-normalized motion magnitude (`motion_*_enmo_norm`)
   - HRV-style proxies on 120s / 300s / 1800s windows: range, successive-difference RMS, coefficient of variation
   - Wider rolling windows (added 300s and 3600s)
   - Activity proxies at finer timescales (15-min, 1 h, 3 h) replacing the 7-day lookback (which carried almost no signal)
2. **Model zoo upgraded.** Boosted-tree trio: XGBoost (CUDA), LightGBM, CatBoost — all with tuned hyperparameters and class-balanced weighting.
3. **HMM/Viterbi post-processing.** Predicted class probabilities for every test epoch are decoded against a transition matrix estimated from training labels (Laplace-smoothed) and per-subject initial probabilities. This cleans up the kind of single-epoch wake flickers that hurt macro F1 and kappa.
4. **Stacking meta-learner** — included for ablation; underperformed individual models in this configuration (see notes below).

## Full leaderboard

| Rank | Version | Model | Acc | Bal Acc | Macro F1 | Wtd F1 | Kappa |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | **v2** | **base_catboost_hmm** | 0.7746 | **0.7159** | **0.6993** | 0.7793 | **0.5501** |
| 2 | v2 | ensemble_mean (avg of XGB+LGBM+Cat) | 0.7704 | 0.6783 | 0.6785 | 0.7714 | 0.5178 |
| 3 | v2 | lightgbm | 0.7796 | 0.6605 | 0.6762 | 0.7761 | 0.5176 |
| 4 | v2 | lightgbm_hmm | **0.7897** | 0.6332 | 0.6687 | 0.7794 | 0.5153 |
| 5 | v1 | xgboost_cuda | 0.7646 | 0.6659 | 0.6678 | 0.7651 | 0.5050 |
| 6 | v2 | xgboost_cuda_hmm | 0.7766 | 0.6456 | 0.6667 | 0.7715 | 0.5079 |
| 7 | v2 | xgboost_cuda | 0.7588 | 0.6631 | 0.6646 | 0.7599 | 0.4934 |
| 8 | v2 | catboost (no HMM) | 0.7369 | 0.7041 | 0.6641 | 0.7470 | 0.4968 |
| 9 | v2 | ensemble_stacked_hmm | 0.7150 | 0.7058 | 0.6623 | 0.7278 | 0.4744 |
| 10 | v1 | hist_gradient_boosting | 0.7581 | 0.6490 | 0.6583 | 0.7574 | 0.4835 |
| 11 | v2 | ensemble_stacked | 0.6963 | 0.6936 | 0.6419 | 0.7109 | 0.4461 |
| 12 | v1 | bilstm_attention | 0.7279 | 0.6472 | 0.6305 | 0.7299 | 0.4327 |
| 13 | v1 | temporal_cnn | 0.6832 | 0.6694 | 0.6194 | 0.6959 | 0.4160 |
| 14 | v1 | tabular_mlp | 0.6877 | 0.6336 | 0.6175 | 0.6970 | 0.3818 |
| 15 | v1 | extra_trees | 0.7615 | 0.5440 | 0.5917 | 0.7240 | 0.3695 |

## Reading the leaderboard

- **CatBoost alone is the best minority-class learner** in v2 (bal_acc = 0.7041 — the highest of any single model), but its raw accuracy was the lowest of the trio (0.7369) because it predicts Wake/REM more aggressively at the cost of NREM precision. HMM smoothing cleans up the noisy transitions while preserving the minority-class recall, lifting accuracy by +3.8 pp to 0.7746 *without* dropping balanced accuracy meaningfully.
- **LightGBM** has the highest raw accuracy (0.7796, 0.7897 with HMM) but pays for it with weaker balanced accuracy — the classic majority-class bias.
- **HMM smoothing improves XGBoost and LightGBM on accuracy and kappa but hurts their balanced accuracy** — because those base models were already biased toward NREM and smoothing reinforces that. Only CatBoost benefits across the board from HMM, because its base predictions were the most class-balanced to begin with.
- **The equal-weight ensemble mean** (`ensemble_mean`) is a defensible second choice — macro-F1 of 0.6785 with kappa 0.5178 — and is robust to single-model failure modes.

## Stacked ensemble — why it underperformed

The logistic-regression meta-learner was fit on the validation split (~4 035 samples) with `class_weight="balanced"`. With only three base learners contributing 9 probability columns, the meta-learner over-corrected toward minority classes and lost overall accuracy. Once the BiLSTM is brought in (next iteration, see below) we'll re-evaluate stacking with cross-validated training probabilities and/or remove class-weighting from the meta-learner.

## BiLSTM (deferred)

The v2 BiLSTM trainer is implemented (`train_sequence.py`) — focal loss (γ=1.5), 31-epoch context (radius 15), 3-layer 128-hidden BiLSTM with attention pooling, OneCycleLR. It was not run for this report because the freshly-installed CUDA PyTorch wheel hit a `WinError 1114` DLL init error on the Windows host (likely a runtime conflict with a system library; the install verified clean and base imports work but `torch` itself fails at bootstrap). Given that v1's BiLSTM was *weaker* than v1's XGBoost on macro-F1 (0.6305 vs 0.6678) and we already beat v1 substantially with the booster trio, finalizing v2 without BiLSTM is justified.

If we resolve the torch DLL issue later, the BiLSTM probabilities can be slotted into the existing stacking pipeline — `stacking.py` already detects `sequence_probabilities.npz` automatically.

## Dataset and split

- Source: Walch et al. wrist-wearable + PSG dataset (same as v1)
- 26 773 labeled 30-s epochs across 31 subjects
- Class counts: Wake 2 429, NREM 18 460, REM 5 884 (imbalanced, NREM dominant)
- Subject-level `GroupShuffleSplit` (seed 42): 19 train / 5 val / 7 test — same split as v1, so leaderboards are directly comparable

## Reproducibility

```bash
cd sleep_twin_v2
.venv/Scripts/python.exe -m sleep_twin_v2.features                # build cache (~6 min)
.venv/Scripts/python.exe -m sleep_twin_v2.train_boosting          # train XGB+LGBM+Cat (~90 s)
.venv/Scripts/python.exe -m sleep_twin_v2.stacking                # stack + HMM smooth
.venv/Scripts/python.exe scripts/compare_to_v1.py                 # write v1_vs_v2_summary.md
```

Artifacts:

- Feature cache: `artifacts/features/sleep_features_v2.npz` (30 MB)
- Booster models: `artifacts/models/boosters/{xgboost_cuda,lightgbm,catboost}.joblib`
- Booster probabilities: `artifacts/models/boosters/boosting_probabilities.npz`
- Ensemble package: `artifacts/models/ensemble/ensemble.joblib` (meta-learner + HMM transitions)
- Reports: `artifacts/reports/training_summary.md`, `artifacts/reports/v1_vs_v2_summary.md`
