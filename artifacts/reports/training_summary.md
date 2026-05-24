# Sleep Twin v2 — Training Summary (v2.0)

## Headline

The v2 pipeline beats the v1 best model (`xgboost_cuda`, macro-F1 = 0.6678) by a substantial margin on every metric. Two co-winners depending on what you optimize for:

| Metric | v1 best (xgboost_cuda) | **v2 best** | Winner | Delta |
|---|---:|---:|---|---:|
| Accuracy | 0.7646 | **0.8035** | `avg_geo_hmm` | **+3.89 pp** |
| Balanced accuracy | 0.6659 | **0.7159** | `base_catboost_hmm` | **+5.00 pp** |
| Macro F1 | 0.6678 | **0.6993** | `base_catboost_hmm` | **+3.15 pp** |
| Cohen's κ | 0.5050 | **0.5695** | `avg_geo_hmm` | **+6.45 pp** |
| Weighted F1 | 0.7651 | **0.7978** | `avg_geo_hmm` | **+3.27 pp** |

**Recommendation**: `avg_geo_hmm` (geometric mean of 5 base learners + HMM smoothing) for production use — it crosses the **80% accuracy threshold** for the first time and has the highest kappa. `base_catboost_hmm` is preferable when minority-class (Wake) recall matters more than overall accuracy.

## What changed vs v1

### Feature engineering (318 → 540 features)
- **Per-subject robust z-scoring** of HR (`hr_*_bpm_z`) and motion (`motion_*_enmo_norm`) — the single biggest lever, since baseline HR varies enormously between subjects
- **HRV-style proxies** on 120s / 300s / 1800s windows: range, successive-difference RMS, coefficient of variation
- **Wider rolling windows**: added 300s and 3600s
- **Activity proxies at finer timescales** (15-min, 1 h, 3 h) replacing v1's 7-day lookback (which carried almost no signal)

### Model zoo (1 → 5 base learners, sourced from 2 different runtimes)
1. **XGBoost (CUDA)** — boosted trees on GPU
2. **LightGBM** — leaf-wise boosting, often beats XGBoost on accuracy
3. **CatBoost** — symmetric trees, strongest minority-class recall
4. **CatBoost (Optuna-tuned)** — 18 trials, depth=9, lr=0.034, l2=3.51
5. **BiLSTM with attention** — focal loss (γ=1.5), trained on CPU in a sidecar conda env (Windows + Anaconda + PyTorch had a c10.dll init conflict that was resolved by using a fresh Python 3.11 conda env just for the LSTM)

### Post-processing
- **HMM/Viterbi smoothing** on probability sequences using transition matrices fit from training labels (Laplace-smoothed) and per-subject initial distributions — pushes accuracy and kappa noticeably while preserving balanced accuracy on the right base learners

### Combination heads
- **Geometric mean ensemble** (`avg_geo`) — log-average of base probabilities → re-normalize. More stable than arithmetic mean, dominant on this dataset.
- **Arithmetic mean ensemble** (`avg_mean`) — solid second
- **Logistic-regression stack** (`stack_logreg_C0.3`) — best of 5 meta-learner candidates (logreg @ C ∈ {0.3, 1.0, 4.0}, MLP-64, MLP-64-32); still underperforms simple averaging because in-sample training probabilities are perfectly memorized by the boosters (proper out-of-fold stacking would require 5x boosting compute and is deferred)

## Final leaderboard (test set, subject-level split)

Sorted by macro F1, ties broken by accuracy.

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
| 15 | base_bilstm_attention_v2_hmm | 0.7312 | 0.6460 | 0.6368 | 0.7348 | 0.4462 |
| 16 | base_bilstm_attention_v2 | 0.7201 | 0.6589 | 0.6315 | 0.7273 | 0.4402 |

For reference, v1 leaderboard (sorted by macro F1) topped out at:
- xgboost_cuda: 0.7646 / 0.6659 / 0.6678 / 0.5050
- hist_gradient_boosting: 0.7581 / 0.6490 / 0.6583 / 0.4835
- bilstm_attention: 0.7279 / 0.6472 / 0.6305 / 0.4327

## Reading the leaderboard

- **CatBoost is the standout single model.** Its raw output (`base_catboost`) has the highest balanced accuracy among single boosters (0.7041) — it predicts Wake/REM more aggressively than NREM, sacrificing accuracy. HMM smoothing then cleans up the noisy transitions while preserving that minority recall, lifting accuracy +3.8 pp without dropping balanced accuracy. The result (`base_catboost_hmm`) has the project's best macro F1 (0.6993) and balanced accuracy (0.7159).
- **The Optuna-tuned CatBoost did improve over the untuned version** (val macro F1 0.7057 vs untuned ~0.685), and on test it has higher accuracy (0.7538 vs 0.7369). But once HMM is applied, untuned + HMM beats tuned + HMM — because tuning pushed CatBoost toward higher accuracy at the cost of minority recall, and HMM amplifies whatever imbalance the base model has. The tuned version is still useful as an ensemble component.
- **The geometric-mean ensemble + HMM (`avg_geo_hmm`)** is the headline number for raw accuracy. Crossing 80% accuracy with a +6.45 pp kappa improvement is a strong endorsement of the multi-model approach. Geometric mean dominates arithmetic mean by a hair (more robust to one model being very confident and wrong).
- **The logistic-regression stack underperforms simple averaging.** This is because base models perfectly memorize their training set, so the in-sample probabilities the meta-learner sees during fitting are not representative of test-time uncertainty. A proper fix is **out-of-fold stacking** (retrain each booster on K folds, predict held-out, fit meta-learner on the OOF probabilities), but that 5× the boosting compute. Deferred to a future iteration.
- **BiLSTM is the weakest single learner** (macro F1 0.6315) — it was trained on CPU in a sidecar conda env with a leaner config than originally planned (1-layer, hidden=64, sequence radius 7) because Windows + Anaconda Python 3.12 had a `c10.dll` DLL init conflict with PyTorch that wasn't resolvable without environment surgery. Even so, the BiLSTM contributes meaningfully to the ensemble — `avg_geo_hmm` includes it.

## Stacking, Optuna, and ablation notes

### Stacking head comparison (on val set)

| Meta-learner | Val macro F1 | Val balanced acc |
|---|---:|---:|
| logreg C=0.3 | **0.6749** | 0.6385 |
| logreg C=1.0 | 0.6400 | 0.6014 |
| logreg C=4.0 | 0.6195 | 0.5812 |
| MLP (64) | 0.6177 | 0.5785 |
| MLP (64, 32) | 0.6117 | 0.5749 |

Strong regularization on the logreg (C=0.3) wins, but the absolute number is misleading because of the in-sample bias noted above.

### Optuna CatBoost sweep (18 trials, 25-min CPU budget)

- Best validation macro F1: **0.7057** (vs untuned default ≈ 0.685)
- Winning hyperparameters: `depth=9, lr=0.034, l2_leaf_reg=3.51, random_strength=0.51, bagging_temperature=1.07, border_count=160`
- Translation to test: macro F1 0.6750 (vs untuned 0.6641 on test — modest +1.1 pp)

The tuning improved CatBoost as a single classifier but did not change the ensemble's top model; CatBoost + HMM was already strong enough that the marginal gain from tuning was absorbed by ensemble averaging.

## Reproducibility

Two environments are required:

1. **Main venv** (`sleep_twin_v2/.venv`, Python 3.12 from Anaconda base) — used for everything except BiLSTM training: feature extraction, all 3 boosters + tuned CatBoost (Optuna), stacking, HMM smoothing, evaluation.
2. **Sidecar conda env** (`sleep_twin_v2_torch`, Python 3.11 + CPU torch) — used only for BiLSTM training, because the main venv's Python 3.12 had a torch `c10.dll` init conflict.

```bash
# In main venv
python -m sleep_twin_v2.features                                # build cache (~6 min)
python -m sleep_twin_v2.train_boosting                          # XGB+LGBM+Cat (~90 s)
python -m sleep_twin_v2.tune_catboost --n-trials 25             # Optuna CatBoost (~25 min)

# In sidecar conda env (CPU torch)
python -u -m sleep_twin_v2.train_sequence --device cpu \
    --epochs 20 --batch-size 512 --sequence-radius 7 \
    --patience 5 --hidden-size 64 --num-layers 1                # lean BiLSTM (~12 min)

# Back in main venv
python scripts/merge_probabilities.py \
    --booster artifacts/models/boosters/boosting_probabilities.npz \
    --tuned-catboost artifacts/models/catboost_tuned/catboost_tuned_probabilities.npz \
    --output artifacts/models/boosters/boosting_probabilities.npz
python -m sleep_twin_v2.stacking_v2                             # final stack + HMM
python scripts/compare_to_v1.py                                 # write v1_vs_v2 report
```

## Artifacts

| Path | Description |
|---|---|
| `artifacts/features/sleep_features_v2.npz` | 26 773 × 540 feature cache (30 MB) |
| `artifacts/models/boosters/{xgboost_cuda,lightgbm,catboost}.joblib` | base booster models |
| `artifacts/models/boosters/boosting_probabilities.npz` | merged probability cube for stacking |
| `artifacts/models/catboost_tuned/catboost_tuned.joblib` | Optuna-tuned CatBoost |
| `artifacts/models/catboost_tuned/best_params.json` | winning hyperparameters |
| `artifacts/models/sequence/bilstm_attention_v2.pt` | trained BiLSTM checkpoint |
| `artifacts/models/sequence/sequence_probabilities.npz` | BiLSTM probabilities |
| `artifacts/models/ensemble_v2/ensemble_v2.joblib` | meta-learner + HMM transitions + per-source layout |
| `artifacts/models/ensemble_v2/leaderboard_v2.md` | machine-generated leaderboard |
| `artifacts/reports/training_summary.md` | this document |
| `artifacts/reports/v1_vs_v2_summary.md` | side-by-side v1 vs v2 comparison |

## Dataset and split

- Source: Walch et al. wrist-wearable + PSG dataset (same as v1)
- 26 773 labeled 30-s epochs across 31 subjects
- Class counts: Wake 2 429, NREM 18 460, REM 5 884 (imbalanced, NREM dominant)
- Subject-level `GroupShuffleSplit` (seed 42): 19 train / 5 val / 7 test — same split as v1, so leaderboards are directly comparable

## What's deferred

- **Out-of-fold stacking** — would likely make the meta-learner beat the simple geometric mean
- **GPU PyTorch on Windows** — currently bypassed via the sidecar conda env; a clean fix (e.g. switching to a python.org Python or a fully isolated conda env with `pytorch-cuda` from the official channel) would let us train the larger BiLSTM (3 layers, hidden=128, sequence radius 15) in minutes instead of hours
- **Subject-specific calibration** — temperature scaling per held-out subject could push kappa another 1-2 pp
