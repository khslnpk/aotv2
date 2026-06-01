# Sleep Twin

Wearable sleep-stage classifier + Digital Twin scoring layer.

Take 30-second epochs of wrist motion, heart rate, and step data, predict whether each epoch is **Wake / NREM / REM**, smooth the resulting hypnogram, and turn it into a virtual human-state model with six interpretable scores: **recovery, fatigue, stress, energy, sleep quality, sleep debt**.

## Headline accuracy (test set, subject-level split)

| Metric | Value | Model |
|---|---:|---|
| Accuracy | **0.8035** | geometric-mean ensemble + HMM |
| Balanced accuracy | **0.7159** | CatBoost + HMM |
| Macro F1 | **0.6993** | CatBoost + HMM |
| Cohen's κ | **0.5695** | geometric-mean ensemble + HMM |

Two co-winners depending on which metric you optimize. Full leaderboard in [artifacts/reports/training_summary.md](artifacts/reports/training_summary.md).

## Project layout

```
sleep-twin/
├── pyproject.toml
├── requirements.txt              base ML stack (numpy, pandas, sklearn, xgboost, lightgbm, catboost, hmmlearn, optuna, ...)
├── requirements-cuda.txt         optional CUDA PyTorch (for GPU BiLSTM training)
│
├── src/sleep_twin/
│   ├── paths.py                  path resolution + env-var overrides
│   ├── labels.py                 PSG -> 3-class (Wake/NREM/REM) mapping
│   ├── splits.py                 GroupShuffleSplit by subject_id
│   ├── features.py               feature builder (~540 features per epoch)
│   ├── evaluation.py             accuracy / balanced acc / macro F1 / kappa / confusion
│   ├── train_boosting.py         trains XGBoost + LightGBM + CatBoost
│   ├── tune_catboost.py          Optuna hyperparameter sweep on CatBoost
│   ├── train_sequence.py         trains BiLSTM with attention pooling + focal loss
│   ├── temporal_smoothing.py     HMM/Viterbi post-processor
│   ├── stacking.py               meta-learner + HMM ensemble
│   └── digital_twin.py           sleep summary + 6 human-state scores + what-if sims
│
├── scripts/
│   ├── train_all.py              end-to-end pipeline (features -> boosters -> tune -> BiLSTM -> stack)
│   └── merge_probabilities.py    helper to combine per-source probability files
│
└── artifacts/
    ├── features/sleep_features.npz       (.gitignored, ~30 MB; rebuild with `python -m sleep_twin.features`)
    ├── models/
    │   ├── boosters/                     (.gitignored) XGB/LGBM/CatBoost + probability cube
    │   ├── catboost_tuned/               (.gitignored) Optuna-tuned CatBoost + best params
    │   ├── sequence/                     (.gitignored) BiLSTM checkpoint + probabilities
    │   └── ensemble/                     (.gitignored) meta-learner + HMM + leaderboard.md
    └── reports/                          (tracked)
        └── training_summary.md
```

## Environments

Two Python environments are needed. The main venv handles everything except BiLSTM training. The conda sidecar exists because the main venv's Python 3.12 hit a `c10.dll` init conflict with PyTorch on Windows; isolating the LSTM training into a Python 3.11 conda env was the cleanest fix.

| Env | Python | Used for |
|---|---|---|
| `.venv/` (project-local) | 3.12 | features, boosters, Optuna, stacking, HMM, evaluation, scoring |
| `sleep-twin-torch` (conda) | 3.11 + CPU torch | BiLSTM training only |

### One-time setup

```bash
# Main venv
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt optuna
.venv/Scripts/python.exe -m pip install -e .

# Sidecar conda env (BiLSTM only)
conda create -y -n sleep-twin-torch python=3.11
C:\Users\user\anaconda3\envs\sleep-twin-torch\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
C:\Users\user\anaconda3\envs\sleep-twin-torch\python.exe -m pip install numpy pandas scikit-learn joblib tqdm
C:\Users\user\anaconda3\envs\sleep-twin-torch\python.exe -m pip install -e .
```

### Dataset path

The project uses the Walch et al. wrist-wearable + PSG dataset (~10 GB, not in this repo). By default `src/sleep_twin/paths.py` looks in this order:
1. `$SLEEP_TWIN_DATA_ROOT` env var
2. `./motion-and-heart-rate-...` next to this repo
3. `Desktop/AOT Project/motion-and-heart-rate-...` (historical location on this machine)

If you move the dataset, set the env var:
```powershell
$env:SLEEP_TWIN_DATA_ROOT = "X:\path\to\dataset"
```

## Running the pipeline

```bash
# Full pipeline: features -> boosters -> Optuna -> BiLSTM -> merge -> stack
.venv/Scripts/python.exe scripts/train_all.py
```

Or run stages individually:

```bash
# 1. Build feature cache (~6 min, 26 773 epochs x 540 features)
.venv/Scripts/python.exe -m sleep_twin.features

# 2. Train booster trio (~90 seconds with CUDA XGBoost)
.venv/Scripts/python.exe -m sleep_twin.train_boosting

# 3. Optuna sweep on CatBoost (~25 min CPU, 25 trials)
.venv/Scripts/python.exe -m sleep_twin.tune_catboost --n-trials 25

# 4. BiLSTM training (sidecar conda env, CPU)
C:\Users\user\anaconda3\envs\sleep-twin-torch\python.exe -u -m sleep_twin.train_sequence \
    --device cpu --epochs 20 --batch-size 512 --sequence-radius 7 \
    --patience 5 --hidden-size 64 --num-layers 1

# 5. Merge per-source probabilities into one .npz, then stack + HMM smooth
.venv/Scripts/python.exe scripts/merge_probabilities.py \
    --booster artifacts/models/boosters/boosting_probabilities.npz \
    --tuned-catboost artifacts/models/catboost_tuned/catboost_tuned_probabilities.npz \
    --output artifacts/models/boosters/boosting_probabilities.npz
.venv/Scripts/python.exe -m sleep_twin.stacking
```

The final leaderboard lands at `artifacts/models/ensemble/leaderboard.md` and the full writeup at [artifacts/reports/training_summary.md](artifacts/reports/training_summary.md).

## Running the app

A FastAPI server + single-page web app that visualizes any subject's predicted hypnogram, the six Digital Twin scores with full per-component breakdowns, an interactive what-if simulator, an Apple Health import path, and a methodology drawer.

```bash
.venv/Scripts/python.exe scripts/run_app.py
# -> http://127.0.0.1:8765
```

### Running from a freshly booted laptop

The steps below assume Windows + the `.venv/` already exists at the repo root (created during initial setup). If you haven't cloned yet, do that first:

```bash
git clone https://github.com/khslnpk/aotv2.git sleep-twin
cd sleep-twin
```

**Step 1 — One-time environment setup (only if `.venv/` does not already exist):**

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e .

# Sidecar conda env (only needed if you'll re-train the BiLSTM)
conda create -y -n sleep-twin-torch python=3.11
C:\Users\user\anaconda3\envs\sleep-twin-torch\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
C:\Users\user\anaconda3\envs\sleep-twin-torch\python.exe -m pip install numpy pandas scikit-learn joblib tqdm
C:\Users\user\anaconda3\envs\sleep-twin-torch\python.exe -m pip install -e .
```

**Step 2 — Make sure the trained model artifacts are present.** They are gitignored, so a fresh clone will not have them. Check:

```bash
ls artifacts/models/ensemble/ensemble.joblib
ls artifacts/models/boosters/boosting_probabilities.npz
ls artifacts/models/sequence/sequence_probabilities.npz
ls artifacts/features/sleep_features.npz
```

If any of those are missing, rebuild from scratch (~40 min total wall time on the spec machine):

```bash
# Make sure the dataset is reachable (env var or default location)
# $env:SLEEP_TWIN_DATA_ROOT = "C:\path\to\motion-and-heart-rate-from-a-wrist-worn-wearable-..."

.venv/Scripts/python.exe scripts/train_all.py
```

`train_all.py` runs the full pipeline: features → boosters (XGB/LGBM/CatBoost) → Optuna tune → BiLSTM (via the sidecar conda env) → merge probabilities → stacking + HMM. Add `--skip-tune` and/or `--skip-sequence` to shorten the run.

**Step 3 — Boot the app.** Two commands from a cold laptop:

```bash
cd "C:\Users\user\OneDrive\Desktop\sleep-twin"
.venv/Scripts/python.exe scripts/run_app.py
```

Then open **http://127.0.0.1:8765** in any modern browser.

**Useful flags:**

```bash
.venv/Scripts/python.exe scripts/run_app.py --host 0.0.0.0 --port 9000  # bind to LAN, custom port
.venv/Scripts/python.exe scripts/run_app.py --reload                    # auto-reload on code changes
```

**To stop:** press `Ctrl+C` in the terminal. The in-memory upload store (Apple Health imports) is cleared on shutdown — re-uploading takes a few seconds.

Architecture:

```
app/                        single-page frontend (no build step)
├── index.html
├── styles.css              aurora dark theme, glass bento, segmented dial
└── app.js                  vanilla — fetch, render, what-if, drawer

src/sleep_twin/inference.py SleepTwinService — loads cached probabilities,
                            applies HMM/Viterbi smoothing per subject,
                            computes Sleep + Physio + HumanState
src/sleep_twin/api.py       FastAPI endpoints: /api/subjects, /api/whatif, /
scripts/run_app.py          uvicorn launcher
```

The app does **not** retrain anything — it reuses the per-base-learner probability files already produced by the training pipeline, geometric-mean-averages them, HMM-smooths the result, and runs the Digital Twin scoring on top. Server boots in ~1s.

Score methodology (what feeds each of the 6 scores, with literature references) is documented at [artifacts/reports/digital_twin_methodology.md](artifacts/reports/digital_twin_methodology.md) and surfaced in the app's Methodology drawer (`M` key).

## What gets ignored by git

`.venv/`, `catboost_info/`, run logs, the feature cache (`artifacts/features/`), all trained model binaries (`artifacts/models/`), and the raw dataset folder. Only the source code, configuration, and small markdown reports are committed.

## Disclaimers

- The Digital Twin scores (recovery, fatigue, stress, energy, sleep quality, sleep debt) are **derived engineering estimates**, not medical predictions. The training dataset has no ground-truth labels for these.
- The classifier is trained on 31 subjects from the Walch et al. PhysioNet dataset; generalization to other wearable hardware or longer-form data is not validated.
