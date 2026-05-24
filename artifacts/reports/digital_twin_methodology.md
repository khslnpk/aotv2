# Digital Twin Methodology

The Digital Twin layer turns a predicted sleep hypnogram (and, optionally, raw heart-rate and accelerometer signals) into six interpretable human-state scores: **sleep_quality, sleep_debt, recovery, stress_index, fatigue, energy**.

Every score is a weighted sum of **named, individually-bounded components**, each of which traces back to a specific measurable signal and a reference value drawn from sleep medicine or wearable-device literature. The full breakdown for each score is exposed via `HumanState.components` so a UI can show "this is what this number is made of."

This document specifies:

1. The signals available to the scoring layer
2. The exact formula and ingredient list for each score
3. The target ranges and references each component is benchmarked against
4. What the scores **cannot** represent given the Walch et al. PhysioNet dataset

---

## 1. Available signals

| Source | Where it comes from | Sampling | Used in |
|---|---|---|---|
| Predicted hypnogram (Wake / NREM / REM, 30-s epochs) | classifier ensemble + HMM smoothing | 30 s | every score |
| Wrist heart rate | `{subject}_heartrate.txt` | seconds | recovery, stress, fatigue |
| Wrist motion (accelerometer ENMO) | `{subject}_acceleration.txt` | ~50 Hz | sleep_quality (movement context), stress |
| Step counts | `{subject}_steps.txt` | per minute | fatigue, stress (as daily activity load) |

We do not have: SpO2, body temperature, respiratory rate, multi-night history, subjective ratings (PSQI/KSS/RPE), demographics (age, sex, fitness baseline). The scores are therefore **physiological-correlate estimates**, not diagnostic or validated medical predictions.

---

## 2. Derived sleep architecture (`SleepSummary`)

Computed in `summarize_3class_sleep`:

| Field | How it's computed |
|---|---|
| `total_time_in_bed_min` | total epochs × 0.5 |
| `total_sleep_time_min` (TST) | (NREM + REM) epochs × 0.5 |
| `wake_min` / `nrem_min` / `rem_min` | per-class minute counts |
| `sleep_efficiency` | TST / TIB |
| `rem_fraction_of_sleep` | REM / TST |
| `nrem_fraction_of_sleep` | NREM / TST |
| `sleep_onset_latency_min` | minutes from recording start to first non-Wake epoch |
| `rem_latency_min` | minutes from sleep onset to first REM epoch |
| `waso_min` | wake minutes occurring *after* first sleep onset |
| `awakenings` | count of (Wake epoch following a sleep epoch), post-onset |
| `n_cycles` | count of NREM→REM transitions (proxy for completed sleep cycles) |

## 3. Derived physiology (`PhysioSummary`)

Computed in `summarize_physio` from raw HR (and optionally motion) windowed against the predicted sleep epochs:

| Field | How it's computed |
|---|---|
| `pre_sleep_hr_bpm` | mean HR in the 10 min before first sleep epoch (falls back to all pre-sleep HR, then to the 75th percentile of the recording) |
| `avg_sleep_hr_bpm` | mean HR during predicted non-Wake epochs |
| `min_sleep_hr_bpm` | 5th percentile of HR during predicted sleep (robust minimum) |
| `hr_dip_pct` | (pre_sleep_hr − min_sleep_hr) / pre_sleep_hr — magnitude of overnight HR drop |
| `hr_drift_bpm_per_hour` | linear slope of HR through the sleep period |
| `hr_succdiff_rms` | RMS of successive HR differences during sleep — proxy for HRV |
| `sleep_movement_index` | mean ENMO during sleep epochs |

**Honesty note on HRV.** True RMSSD requires beat-to-beat R-R intervals. The dataset gives us sampled HR (typically a few seconds between samples), so `hr_succdiff_rms` is a coarse proxy that correlates with HRV but is not equivalent to it. We label it `hrv_proxy` everywhere and report units of bpm, not ms.

---

## 4. Score: `sleep_quality` (0–100)

A 100-point weighted sum of seven components. Each component caps at its allotted points; no component can drag the score below zero.

| Component | Max pts | Source signal | Target / threshold | Reference |
|---|---:|---|---|---|
| Duration | 25 | TST | 7–9 h (band) | Hirshkowitz et al. 2015 (NSF) |
| Efficiency | 20 | TST / TIB | ≥ 0.85 (saturating) | Buysse et al. 1989 (PSQI) |
| REM balance | 15 | REM / TST | 0.20 – 0.25 (band, soft 0.5) | Carskadon & Dement 2005 |
| NREM balance | 10 | NREM / TST | 0.55 – 0.65 (band, soft 0.5) | Carskadon & Dement 2005 |
| WASO | 10 | wake after onset | full points at 0 min; zero at ≥ 90 min | Reed & Sacco 2016 |
| Onset latency | 10 | min to first sleep | 10 – 20 min (band, soft 0.8). Both very long (insomnia) and very short (sleep deprivation) onsets are penalized. | Buysse et al. 1989 |
| Cycle count | 10 | NREM→REM transitions | 4 – 6 cycles (band, soft 0.5) | Carskadon & Dement 2005 |

A "band" score returns 1.0 inside the target range and decays smoothly outside; the `soft` parameter controls how forgiving the slope is. A "saturating" score is a linear ramp from 0 to 1 that caps at 1.

---

## 5. Score: `sleep_debt` (0–100)

Single-night deficit only:

```
sleep_debt = clip((480 − TST_min) / 480 × 100, 0, 100)
```

**Important caveat.** True sleep debt accumulates across consecutive nights (Van Dongen et al. 2003 — *Cumulative cost of additional wakefulness*). The Walch dataset gives us **one night per subject**, so multi-night debt cannot be computed from training data alone. The UI should label this score "tonight's deficit" rather than "sleep debt" to avoid overclaiming.

---

## 6. Score: `recovery` (0–100)

Two regimes depending on whether physiology was supplied.

### With heart-rate data (preferred — total = 100)

| Component | Max pts | Source signal | Why |
|---|---:|---|---|
| Sleep contribution | 40 | `sleep_quality` | Sleep is the single largest driver of next-day recovery (Walker 2017, *Why We Sleep*). |
| HR dip | 25 | `hr_dip_pct` (target ≥ 0.10) | Larger dip = parasympathetic dominance during NREM (Trinder et al. 2001). |
| HRV proxy | 20 | `hr_succdiff_rms` (target ≥ 8 bpm-units) | Higher vagal tone = better recovery (Plews et al. 2013). |
| HR stability | 15 | absence of upward drift; penalty saturates at +5 bpm/hr | Climbing HR through the night indicates incomplete recovery (Whoop methodology). |

### Without heart-rate data (fallback — total still 100)

| Component | Max pts | Source |
|---|---:|---|
| Sleep contribution | 40 | `sleep_quality` |
| Autonomic estimate | 60 | `resting_hr_delta` argument (bpm above baseline) |

---

## 7. Score: `stress_index` (0–100)

| Component | Max pts | Source signal | Why |
|---|---:|---|---|
| Autonomic arousal | 30 | inverse of `hr_dip_pct` | Shallow overnight dip = sustained sympathetic tone (Kim et al. 2018). |
| Low HRV | 25 | inverse of `hr_succdiff_rms` | Reduced HRV correlates with psychological and physiological stress (Kim et al. 2018). |
| Overnight HR climb | 15 | positive `hr_drift_bpm_per_hour` | Rising HR through the night = sympathetic activation. |
| Activity load | 15 | prior-day steps vs 14k baseline | High physical load increases sympathetic load. |
| Sleep inefficiency | 15 | `(1 − efficiency / 0.85)` | Poor sleep elevates next-day stress markers. |

Fallback (no physio): `hr_elevation_estimate` (70 pts) + `activity_load` (15) + `sleep_inefficiency` (15).

---

## 8. Score: `fatigue` (0–100)

| Component | Max pts | Source |
|---|---:|---|
| Low recovery | 40 | `100 − recovery` |
| Activity load | 20 | prior-day steps |
| Sleep deficit | 20 | minutes under 8 h, capped at a 180-min deficit |
| Fragmentation | 10 | awakenings, capped at 10 |
| Overnight strain | 10 | positive HR drift; neutral 5/10 when physio unavailable |

---

## 9. Score: `energy` (0–100)

A thin composite of the other scores — felt-sense energy is largely explained by recovery and fatigue together.

```
energy = 0.50 × recovery + 0.30 × (100 − fatigue) + 0.20 × sleep_quality
```

---

## 10. What this dataset cannot support

Honest list of dimensions that the audience may ask about, but that we **cannot** compute from Walch et al. no matter what feature engineering we do:

| Wanted | Required signal | Why we can't |
|---|---|---|
| Apnea-related fatigue | SpO2 / oxygen desaturation index | not in dataset |
| Circadian misalignment | core body temperature / melatonin proxy | not in dataset |
| Respiratory recovery | respiratory rate | not in dataset |
| True multi-night sleep debt | ≥ 3 consecutive nights per subject | one night per subject |
| **Score validation** | subjective KSS, PSQI, RPE, mood labels | no ground truth for any of the 6 scores |
| Age- / sex- / fitness-adjusted normals | demographics | not in dataset |
| True HRV (RMSSD in ms) | beat-to-beat R-R intervals | only sampled HR is provided |

These limits frame the demo answer honestly: the scores combine the strongest physiological correlates available in the dataset, but they remain **engineering estimates**, not validated diagnostics.

---

## 11. UI integration

`HumanState.components` is a nested dict mirroring the tables above. A UI can render it as a per-score "info" panel:

```
Recovery: 72
  Sleep contribution ............ 30 / 40
  HR dip ........................ 18 / 25
  HRV proxy ..................... 14 / 20
  HR stability .................. 10 / 15
```

This answers the demo's "what is recovery *made of*?" question by construction. Every visible number can be drilled into to show its underlying signal and its target reference.

---

## 12. References

- Buysse, D. J. et al. (1989). The Pittsburgh Sleep Quality Index. *Psychiatry Research*, 28(2).
- Carskadon, M. A. & Dement, W. C. (2005). Normal human sleep: an overview. In *Principles and Practice of Sleep Medicine* (4th ed.).
- Hirshkowitz, M. et al. (2015). National Sleep Foundation's sleep time duration recommendations. *Sleep Health*, 1(1).
- Kim, H. G. et al. (2018). Stress and heart rate variability: a meta-analysis and review of the literature. *Psychiatry Investigation*, 15(3).
- Plews, D. J. et al. (2013). Heart rate variability in elite athletes. *Sports Medicine*, 43.
- Reed, D. L. & Sacco, W. P. (2016). Measuring sleep efficiency: what should the denominator be? *Journal of Clinical Sleep Medicine*, 12(2).
- Trinder, J. et al. (2001). Autonomic activity during human sleep. *Journal of Sleep Research*, 10(4).
- Van Dongen, H. P. A. et al. (2003). The cumulative cost of additional wakefulness. *Sleep*, 26(2).
- Walker, M. (2017). *Why We Sleep*. Scribner.
- Whoop Inc. (2020). *Heart rate variability and recovery: a methodology overview*. (Vendor whitepaper.)
