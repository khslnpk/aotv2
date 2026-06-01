"""Digital Twin scoring layer.

Produces six interpretable human-state scores (sleep_quality, sleep_debt,
recovery, stress_index, fatigue, energy) from a predicted hypnogram plus
optional raw heart rate and accelerometer signals.

Every score is a weighted sum of named, individually-bounded components, and
the per-score breakdown is exposed in `HumanState.components` so a UI can show
exactly what each number is made of. The literature citations and target
ranges that justify each component live in
`artifacts/reports/digital_twin_methodology.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Target ranges (AASM / NSF / consumer-wearable methodology references)
# ---------------------------------------------------------------------------

_RECOMMENDED_TST_MIN = 480.0                # 8 hours, NSF adult midpoint
_OPTIMAL_TST_RANGE = (420.0, 540.0)         # 7-9 h (Hirshkowitz 2015)
_OPTIMAL_EFFICIENCY = 0.85                  # AASM/Buysse PSQI threshold
_OPTIMAL_REM_FRACTION = (0.20, 0.25)        # Carskadon 2005
_OPTIMAL_NREM_FRACTION = (0.55, 0.65)       # Carskadon 2005
_OPTIMAL_ONSET_LATENCY_MIN = (10.0, 20.0)   # short onset = sleep deprivation sign
_WASO_FULL_PENALTY_MIN = 90.0               # WASO ≥ 90 min = full penalty
_OPTIMAL_CYCLE_COUNT = (4.0, 6.0)           # 4-6 NREM->REM cycles per night
_OPTIMAL_HR_DIP = 0.10                      # 10% drop from waking baseline
_BAD_HR_DRIFT_BPM_PER_HR = 5.0              # +5 bpm/hr through the night = bad
_HRV_PROXY_GOOD = 8.0                       # bpm-units of succ-diff RMS (proxy)
_HRV_SDNN_GOOD_MS = 50.0                    # SDNN ≥ 50 ms = healthy adult overnight HRV
_STEPS_HIGH_LOAD = 14000.0                  # daily step count baseline
_RESTING_HR_DELTA_HIGH = 15.0               # bpm above baseline that = full stress


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SleepSummary:
    """Sleep architecture summary derived purely from the predicted hypnogram."""
    total_time_in_bed_min: float
    total_sleep_time_min: float
    wake_min: float
    nrem_min: float
    rem_min: float
    sleep_efficiency: float
    rem_fraction_of_sleep: float
    nrem_fraction_of_sleep: float
    sleep_onset_latency_min: float
    rem_latency_min: float
    waso_min: float
    awakenings: int
    n_cycles: int


@dataclass(frozen=True)
class PhysioSummary:
    """Autonomic and movement summary derived from raw HR and motion during sleep.

    `hrv_sdnn_ms` is the *real* RMS-of-successive-NN-intervals HRV in
    milliseconds (e.g. as reported by Apple Health). When present, the
    scoring layer prefers it over the bpm-domain `hr_succdiff_rms` proxy.
    Defaults to NaN so the existing PSG path (which doesn't have true SDNN)
    keeps using the proxy.
    """
    pre_sleep_hr_bpm: float
    avg_sleep_hr_bpm: float
    min_sleep_hr_bpm: float
    hr_dip_pct: float
    hr_drift_bpm_per_hour: float
    hr_succdiff_rms: float
    sleep_movement_index: float
    hrv_sdnn_ms: float = float("nan")


@dataclass(frozen=True)
class HumanState:
    recovery_score: float
    fatigue_score: float
    stress_index: float
    energy_level: float
    sleep_quality_score: float
    sleep_debt_score: float
    components: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _band_score(value: float, lo: float, hi: float, soft: float = 0.25) -> float:
    """1.0 inside [lo, hi], smoothly decays to 0 outside.

    `soft` controls how forgiving the slope is — gap is measured in units of
    `soft * lo` below the band and `soft * hi` above the band. A larger soft
    means a more gradual penalty.
    """
    if lo <= value <= hi:
        return 1.0
    if value < lo:
        denom = max(lo * soft, 1e-6)
        gap = (lo - value) / denom
    else:
        denom = max(hi * soft, 1e-6)
        gap = (value - hi) / denom
    return float(max(0.0, 1.0 - gap))


def _saturating(value: float, target: float) -> float:
    """0 at value=0, 1 at value>=target, linear in between."""
    return float(np.clip(value / max(target, 1e-6), 0.0, 1.0))


def _saturating_neg(value: float, threshold: float) -> float:
    """1.0 at value=0, decays to 0 at value=threshold. Useful for penalties."""
    return float(np.clip(1.0 - value / max(threshold, 1e-6), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Sleep architecture: hypnogram -> SleepSummary
# ---------------------------------------------------------------------------

def summarize_3class_sleep(
    predicted_stages: np.ndarray, epoch_seconds: float = 30.0
) -> SleepSummary:
    """Compute sleep architecture from a per-epoch class sequence.

    Stages: 0=Wake, 1=NREM, 2=REM. Epochs must be in chronological order.
    """
    stages = np.asarray(predicted_stages).reshape(-1)
    n = int(stages.size)
    epoch_min = epoch_seconds / 60.0

    total = float(n * epoch_min)
    wake = float(np.sum(stages == 0) * epoch_min)
    nrem = float(np.sum(stages == 1) * epoch_min)
    rem = float(np.sum(stages == 2) * epoch_min)
    sleep = nrem + rem

    eff = sleep / total if total > 0 else 0.0
    rem_frac = rem / sleep if sleep > 0 else 0.0
    nrem_frac = nrem / sleep if sleep > 0 else 0.0

    sleep_indices = np.where(stages != 0)[0]
    if sleep_indices.size > 0:
        first_sleep_idx = int(sleep_indices[0])
        onset_latency = float(first_sleep_idx * epoch_min)
    else:
        first_sleep_idx = n
        onset_latency = total

    rem_indices = np.where(stages == 2)[0]
    if rem_indices.size > 0 and first_sleep_idx < n:
        first_rem_idx = int(rem_indices[0])
        rem_latency = float(max(first_rem_idx - first_sleep_idx, 0) * epoch_min)
    else:
        rem_latency = float(sleep) if sleep > 0 else 0.0

    if first_sleep_idx < n:
        post_onset = stages[first_sleep_idx:]
        waso = float(np.sum(post_onset == 0) * epoch_min)
    else:
        waso = 0.0

    if n > 1 and first_sleep_idx < n:
        asleep = stages != 0
        sleep_started = np.maximum.accumulate(asleep)
        awakenings = int(np.sum((stages[1:] == 0) & (stages[:-1] != 0) & sleep_started[1:]))
        n_cycles = int(np.sum((stages[1:] == 2) & (stages[:-1] == 1)))
    else:
        awakenings = 0
        n_cycles = 0

    return SleepSummary(
        total_time_in_bed_min=total,
        total_sleep_time_min=sleep,
        wake_min=wake,
        nrem_min=nrem,
        rem_min=rem,
        sleep_efficiency=eff,
        rem_fraction_of_sleep=rem_frac,
        nrem_fraction_of_sleep=nrem_frac,
        sleep_onset_latency_min=onset_latency,
        rem_latency_min=rem_latency,
        waso_min=waso,
        awakenings=awakenings,
        n_cycles=n_cycles,
    )


# ---------------------------------------------------------------------------
# Physiology: raw HR + motion during sleep -> PhysioSummary
# ---------------------------------------------------------------------------

def _sleep_window_mask(
    sample_times: np.ndarray,
    epoch_times: np.ndarray,
    stages: np.ndarray,
    epoch_seconds: float,
) -> np.ndarray:
    """Mask of samples that fall inside any non-Wake epoch."""
    if sample_times.size == 0:
        return np.zeros(0, dtype=bool)
    starts = epoch_times[stages != 0]
    if starts.size == 0:
        return np.zeros(len(sample_times), dtype=bool)
    ends = starts + epoch_seconds
    order = np.argsort(starts)
    starts = starts[order]
    ends = ends[order]
    # For each sample find the rightmost epoch start <= sample time
    idx = np.searchsorted(starts, sample_times, side="right") - 1
    mask = np.zeros(len(sample_times), dtype=bool)
    valid = idx >= 0
    mask[valid] = sample_times[valid] < ends[idx[valid]]
    return mask


def summarize_physio(
    predicted_stages: np.ndarray,
    epoch_times: np.ndarray,
    hr_times: np.ndarray,
    hr_bpm: np.ndarray,
    motion_times: Optional[np.ndarray] = None,
    motion_enmo: Optional[np.ndarray] = None,
    epoch_seconds: float = 30.0,
) -> PhysioSummary:
    """Aggregate HR (and optionally motion ENMO) over the predicted sleep period.

    epoch_times: per-epoch start times in seconds, aligned with predicted_stages.
    hr_times / hr_bpm: HR sample timestamps and values (same units of time as epoch_times).
    motion_times / motion_enmo: optional accelerometer ENMO samples.
    """
    stages = np.asarray(predicted_stages).reshape(-1)
    epoch_times = np.asarray(epoch_times, dtype=np.float64).reshape(-1)
    hr_times = np.asarray(hr_times, dtype=np.float64).reshape(-1)
    hr_bpm = np.asarray(hr_bpm, dtype=np.float64).reshape(-1)

    finite_hr = np.isfinite(hr_bpm)
    if hr_bpm.size == 0 or not finite_hr.any():
        return PhysioSummary(
            pre_sleep_hr_bpm=float("nan"),
            avg_sleep_hr_bpm=float("nan"),
            min_sleep_hr_bpm=float("nan"),
            hr_dip_pct=0.0,
            hr_drift_bpm_per_hour=0.0,
            hr_succdiff_rms=0.0,
            sleep_movement_index=0.0,
        )

    sleep_idx = np.where(stages != 0)[0]
    if sleep_idx.size == 0:
        # No predicted sleep at all
        pre_hr = float(hr_bpm[finite_hr].mean())
        return PhysioSummary(
            pre_sleep_hr_bpm=pre_hr,
            avg_sleep_hr_bpm=float("nan"),
            min_sleep_hr_bpm=float("nan"),
            hr_dip_pct=0.0,
            hr_drift_bpm_per_hour=0.0,
            hr_succdiff_rms=0.0,
            sleep_movement_index=0.0,
        )

    first_sleep_time = float(epoch_times[sleep_idx[0]])

    # Pre-sleep HR: 10-minute window before first sleep epoch, fall back to any
    # awake samples, then to the 75th percentile of the recording.
    pre_window = (hr_times >= first_sleep_time - 600.0) & (hr_times < first_sleep_time) & finite_hr
    if pre_window.any():
        pre_sleep_hr = float(hr_bpm[pre_window].mean())
    else:
        awake_mask = (hr_times < first_sleep_time) & finite_hr
        if awake_mask.any():
            pre_sleep_hr = float(hr_bpm[awake_mask].mean())
        else:
            pre_sleep_hr = float(np.percentile(hr_bpm[finite_hr], 75))

    sleep_mask = _sleep_window_mask(hr_times, epoch_times, stages, epoch_seconds) & finite_hr
    sleep_hr = hr_bpm[sleep_mask]
    sleep_hr_t = hr_times[sleep_mask]

    if sleep_hr.size == 0:
        return PhysioSummary(
            pre_sleep_hr_bpm=pre_sleep_hr,
            avg_sleep_hr_bpm=float("nan"),
            min_sleep_hr_bpm=float("nan"),
            hr_dip_pct=0.0,
            hr_drift_bpm_per_hour=0.0,
            hr_succdiff_rms=0.0,
            sleep_movement_index=0.0,
        )

    avg_sleep_hr = float(sleep_hr.mean())
    min_sleep_hr = float(np.percentile(sleep_hr, 5))  # robust min via 5th pct

    hr_dip_pct = float(max(0.0, (pre_sleep_hr - min_sleep_hr) / pre_sleep_hr)) if pre_sleep_hr > 0 else 0.0

    # Linear drift across sleep period (bpm/hour)
    if sleep_hr_t.size >= 2 and sleep_hr_t[-1] > sleep_hr_t[0]:
        t = sleep_hr_t - sleep_hr_t[0]
        slope_per_sec = float(np.polyfit(t, sleep_hr, 1)[0])
        hr_drift_bpm_per_hour = slope_per_sec * 3600.0
    else:
        hr_drift_bpm_per_hour = 0.0

    # Successive-difference RMS as a (rough) HRV proxy. Not true RMSSD because
    # the dataset gives sampled HR, not beat-to-beat R-R intervals.
    if sleep_hr.size >= 2:
        diffs = np.diff(sleep_hr)
        hr_succdiff_rms = float(np.sqrt(np.mean(diffs ** 2)))
    else:
        hr_succdiff_rms = 0.0

    sleep_movement_index = 0.0
    if motion_times is not None and motion_enmo is not None:
        m_times = np.asarray(motion_times, dtype=np.float64).reshape(-1)
        m_enmo = np.asarray(motion_enmo, dtype=np.float64).reshape(-1)
        finite_m = np.isfinite(m_enmo)
        if m_times.size > 0 and finite_m.any():
            m_sleep_mask = _sleep_window_mask(m_times, epoch_times, stages, epoch_seconds) & finite_m
            if m_sleep_mask.any():
                sleep_movement_index = float(m_enmo[m_sleep_mask].mean())

    return PhysioSummary(
        pre_sleep_hr_bpm=pre_sleep_hr,
        avg_sleep_hr_bpm=avg_sleep_hr,
        min_sleep_hr_bpm=min_sleep_hr,
        hr_dip_pct=hr_dip_pct,
        hr_drift_bpm_per_hour=hr_drift_bpm_per_hour,
        hr_succdiff_rms=hr_succdiff_rms,
        sleep_movement_index=sleep_movement_index,
    )


# ---------------------------------------------------------------------------
# Per-score breakdown (each helper returns (total_pts, components_dict))
# ---------------------------------------------------------------------------

def _score_sleep_quality(sleep: SleepSummary) -> tuple[float, dict]:
    components = {
        "duration":          25.0 * _band_score(sleep.total_sleep_time_min, *_OPTIMAL_TST_RANGE),
        "efficiency":        20.0 * _saturating(sleep.sleep_efficiency, _OPTIMAL_EFFICIENCY),
        "rem_balance":       15.0 * _band_score(sleep.rem_fraction_of_sleep, *_OPTIMAL_REM_FRACTION, soft=0.5),
        "nrem_balance":      10.0 * _band_score(sleep.nrem_fraction_of_sleep, *_OPTIMAL_NREM_FRACTION, soft=0.5),
        "waso":              10.0 * _saturating_neg(sleep.waso_min, _WASO_FULL_PENALTY_MIN),
        "onset_latency":     10.0 * _band_score(sleep.sleep_onset_latency_min, *_OPTIMAL_ONSET_LATENCY_MIN, soft=0.8),
        "cycle_count":       10.0 * _band_score(float(sleep.n_cycles), *_OPTIMAL_CYCLE_COUNT, soft=0.5),
    }
    total = float(np.clip(sum(components.values()), 0.0, 100.0))
    return total, components


def _score_sleep_debt(sleep: SleepSummary) -> tuple[float, dict]:
    deficit_min = max(0.0, _RECOMMENDED_TST_MIN - sleep.total_sleep_time_min)
    debt = float(np.clip(deficit_min / _RECOMMENDED_TST_MIN * 100.0, 0.0, 100.0))
    components = {
        "single_night_deficit_min": float(deficit_min),
        "_note": "single-night only; true sleep debt accumulates across nights",
    }
    return debt, components


def _score_recovery(
    sleep_quality: float,
    physio: Optional[PhysioSummary],
    resting_hr_delta: float,
) -> tuple[float, dict]:
    sleep_pts = 40.0 * (sleep_quality / 100.0)

    if physio is not None and np.isfinite(physio.hr_dip_pct):
        dip_pts = 25.0 * _saturating(physio.hr_dip_pct, _OPTIMAL_HR_DIP)
        # Prefer real Apple-reported SDNN (ms) when available; otherwise fall
        # back to the bpm-domain succdiff RMS proxy from raw HR samples.
        if np.isfinite(physio.hrv_sdnn_ms):
            hrv_pts = 20.0 * _saturating(physio.hrv_sdnn_ms, _HRV_SDNN_GOOD_MS)
            hrv_label = "hrv_sdnn"
        else:
            hrv_pts = 20.0 * _saturating(physio.hr_succdiff_rms, _HRV_PROXY_GOOD)
            hrv_label = "hrv_proxy"
        drift_pts = 15.0 * _saturating_neg(max(physio.hr_drift_bpm_per_hour, 0.0), _BAD_HR_DRIFT_BPM_PER_HR)
        components = {
            "sleep_contribution": sleep_pts,
            "hr_dip": dip_pts,
            hrv_label: hrv_pts,
            "hr_stability": drift_pts,
        }
    else:
        hr_stress = float(np.clip(resting_hr_delta / _RESTING_HR_DELTA_HIGH, 0.0, 2.0))
        autonomic_pts = 60.0 * (1.0 - min(hr_stress / 2.0, 1.0))
        components = {
            "sleep_contribution": sleep_pts,
            "autonomic_estimate": autonomic_pts,
            "_note": "hr_dip/hrv/drift unavailable — used resting_hr_delta fallback",
        }

    numeric = [v for v in components.values() if isinstance(v, (int, float))]
    total = float(np.clip(sum(numeric), 0.0, 100.0))
    return total, components


def _score_stress(
    sleep: SleepSummary,
    physio: Optional[PhysioSummary],
    resting_hr_delta: float,
    prior_day_steps: float,
) -> tuple[float, dict]:
    if physio is not None and np.isfinite(physio.avg_sleep_hr_bpm):
        autonomic_arousal = 30.0 * (1.0 - _saturating(physio.hr_dip_pct, _OPTIMAL_HR_DIP))
        if np.isfinite(physio.hrv_sdnn_ms):
            low_hrv = 25.0 * (1.0 - _saturating(physio.hrv_sdnn_ms, _HRV_SDNN_GOOD_MS))
            low_hrv_label = "low_hrv_sdnn"
        else:
            low_hrv = 25.0 * (1.0 - _saturating(physio.hr_succdiff_rms, _HRV_PROXY_GOOD))
            low_hrv_label = "low_hrv_proxy"
        overnight_climb = 15.0 * _saturating(max(physio.hr_drift_bpm_per_hour, 0.0), _BAD_HR_DRIFT_BPM_PER_HR)
        components = {
            "autonomic_arousal": autonomic_arousal,
            low_hrv_label: low_hrv,
            "overnight_hr_climb": overnight_climb,
        }
    else:
        hr_stress = float(np.clip(resting_hr_delta / _RESTING_HR_DELTA_HIGH, 0.0, 2.0))
        components = {"hr_elevation_estimate": 70.0 * (hr_stress / 2.0)}

    activity_load = float(np.clip(prior_day_steps / _STEPS_HIGH_LOAD, 0.0, 1.5))
    components["activity_load"] = 15.0 * (activity_load / 1.5)
    components["sleep_inefficiency"] = 15.0 * max(0.0, 1.0 - sleep.sleep_efficiency / _OPTIMAL_EFFICIENCY)

    numeric = [v for v in components.values() if isinstance(v, (int, float))]
    total = float(np.clip(sum(numeric), 0.0, 100.0))
    return total, components


def _score_fatigue(
    recovery: float,
    sleep: SleepSummary,
    physio: Optional[PhysioSummary],
    prior_day_steps: float,
) -> tuple[float, dict]:
    inverse_recovery = 40.0 * (1.0 - recovery / 100.0)

    activity_load = float(np.clip(prior_day_steps / _STEPS_HIGH_LOAD, 0.0, 1.5))
    activity_pts = 20.0 * (activity_load / 1.5)

    deficit_min = max(0.0, _RECOMMENDED_TST_MIN - sleep.total_sleep_time_min)
    sleep_deficit = 20.0 * float(np.clip(deficit_min / 180.0, 0.0, 1.0))  # 3h deficit = full

    fragmentation = 10.0 * float(np.clip(sleep.awakenings / 10.0, 0.0, 1.0))

    if physio is not None and np.isfinite(physio.hr_drift_bpm_per_hour):
        overnight_strain = 10.0 * _saturating(max(physio.hr_drift_bpm_per_hour, 0.0), _BAD_HR_DRIFT_BPM_PER_HR)
    else:
        overnight_strain = 5.0  # neutral midpoint when unknown

    components = {
        "low_recovery": inverse_recovery,
        "activity_load": activity_pts,
        "sleep_deficit": sleep_deficit,
        "fragmentation": fragmentation,
        "overnight_strain": overnight_strain,
    }
    total = float(np.clip(sum(components.values()), 0.0, 100.0))
    return total, components


def _score_energy(recovery: float, fatigue: float, sleep_quality: float) -> tuple[float, dict]:
    components = {
        "recovery_contribution": 0.50 * recovery,
        "fatigue_inverse":       0.30 * (100.0 - fatigue),
        "sleep_quality_contribution": 0.20 * sleep_quality,
    }
    total = float(np.clip(sum(components.values()), 0.0, 100.0))
    return total, components


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_human_state(
    sleep: SleepSummary,
    physio: Optional[PhysioSummary] = None,
    prior_day_steps: float = 0.0,
    resting_hr_delta: float = 0.0,
) -> HumanState:
    """Combine sleep architecture and (optional) physiology into six scores.

    `prior_day_steps`: total step count for the preceding day. Used for activity
        load in stress and fatigue.
    `resting_hr_delta`: bpm above the subject's personal baseline. Only consulted
        when no PhysioSummary is supplied.
    """
    sq, sq_c = _score_sleep_quality(sleep)
    debt, debt_c = _score_sleep_debt(sleep)
    rec, rec_c = _score_recovery(sq, physio, resting_hr_delta)
    stress, stress_c = _score_stress(sleep, physio, resting_hr_delta, prior_day_steps)
    fat, fat_c = _score_fatigue(rec, sleep, physio, prior_day_steps)
    energy, energy_c = _score_energy(rec, fat, sq)

    return HumanState(
        recovery_score=rec,
        fatigue_score=fat,
        stress_index=stress,
        energy_level=energy,
        sleep_quality_score=sq,
        sleep_debt_score=debt,
        components={
            "sleep_quality": sq_c,
            "sleep_debt": debt_c,
            "recovery": rec_c,
            "stress": stress_c,
            "fatigue": fat_c,
            "energy": energy_c,
        },
    )


def simulate_sleep_less(state: HumanState, lost_minutes: float) -> HumanState:
    p = float(np.clip(lost_minutes / 120.0, 0.0, 2.0))
    return HumanState(
        recovery_score=float(np.clip(state.recovery_score - 18.0 * p, 0.0, 100.0)),
        fatigue_score=float(np.clip(state.fatigue_score + 20.0 * p, 0.0, 100.0)),
        stress_index=float(np.clip(state.stress_index + 10.0 * p, 0.0, 100.0)),
        energy_level=float(np.clip(state.energy_level - 16.0 * p, 0.0, 100.0)),
        sleep_quality_score=float(np.clip(state.sleep_quality_score - 15.0 * p, 0.0, 100.0)),
        sleep_debt_score=float(np.clip(state.sleep_debt_score + 12.0 * p, 0.0, 100.0)),
        components=state.components,
    )


def simulate_train_harder(state: HumanState, extra_steps: float) -> HumanState:
    load = float(np.clip(extra_steps / 5000.0, 0.0, 3.0))
    return HumanState(
        recovery_score=float(np.clip(state.recovery_score - 8.0 * load, 0.0, 100.0)),
        fatigue_score=float(np.clip(state.fatigue_score + 12.0 * load, 0.0, 100.0)),
        stress_index=float(np.clip(state.stress_index + 7.0 * load, 0.0, 100.0)),
        energy_level=float(np.clip(state.energy_level - 6.0 * load, 0.0, 100.0)),
        sleep_quality_score=state.sleep_quality_score,
        sleep_debt_score=state.sleep_debt_score,
        components=state.components,
    )
