"""Digital Twin scoring layer. Same physics as v1, with a few tightened bounds
and an additional `sleep_debt` score that the API will surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SleepSummary:
    total_time_in_bed_min: float
    total_sleep_time_min: float
    wake_min: float
    nrem_min: float
    rem_min: float
    sleep_efficiency: float
    rem_fraction_of_sleep: float
    nrem_fraction_of_sleep: float
    awakenings: int


@dataclass(frozen=True)
class HumanState:
    recovery_score: float
    fatigue_score: float
    stress_index: float
    energy_level: float
    sleep_quality_score: float
    sleep_debt_score: float


def summarize_3class_sleep(predicted_stages: np.ndarray, epoch_seconds: float = 30.0) -> SleepSummary:
    stages = np.asarray(predicted_stages)
    epoch_min = epoch_seconds / 60.0
    total = float(len(stages) * epoch_min)
    wake = float(np.sum(stages == 0) * epoch_min)
    nrem = float(np.sum(stages == 1) * epoch_min)
    rem = float(np.sum(stages == 2) * epoch_min)
    sleep = nrem + rem
    eff = sleep / total if total else 0.0
    rem_frac = rem / sleep if sleep else 0.0
    nrem_frac = nrem / sleep if sleep else 0.0
    if stages.size == 0:
        awakenings = 0
    else:
        asleep = stages != 0
        sleep_started = np.maximum.accumulate(asleep)
        awakenings = int(np.sum((stages[1:] == 0) & (stages[:-1] != 0) & sleep_started[1:]))
    return SleepSummary(
        total_time_in_bed_min=total,
        total_sleep_time_min=sleep,
        wake_min=wake,
        nrem_min=nrem,
        rem_min=rem,
        sleep_efficiency=eff,
        rem_fraction_of_sleep=rem_frac,
        nrem_fraction_of_sleep=nrem_frac,
        awakenings=awakenings,
    )


def estimate_human_state(
    summary: SleepSummary,
    resting_hr_delta: float = 0.0,
    prior_day_steps: float = 0.0,
) -> HumanState:
    duration_score = np.clip(summary.total_sleep_time_min / 480.0, 0.0, 1.1)
    efficiency_score = np.clip(summary.sleep_efficiency / 0.90, 0.0, 1.1)
    rem_score = np.clip(summary.rem_fraction_of_sleep / 0.22, 0.0, 1.1)
    fragmentation_penalty = np.clip(summary.awakenings / 10.0, 0.0, 1.0)

    sleep_quality = 100.0 * (
        0.40 * duration_score
        + 0.30 * efficiency_score
        + 0.20 * rem_score
        + 0.10 * (1.0 - fragmentation_penalty)
    )
    sleep_quality = float(np.clip(sleep_quality, 0.0, 100.0))

    activity_load = np.clip(prior_day_steps / 14000.0, 0.0, 1.5)
    hr_stress = np.clip(resting_hr_delta / 15.0, -1.0, 2.0)

    recovery = float(np.clip(0.75 * sleep_quality + 15.0 * (1.0 - max(hr_stress, 0.0)), 0.0, 100.0))
    stress = float(
        np.clip(35.0 * max(hr_stress, 0.0) + 20.0 * activity_load + 20.0 * (1.0 - summary.sleep_efficiency), 0.0, 100.0)
    )
    fatigue = float(np.clip(100.0 - recovery + 25.0 * activity_load + 15.0 * max(hr_stress, 0.0), 0.0, 100.0))
    energy = float(np.clip(0.70 * recovery + 0.30 * (100.0 - fatigue), 0.0, 100.0))

    debt = float(np.clip((480.0 - summary.total_sleep_time_min) / 480.0 * 100.0, 0.0, 100.0))

    return HumanState(
        recovery_score=recovery,
        fatigue_score=fatigue,
        stress_index=stress,
        energy_level=energy,
        sleep_quality_score=sleep_quality,
        sleep_debt_score=debt,
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
    )
