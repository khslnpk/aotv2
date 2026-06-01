"""Parser for Apple Health exports.

Users export from the Health app (profile → Export All Health Data), which
produces an `export.zip` containing `apple_health_export/export.xml`. Files
can be multi-GB, so we stream with `xml.etree.ElementTree.iterparse` and
clear elements as we go.

Pipeline:
    1. Stream every `<Record>` in the export
    2. Retain sleep-stage records and the six quantity types the Digital Twin
       can consume: heart rate, HRV SDNN, resting HR, step count, active
       energy, exercise time
    3. Group sleep records into chronologically-contiguous "nights"
    4. Per night, build a 30-second-epoch hypnogram (Wake / NREM / REM) with
       optional brief-wake smoothing, attach HR samples that fall inside the
       session window, compute a per-session wearable summary (HRV SDNN,
       mean sleep HR, resting HR delta vs a multi-day baseline, blended
       activity load), and compute a 7-day history context
    5. Return SleepSummary + PhysioSummary (with real Apple HRV SDNN when
       available) + HumanState — same schema as PSG-subject analyses
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO, Any, Optional
from xml.etree import ElementTree as ET

import numpy as np

from sleep_twin.digital_twin import (
    estimate_human_state,
    summarize_3class_sleep,
    summarize_physio,
)
from sleep_twin.labels import LABEL_NAMES_3CLASS


# Apple's category values mapped into our 3-class scheme.
# Pre-iOS 16 only emitted "Asleep" / "InBed" / "Awake"; iOS 16+ splits Asleep
# into Core / Deep / REM / Unspecified.
_SLEEP_STAGE_MAP = {
    "HKCategoryValueSleepAnalysisInBed": 0,           # in bed but not asleep
    "HKCategoryValueSleepAnalysisAwake": 0,
    "HKCategoryValueSleepAnalysisAsleep": 1,          # legacy single-stage sleep
    "HKCategoryValueSleepAnalysisAsleepUnspecified": 1,
    "HKCategoryValueSleepAnalysisAsleepCore": 1,
    "HKCategoryValueSleepAnalysisAsleepDeep": 1,
    "HKCategoryValueSleepAnalysisAsleepREM": 2,
}

# Apple stage values that signal a fine-grained (Apple Watch watchOS 9+)
# tagging session, as opposed to old InBed-only / single-stage Asleep records.
_DETAILED_STAGE_VALUES = frozenset(
    {
        "HKCategoryValueSleepAnalysisAsleepCore",
        "HKCategoryValueSleepAnalysisAsleepDeep",
        "HKCategoryValueSleepAnalysisAsleepREM",
        "HKCategoryValueSleepAnalysisAsleepUnspecified",
        "HKCategoryValueSleepAnalysisAwake",
    }
)

_SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
_QUANTITY_TYPES = {
    "HKQuantityTypeIdentifierHeartRate":                   "heart_rate",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":    "hrv_sdnn",
    "HKQuantityTypeIdentifierRestingHeartRate":            "resting_hr",
    "HKQuantityTypeIdentifierStepCount":                   "steps",
    "HKQuantityTypeIdentifierActiveEnergyBurned":          "active_energy",
    "HKQuantityTypeIdentifierAppleExerciseTime":           "exercise_time",
}


# ---------------------------------------------------------------------------
# Record dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SleepRecord:
    start: datetime
    end: datetime
    stage: int                # 0/1/2 (Wake/NREM/REM)
    raw_value: str            # original Apple category string, kept so we can
                              # tell detailed-stage records from InBed-only
    source: str


@dataclass
class TimedSample:
    start: datetime
    end: datetime
    value: float
    source: str


# ---------------------------------------------------------------------------
# Date parser
# ---------------------------------------------------------------------------

def _parse_apple_date(raw: str) -> datetime:
    # Apple format: "2026-05-22 23:47:12 -0700"
    # Convert "-0700" to "-07:00" so fromisoformat is happy
    if not raw:
        raise ValueError("empty date string")
    fixed = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", raw.replace(" ", "T", 1).replace(" ", "T", 1))
    try:
        return datetime.fromisoformat(fixed)
    except ValueError:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S %z")


# ---------------------------------------------------------------------------
# Streaming parser
# ---------------------------------------------------------------------------

def _open_xml(file_obj: IO[bytes] | Path) -> tuple[IO[bytes], zipfile.ZipFile | None]:
    """Return a binary XML stream from either a raw XML file or a zip upload."""
    if isinstance(file_obj, Path):
        if file_obj.suffix.lower() == ".zip":
            z = zipfile.ZipFile(file_obj, "r")
            inner = next((n for n in z.namelist() if n.endswith("export.xml")), None)
            if inner is None:
                z.close()
                raise ValueError("zip does not contain an export.xml")
            return z.open(inner), z
        return open(file_obj, "rb"), None
    head = file_obj.read(4)
    file_obj.seek(0)
    if head[:2] == b"PK":
        buf = io.BytesIO(file_obj.read())
        z = zipfile.ZipFile(buf, "r")
        inner = next((n for n in z.namelist() if n.endswith("export.xml")), None)
        if inner is None:
            z.close()
            raise ValueError("zip does not contain an export.xml")
        return z.open(inner), z
    return file_obj, None


def parse_apple_export(
    file_obj: IO[bytes] | Path,
    *,
    days_back: int = 30,
) -> dict[str, Any]:
    """Stream-parse an Apple Health export and return raw record collections.

    Only records within the last `days_back` days of the most recent sleep /
    HR / step record are retained — this keeps the in-memory footprint bounded
    even for multi-year exports.
    """
    stream, archive = _open_xml(file_obj)

    cutoff_anchor: datetime | None = None
    sleep_records: list[SleepRecord] = []
    quantity: dict[str, list[TimedSample]] = {name: [] for name in _QUANTITY_TYPES.values()}
    source_counts: dict[str, int] = {"sleep": 0, **{name: 0 for name in _QUANTITY_TYPES.values()}}

    try:
        ctx = ET.iterparse(stream, events=("end",))
        for _, elem in ctx:
            if elem.tag != "Record":
                continue
            rtype = elem.get("type")
            if rtype is None:
                elem.clear()
                continue

            try:
                if rtype == _SLEEP_TYPE:
                    val = elem.get("value", "")
                    if val not in _SLEEP_STAGE_MAP:
                        elem.clear()
                        continue
                    start = _parse_apple_date(elem.get("startDate"))
                    end = _parse_apple_date(elem.get("endDate"))
                    sleep_records.append(
                        SleepRecord(
                            start=start,
                            end=end,
                            stage=_SLEEP_STAGE_MAP[val],
                            raw_value=val,
                            source=elem.get("sourceName", ""),
                        )
                    )
                    source_counts["sleep"] += 1
                    if cutoff_anchor is None or end > cutoff_anchor:
                        cutoff_anchor = end
                elif rtype in _QUANTITY_TYPES:
                    key = _QUANTITY_TYPES[rtype]
                    raw_value = elem.get("value")
                    try:
                        v = float(raw_value) if raw_value is not None else None
                    except (TypeError, ValueError):
                        v = None
                    if v is None:
                        elem.clear()
                        continue
                    start = _parse_apple_date(elem.get("startDate"))
                    end_attr = elem.get("endDate") or elem.get("startDate")
                    end = _parse_apple_date(end_attr)
                    quantity[key].append(
                        TimedSample(
                            start=start,
                            end=end,
                            value=v,
                            source=elem.get("sourceName", ""),
                        )
                    )
                    source_counts[key] += 1
                    if cutoff_anchor is None or end > cutoff_anchor:
                        cutoff_anchor = end
            except Exception:
                pass
            finally:
                elem.clear()
    finally:
        try:
            stream.close()
        except Exception:
            pass
        if archive is not None:
            try:
                archive.close()
            except Exception:
                pass

    if cutoff_anchor is not None:
        cutoff = cutoff_anchor - timedelta(days=days_back)
        sleep_records = [r for r in sleep_records if r.end >= cutoff]
        for key in quantity:
            quantity[key] = [s for s in quantity[key] if s.end >= cutoff]

    return {
        "sleep": sleep_records,
        "heart_rate": quantity["heart_rate"],
        "hrv_sdnn": quantity["hrv_sdnn"],
        "resting_hr": quantity["resting_hr"],
        "steps": quantity["steps"],
        "active_energy": quantity["active_energy"],
        "exercise_time": quantity["exercise_time"],
        "source_counts": source_counts,
    }


# ---------------------------------------------------------------------------
# Night grouping
# ---------------------------------------------------------------------------

@dataclass
class Night:
    night_id: str
    start: datetime
    end: datetime
    duration_hours: float
    source: str
    has_apple_stages: bool        # True if any record uses fine-grained Core/Deep/REM/Unspecified/Awake tagging
    n_sleep_records: int


def _group_into_nights(
    sleep_records: list[SleepRecord],
    *,
    gap_threshold_min: float = 120.0,
    min_duration_min: float = 30.0,
) -> list[tuple[datetime, datetime, list[SleepRecord]]]:
    """Cluster sleep records into contiguous nights.

    A new night starts when there is a gap of > gap_threshold_min between the
    previous record's end and the next record's start. 120 min default matches
    the demo and is more forgiving than 60 min for fragmented exports.
    """
    if not sleep_records:
        return []
    records = sorted(sleep_records, key=lambda r: r.start)
    nights: list[tuple[datetime, datetime, list[SleepRecord]]] = []
    cur: list[SleepRecord] = [records[0]]
    cur_start = records[0].start
    cur_end = records[0].end
    for r in records[1:]:
        gap_min = (r.start - cur_end).total_seconds() / 60.0
        if gap_min > gap_threshold_min:
            duration_min = (cur_end - cur_start).total_seconds() / 60.0
            if duration_min >= min_duration_min:
                nights.append((cur_start, cur_end, cur))
            cur = [r]
            cur_start = r.start
            cur_end = r.end
        else:
            cur.append(r)
            cur_end = max(cur_end, r.end)
    duration_min = (cur_end - cur_start).total_seconds() / 60.0
    if duration_min >= min_duration_min:
        nights.append((cur_start, cur_end, cur))
    return nights


def list_nights(parsed: dict[str, Any]) -> list[Night]:
    grouped = _group_into_nights(parsed["sleep"])
    nights: list[Night] = []
    for start, end, recs in grouped:
        sources = {r.source for r in recs}
        has_apple_stages = any(r.raw_value in _DETAILED_STAGE_VALUES for r in recs)
        night_id = start.strftime("%Y-%m-%d")
        nights.append(
            Night(
                night_id=night_id,
                start=start,
                end=end,
                duration_hours=round((end - start).total_seconds() / 3600.0, 2),
                source=", ".join(sorted(sources)) or "unknown",
                has_apple_stages=has_apple_stages,
                n_sleep_records=len(recs),
            )
        )
    nights.sort(key=lambda n: n.start, reverse=True)
    return nights


def _select_night(parsed: dict[str, Any], night_id: str | None) -> tuple[datetime, datetime, list[SleepRecord]]:
    grouped = _group_into_nights(parsed["sleep"])
    if not grouped:
        raise ValueError("no sleep sessions found in the export")
    if night_id is None:
        return max(grouped, key=lambda g: g[0])
    for g in grouped:
        if g[0].strftime("%Y-%m-%d") == night_id:
            return g
    raise ValueError(f"night {night_id!r} not found")


# ---------------------------------------------------------------------------
# Hypnogram construction + brief-wake smoothing
# ---------------------------------------------------------------------------

def _build_hypnogram(
    start: datetime, end: datetime, records: list[SleepRecord], epoch_seconds: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    """Render an Apple sleep session into a 30-second-epoch hypnogram.

    Priority rule on overlapping records: REM (2) > NREM (1) > Wake (0).
    Default value for gaps inside the session is Wake (0).
    """
    total_sec = (end - start).total_seconds()
    n = max(int(np.ceil(total_sec / epoch_seconds)), 1)
    stages = np.zeros(n, dtype=np.int64)
    epoch_starts = np.arange(n, dtype=np.float64) * epoch_seconds
    priority = {2: 3, 1: 2, 0: 1}
    chosen_priority = np.zeros(n, dtype=np.int64)
    for r in records:
        s = (r.start - start).total_seconds()
        e = (r.end - start).total_seconds()
        lo = max(int(np.floor(s / epoch_seconds)), 0)
        hi = min(int(np.ceil(e / epoch_seconds)), n)
        if hi <= lo:
            continue
        p = priority.get(r.stage, 0)
        mask = chosen_priority[lo:hi] < p
        sub = stages[lo:hi]
        sub[mask] = r.stage
        stages[lo:hi] = sub
        cp = chosen_priority[lo:hi]
        cp[mask] = p
        chosen_priority[lo:hi] = cp
    return epoch_starts, stages


def smooth_brief_awake_epochs(
    stages: np.ndarray,
    epoch_seconds: float = 30.0,
    max_awake_minutes: float = 3.0,
) -> tuple[np.ndarray, int]:
    """Collapse short internal wake runs into surrounding sleep stages.

    Apple Watch sometimes emits brief Awake blips during light sleep that
    look like fragmentation but are likely scoring noise. We fold any
    internal Wake run shorter than max_awake_minutes into the surrounding
    stage. Leading and trailing wake periods are preserved.

    Returns (smoothed_stages, n_epochs_changed).
    """
    stages = np.asarray(stages, dtype=np.int64).copy()
    max_epochs = int(np.floor((max_awake_minutes * 60.0) / epoch_seconds + 1e-9))
    if max_epochs <= 0 or stages.size == 0:
        return stages, 0
    original = stages.copy()

    def _run_bounds(idx: int, direction: int) -> tuple[int, int]:
        stage = stages[idx]
        start, end = idx, idx + 1
        if direction < 0:
            while start - 1 >= 0 and stages[start - 1] == stage:
                start -= 1
        else:
            while end < len(stages) and stages[end] == stage:
                end += 1
        return start, end

    i = 0
    while i < len(stages):
        if stages[i] != 0:
            i += 1
            continue
        j = i + 1
        while j < len(stages) and stages[j] == 0:
            j += 1
        run_len = j - i
        left_stage = int(stages[i - 1]) if i > 0 else None
        right_stage = int(stages[j]) if j < len(stages) else None
        if (
            0 < run_len <= max_epochs
            and left_stage is not None and right_stage is not None
            and left_stage != 0 and right_stage != 0
        ):
            if left_stage == right_stage:
                fill = left_stage
            else:
                left_start, _ = _run_bounds(i - 1, -1)
                _, right_end = _run_bounds(j, +1)
                fill = left_stage if (i - left_start) >= (right_end - j) else right_stage
            stages[i:j] = fill
        i = j
    return stages, int((stages != original).sum())


# ---------------------------------------------------------------------------
# Wearable summary (per session, HRV + resting HR baseline + blended steps)
# ---------------------------------------------------------------------------

@dataclass
class AppleWearableSummary:
    prior_day_steps: float
    blended_activity_steps: float        # 0.65 * prior_day + 0.35 * 7d avg
    active_energy_kcal: float
    exercise_min: float
    mean_sleep_hr_bpm: float
    hrv_sdnn_ms: float                    # mean SDNN in 12h window around the session
    resting_hr_current: float             # nightly RHR closest to session
    resting_hr_baseline: float            # median RHR over the lookback window
    resting_hr_delta: float               # current − baseline (positive = elevated)
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class AppleHistorySummary:
    lookback_days: int
    avg_daily_steps: float
    avg_daily_active_energy_kcal: float
    avg_daily_exercise_min: float
    days_with_step_data: int
    days_with_active_energy_data: int
    days_with_exercise_data: int
    resting_hr_baseline_recent: float
    hrv_recent_mean_ms: float
    detailed_sleep_sessions: int
    avg_sleep_time_min: float
    avg_sleep_efficiency: float
    notes: tuple[str, ...] = field(default_factory=tuple)


def _samples_in_window(samples: list[TimedSample], start: datetime, end: datetime) -> list[TimedSample]:
    if not samples:
        return []
    return [s for s in samples if s.end > start and s.start < end]


def _sum_overlapping(samples: list[TimedSample], start: datetime, end: datetime) -> float:
    """Time-overlap-weighted sum (matches the demo's behavior)."""
    selected = _samples_in_window(samples, start, end)
    if not selected:
        return float("nan")
    total = 0.0
    for s in selected:
        overlap = (min(s.end, end) - max(s.start, start)).total_seconds()
        duration = max((s.end - s.start).total_seconds(), 1.0)
        total += s.value * max(overlap, 0.0) / duration
    return total


def _mean_overlapping(samples: list[TimedSample], start: datetime, end: datetime) -> float:
    selected = _samples_in_window(samples, start, end)
    if not selected:
        return float("nan")
    return float(np.mean([s.value for s in selected]))


def _summarize_wearables(
    parsed: dict[str, Any],
    session_start: datetime,
    session_end: datetime,
    baseline_days: int = 7,
) -> AppleWearableSummary:
    prior_day_start = session_start - timedelta(hours=24)
    context_start = session_start - timedelta(hours=12)
    context_end = session_end + timedelta(hours=12)
    baseline_start = session_start - timedelta(days=baseline_days)

    prior_day_steps = _sum_overlapping(parsed["steps"], prior_day_start, session_start)
    active_energy = _sum_overlapping(parsed["active_energy"], prior_day_start, session_start)
    exercise_time = _sum_overlapping(parsed["exercise_time"], prior_day_start, session_start)
    mean_sleep_hr = _mean_overlapping(parsed["heart_rate"], session_start, session_end)
    hrv_sdnn = _mean_overlapping(parsed["hrv_sdnn"], context_start, context_end)
    resting_hr_current = _mean_overlapping(parsed["resting_hr"], context_start, context_end)

    baseline_rows = [s for s in parsed["resting_hr"] if s.end <= session_start and s.end > baseline_start]
    resting_hr_baseline = float(np.median([s.value for s in baseline_rows])) if baseline_rows else float("nan")

    history_window_start = session_start - timedelta(days=baseline_days)
    history_steps_daily = []
    for k in range(baseline_days):
        ws = history_window_start + timedelta(days=k)
        we = ws + timedelta(days=1)
        if we > session_start:
            break
        history_steps_daily.append(_sum_overlapping(parsed["steps"], ws, we))
    daily_finite = [v for v in history_steps_daily if np.isfinite(v)]
    avg_daily_steps = float(np.mean(daily_finite)) if daily_finite else float("nan")

    notes: list[str] = []
    if not np.isfinite(prior_day_steps):
        prior_day_steps = 0.0
        notes.append("No step records found in the 24 h before the session; using 0 steps.")
    if not np.isfinite(active_energy):
        active_energy = 0.0
    if not np.isfinite(exercise_time):
        exercise_time = 0.0

    if np.isfinite(avg_daily_steps):
        blended = 0.65 * float(prior_day_steps) + 0.35 * avg_daily_steps
    else:
        blended = float(prior_day_steps)

    if not np.isfinite(resting_hr_baseline):
        if np.isfinite(resting_hr_current):
            resting_hr_baseline = resting_hr_current
            notes.append(
                f"No prior resting-HR value in the last {baseline_days} day(s); using the current-day value as baseline."
            )
        else:
            resting_hr_baseline = float("nan")
            notes.append(
                f"Resting-HR baseline is unavailable in the last {baseline_days} day(s)."
            )

    if np.isfinite(resting_hr_current) and np.isfinite(resting_hr_baseline):
        resting_hr_delta = float(resting_hr_current - resting_hr_baseline)
    else:
        resting_hr_delta = 0.0
        notes.append("Resting-HR delta could not be computed; defaulted to 0.")

    return AppleWearableSummary(
        prior_day_steps=float(prior_day_steps),
        blended_activity_steps=float(blended),
        active_energy_kcal=float(active_energy),
        exercise_min=float(exercise_time),
        mean_sleep_hr_bpm=float(mean_sleep_hr) if np.isfinite(mean_sleep_hr) else float("nan"),
        hrv_sdnn_ms=float(hrv_sdnn) if np.isfinite(hrv_sdnn) else float("nan"),
        resting_hr_current=float(resting_hr_current) if np.isfinite(resting_hr_current) else float("nan"),
        resting_hr_baseline=float(resting_hr_baseline) if np.isfinite(resting_hr_baseline) else float("nan"),
        resting_hr_delta=float(resting_hr_delta),
        notes=tuple(notes),
    )


def _summarize_history(
    parsed: dict[str, Any],
    session_start: datetime,
    lookback_days: int = 7,
) -> AppleHistorySummary:
    notes: list[str] = []
    lookback_start = session_start - timedelta(days=lookback_days)

    def _daily_sums(samples: list[TimedSample]) -> tuple[list[float], list[bool]]:
        totals: list[float] = []
        has_data: list[bool] = []
        for k in range(lookback_days):
            we = session_start - timedelta(days=k)
            ws = we - timedelta(days=1)
            v = _sum_overlapping(samples, ws, we)
            totals.append(v)
            has_data.append(any(s.end > ws and s.start < we for s in samples))
        return totals, has_data

    steps_daily, steps_has = _daily_sums(parsed["steps"])
    energy_daily, energy_has = _daily_sums(parsed["active_energy"])
    exercise_daily, exercise_has = _daily_sums(parsed["exercise_time"])

    recent_rhr = [s.value for s in parsed["resting_hr"] if s.end <= session_start and s.end > lookback_start]
    recent_hrv = [s.value for s in parsed["hrv_sdnn"] if s.start < session_start and s.end > lookback_start]

    detailed_summaries: list[Any] = []
    for ws, we, recs in _group_into_nights(parsed["sleep"]):
        if we >= session_start or we <= lookback_start:
            continue
        if not any(r.raw_value in _DETAILED_STAGE_VALUES for r in recs):
            continue
        _, stages = _build_hypnogram(ws, we, recs)
        detailed_summaries.append(summarize_3class_sleep(stages))

    def _mean_observed(values: list[float], mask: list[bool]) -> float:
        finite = [v for v, m in zip(values, mask) if m and np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    if not any(steps_has): notes.append(f"No step records in the {lookback_days}-day history window.")
    if not any(energy_has): notes.append(f"No active-energy records in the {lookback_days}-day history window.")
    if not any(exercise_has): notes.append(f"No exercise-time records in the {lookback_days}-day history window.")
    if not recent_rhr: notes.append(f"No prior resting-HR baseline in the {lookback_days}-day history window.")
    if not recent_hrv: notes.append(f"No HRV samples in the {lookback_days}-day history window.")
    if not detailed_summaries: notes.append(f"No prior detailed sleep sessions in the {lookback_days}-day history window.")

    avg_tst = float(np.mean([s.total_sleep_time_min for s in detailed_summaries])) if detailed_summaries else float("nan")
    avg_eff = float(np.mean([s.sleep_efficiency for s in detailed_summaries])) if detailed_summaries else float("nan")

    return AppleHistorySummary(
        lookback_days=int(lookback_days),
        avg_daily_steps=_mean_observed(steps_daily, steps_has),
        avg_daily_active_energy_kcal=_mean_observed(energy_daily, energy_has),
        avg_daily_exercise_min=_mean_observed(exercise_daily, exercise_has),
        days_with_step_data=int(sum(steps_has)),
        days_with_active_energy_data=int(sum(energy_has)),
        days_with_exercise_data=int(sum(exercise_has)),
        resting_hr_baseline_recent=float(np.median(recent_rhr)) if recent_rhr else float("nan"),
        hrv_recent_mean_ms=float(np.mean(recent_hrv)) if recent_hrv else float("nan"),
        detailed_sleep_sessions=len(detailed_summaries),
        avg_sleep_time_min=avg_tst,
        avg_sleep_efficiency=avg_eff,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# HR samples + downsampling
# ---------------------------------------------------------------------------

def _hr_in_session(
    hr_samples: list[TimedSample], start: datetime, end: datetime
) -> tuple[np.ndarray, np.ndarray]:
    lo = start - timedelta(hours=1)
    hi = end + timedelta(hours=1)
    selected = [s for s in hr_samples if lo <= s.start <= hi]
    if not selected:
        return np.zeros(0), np.zeros(0)
    selected.sort(key=lambda s: s.start)
    times = np.array([(s.start - start).total_seconds() for s in selected], dtype=np.float64)
    bpm = np.array([s.value for s in selected], dtype=np.float64)
    return times, bpm


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points or x.size == 0:
        return x, y
    bins = np.linspace(x[0], x[-1], max_points + 1)
    idx = np.clip(np.searchsorted(bins, x) - 1, 0, max_points - 1)
    sums = np.bincount(idx, weights=y, minlength=max_points)
    counts = np.bincount(idx, minlength=max_points)
    with np.errstate(invalid="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, means


# ---------------------------------------------------------------------------
# End-to-end night analysis
# ---------------------------------------------------------------------------

def analyze_night(
    parsed: dict[str, Any],
    night_id: str | None = None,
    *,
    lookback_days: int = 7,
    wake_blip_minutes: float = 3.0,
) -> dict[str, Any]:
    """Run the full Digital Twin analysis on one Apple Health night."""
    start, end, records = _select_night(parsed, night_id)
    epoch_times, raw_stages = _build_hypnogram(start, end, records)
    smoothed_stages, n_smoothed = smooth_brief_awake_epochs(raw_stages, max_awake_minutes=wake_blip_minutes)
    predicted = smoothed_stages
    hr_times, hr_bpm = _hr_in_session(parsed["heart_rate"], start, end)

    wearables = _summarize_wearables(parsed, start, end, baseline_days=lookback_days)
    history = _summarize_history(parsed, start, lookback_days=lookback_days)

    sleep = summarize_3class_sleep(predicted)
    physio = summarize_physio(predicted, epoch_times, hr_times, hr_bpm, None, None)

    # Override succdiff_rms proxy with real Apple SDNN when available — they
    # measure related-but-different things, so we expose both. The scoring
    # layer in digital_twin.py prefers SDNN when present.
    physio_with_sdnn = physio
    if np.isfinite(wearables.hrv_sdnn_ms):
        # Re-construct with the SDNN field set
        from dataclasses import replace
        physio_with_sdnn = replace(physio, hrv_sdnn_ms=float(wearables.hrv_sdnn_ms))

    state = estimate_human_state(
        sleep,
        physio_with_sdnn,
        prior_day_steps=float(wearables.blended_activity_steps),
        resting_hr_delta=float(wearables.resting_hr_delta),
    )

    hr_x, hr_y = _downsample_xy(hr_times, hr_bpm, max_points=600)

    return {
        "subject_id": f"apple-{start.strftime('%Y-%m-%d')}",
        "source": "apple_health",
        "night_start_iso": start.isoformat(),
        "night_end_iso": end.isoformat(),
        "split": "external",
        "n_epochs": int(predicted.size),
        "duration_hours": round(predicted.size * 0.5 / 60.0, 2),
        "model_agreement_with_psg": 0.0,
        "epoch_times_sec": epoch_times.tolist(),
        "predicted_stages": predicted.tolist(),
        "true_stages": [],
        "epoch_confidence": [1.0] * int(predicted.size),
        "ensemble_probabilities": [],
        "label_names": LABEL_NAMES_3CLASS,
        "hr_timeline": {
            "time_sec": [None if not np.isfinite(v) else float(v) for v in hr_x],
            "bpm": [None if not np.isfinite(v) else float(v) for v in hr_y],
        },
        "motion_timeline": {"time_sec": [], "enmo": []},
        "steps_total": float(wearables.blended_activity_steps),
        "sleep_summary": asdict(sleep),
        "physio_summary": asdict(physio_with_sdnn),
        "state": {
            "sleep_quality_score": float(state.sleep_quality_score),
            "sleep_debt_score": float(state.sleep_debt_score),
            "recovery_score": float(state.recovery_score),
            "stress_index": float(state.stress_index),
            "fatigue_score": float(state.fatigue_score),
            "energy_level": float(state.energy_level),
            "components": state.components,
        },
        "baseline_inputs": {
            "prior_day_steps": float(wearables.prior_day_steps),
            "blended_activity_steps": float(wearables.blended_activity_steps),
            "resting_hr_delta": float(wearables.resting_hr_delta),
        },
        # New: Apple-specific transparency payloads so the UI can audit them
        "apple_wearable_summary": asdict(wearables),
        "apple_history_summary": asdict(history),
        "apple_smoothing": {
            "wake_blip_minutes": float(wake_blip_minutes),
            "epochs_smoothed": int(n_smoothed),
        },
        "apple_source_counts": parsed.get("source_counts", {}),
    }
