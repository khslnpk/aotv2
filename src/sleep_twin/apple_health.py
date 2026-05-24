"""Parser for Apple Health exports.

Users export from the Health app (profile → Export All Health Data), which
produces an `export.zip` containing `apple_health_export/export.xml`. The XML
is a flat list of `<Record>` elements with `type` / `startDate` / `endDate` /
`value` attributes. Files can be multi-GB, so we stream with
`xml.etree.ElementTree.iterparse` and clear elements as we go.

Pipeline:
    1. Stream every `<Record>` in the export
    2. Retain only sleep-stage, heart-rate, and step records
    3. Group sleep records into chronologically-contiguous "nights"
    4. Per night, build a 30-second-epoch hypnogram (Wake / NREM / REM),
       attach the HR samples that fall inside it, and sum the preceding
       day's steps
    5. Return SleepSummary + PhysioSummary + HumanState (same shape as the
       PSG-subject analyses)
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO, Any, Iterable
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
# into Core/Deep/REM/Unspecified.
_SLEEP_STAGE_MAP = {
    "HKCategoryValueSleepAnalysisInBed": 0,          # treat as Wake (in bed but not asleep)
    "HKCategoryValueSleepAnalysisAwake": 0,
    "HKCategoryValueSleepAnalysisAsleep": 1,          # legacy single-stage sleep
    "HKCategoryValueSleepAnalysisAsleepUnspecified": 1,
    "HKCategoryValueSleepAnalysisAsleepCore": 1,
    "HKCategoryValueSleepAnalysisAsleepDeep": 1,
    "HKCategoryValueSleepAnalysisAsleepREM": 2,
}

_SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
_HR_TYPE = "HKQuantityTypeIdentifierHeartRate"
_STEPS_TYPE = "HKQuantityTypeIdentifierStepCount"
_RHR_TYPE = "HKQuantityTypeIdentifierRestingHeartRate"


@dataclass
class SleepRecord:
    start: datetime
    end: datetime
    stage: int            # 0/1/2
    source: str


@dataclass
class HRSample:
    t: datetime
    bpm: float
    source: str


@dataclass
class StepSample:
    start: datetime
    end: datetime
    steps: float


def _parse_apple_date(raw: str) -> datetime:
    # Apple format: "2026-05-22 23:47:12 -0700"
    # Convert "-0700" to "-07:00" so fromisoformat is happy
    fixed = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", raw.replace(" ", "T", 1).replace(" ", "T", 1))
    try:
        return datetime.fromisoformat(fixed)
    except ValueError:
        # Last-resort fallback
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S %z")


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
    # File-like object — peek at the magic bytes
    head = file_obj.read(4)
    file_obj.seek(0)
    if head[:2] == b"PK":
        # Buffer to BytesIO so ZipFile can seek
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

    Only records within the last `days_back` days are retained — this keeps the
    in-memory footprint bounded even for multi-year exports.
    """
    stream, archive = _open_xml(file_obj)

    cutoff_anchor: datetime | None = None
    sleep_records: list[SleepRecord] = []
    hr_samples: list[HRSample] = []
    step_samples: list[StepSample] = []
    resting_hr: list[HRSample] = []

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
                            source=elem.get("sourceName", ""),
                        )
                    )
                    if cutoff_anchor is None or end > cutoff_anchor:
                        cutoff_anchor = end
                elif rtype == _HR_TYPE:
                    t = _parse_apple_date(elem.get("startDate"))
                    try:
                        bpm = float(elem.get("value"))
                    except (TypeError, ValueError):
                        elem.clear()
                        continue
                    hr_samples.append(HRSample(t=t, bpm=bpm, source=elem.get("sourceName", "")))
                elif rtype == _RHR_TYPE:
                    t = _parse_apple_date(elem.get("startDate"))
                    try:
                        bpm = float(elem.get("value"))
                    except (TypeError, ValueError):
                        elem.clear()
                        continue
                    resting_hr.append(HRSample(t=t, bpm=bpm, source=elem.get("sourceName", "")))
                elif rtype == _STEPS_TYPE:
                    start = _parse_apple_date(elem.get("startDate"))
                    end = _parse_apple_date(elem.get("endDate"))
                    try:
                        steps = float(elem.get("value"))
                    except (TypeError, ValueError):
                        elem.clear()
                        continue
                    step_samples.append(StepSample(start=start, end=end, steps=steps))
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

    if cutoff_anchor is None and hr_samples:
        cutoff_anchor = max(s.t for s in hr_samples)
    if cutoff_anchor is None and step_samples:
        cutoff_anchor = max(s.end for s in step_samples)

    if cutoff_anchor is not None:
        cutoff = cutoff_anchor - timedelta(days=days_back)
        sleep_records = [r for r in sleep_records if r.end >= cutoff]
        hr_samples = [s for s in hr_samples if s.t >= cutoff]
        step_samples = [s for s in step_samples if s.end >= cutoff]
        resting_hr = [s for s in resting_hr if s.t >= cutoff]

    return {
        "sleep": sleep_records,
        "hr": hr_samples,
        "steps": step_samples,
        "resting_hr": resting_hr,
    }


# ---------------------------------------------------------------------------
# Nights
# ---------------------------------------------------------------------------

@dataclass
class Night:
    night_id: str
    start: datetime
    end: datetime
    duration_hours: float
    source: str
    has_apple_stages: bool        # True if AsleepCore/AsleepDeep/AsleepREM present
    n_sleep_records: int


def _group_into_nights(
    sleep_records: list[SleepRecord],
    *,
    gap_threshold_min: float = 60.0,
    min_duration_min: float = 30.0,
) -> list[tuple[datetime, datetime, list[SleepRecord]]]:
    """Cluster sleep records into contiguous nights.

    A new night starts whenever there's a gap of > gap_threshold_min between
    the previous record's end and the next record's start.
    """
    if not sleep_records:
        return []
    # Prefer Apple Watch records when multiple sources overlap. Otherwise dedupe
    # by (start, source) to keep one record per source.
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
        has_apple = any(r.stage == 2 for r in recs) or any(r.stage == 1 and "Asleep" in r.source for r in recs)
        # has_apple_stages is true if any record was a fine-grained AsleepX value
        # (we detect this indirectly: REM only appears in fine-grained tagging)
        has_apple_stages = any(r.stage == 2 for r in recs)
        # Date-based night id (the date the night started, in local time of the record)
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
        # Most recent
        return max(grouped, key=lambda g: g[0])
    for g in grouped:
        if g[0].strftime("%Y-%m-%d") == night_id:
            return g
    raise ValueError(f"night {night_id!r} not found")


def _build_hypnogram(
    start: datetime, end: datetime, records: list[SleepRecord], epoch_seconds: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    """Render an Apple sleep session into a 30-second-epoch hypnogram.

    Returns (epoch_times_sec, stages). `epoch_times_sec` is anchored at 0 = start.
    For overlapping records, deeper stages win:
        REM > AsleepX > InBed/Awake
    Default value for gaps inside the session is Wake (0).
    """
    total_sec = (end - start).total_seconds()
    n = max(int(np.ceil(total_sec / epoch_seconds)), 1)
    stages = np.zeros(n, dtype=np.int64)  # default Wake
    epoch_starts = np.arange(n, dtype=np.float64) * epoch_seconds

    # Priority: REM (2) > NREM (1) > Wake (0)
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


def _hr_in_session(
    hr_samples: list[HRSample], start: datetime, end: datetime
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_sec_relative_to_start, bpm) for HR samples inside [start-1h, end+1h]."""
    lo = start - timedelta(hours=1)
    hi = end + timedelta(hours=1)
    selected = [s for s in hr_samples if lo <= s.t <= hi]
    if not selected:
        return np.zeros(0), np.zeros(0)
    selected.sort(key=lambda s: s.t)
    times = np.array([(s.t - start).total_seconds() for s in selected], dtype=np.float64)
    bpm = np.array([s.bpm for s in selected], dtype=np.float64)
    return times, bpm


def _prior_day_steps(step_samples: list[StepSample], start: datetime) -> float:
    """Sum step counts in the 24 hours preceding sleep onset."""
    lo = start - timedelta(hours=24)
    return float(sum(s.steps for s in step_samples if lo <= s.start <= start))


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


def analyze_night(parsed: dict[str, Any], night_id: str | None = None) -> dict[str, Any]:
    """Run the full Digital Twin analysis on one night of Apple Health data."""
    start, end, records = _select_night(parsed, night_id)
    epoch_times, predicted = _build_hypnogram(start, end, records)
    hr_times, hr_bpm = _hr_in_session(parsed["hr"], start, end)
    steps_total = _prior_day_steps(parsed["steps"], start)

    sleep = summarize_3class_sleep(predicted)
    physio = summarize_physio(predicted, epoch_times, hr_times, hr_bpm, None, None)
    state = estimate_human_state(sleep, physio, prior_day_steps=steps_total)

    hr_x, hr_y = _downsample_xy(hr_times, hr_bpm, max_points=600)

    return {
        "subject_id": f"apple-{start.strftime('%Y-%m-%d')}",
        "source": "apple_health",
        "night_start_iso": start.isoformat(),
        "night_end_iso": end.isoformat(),
        "split": "external",
        "n_epochs": int(predicted.size),
        "duration_hours": round(predicted.size * 0.5 / 60.0, 2),
        "model_agreement_with_psg": 0.0,    # no ground truth
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
        "steps_total": steps_total,
        "sleep_summary": asdict(sleep),
        "physio_summary": asdict(physio),
        "state": {
            "sleep_quality_score": float(state.sleep_quality_score),
            "sleep_debt_score": float(state.sleep_debt_score),
            "recovery_score": float(state.recovery_score),
            "stress_index": float(state.stress_index),
            "fatigue_score": float(state.fatigue_score),
            "energy_level": float(state.energy_level),
            "components": state.components,
        },
        "baseline_inputs": {"prior_day_steps": steps_total},
    }
