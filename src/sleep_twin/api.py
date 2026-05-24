"""FastAPI app exposing the Sleep Twin analysis service to the front-end.

Endpoints:
    GET  /api/health
    GET  /api/subjects                          -> list of PSG subjects
    GET  /api/subjects/{id}                     -> full analysis for a PSG subject
    POST /api/whatif                            -> re-score under a hypothetical
    GET  /api/leaderboard                       -> headline model leaderboard rows
    POST /api/upload                            -> ingest an Apple Health export.zip
    GET  /api/uploads/{uid}/nights              -> list parsed nights
    GET  /api/uploads/{uid}/nights/{night_id}   -> analysis for one night
    GET  /                                      -> static front-end

The PSG SleepTwinService is instantiated once at startup and held in module
state. Apple Health uploads are kept in an in-memory dict keyed by uuid — they
do not survive a process restart, which is fine for a demo.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sleep_twin import apple_health
from sleep_twin.inference import SleepTwinService


_APP_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = _APP_ROOT / "app"
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024   # 500 MB cap

app = FastAPI(title="Sleep Twin", version="0.2.0")
_service: SleepTwinService | None = None
_uploads: dict[str, dict[str, Any]] = {}  # upload_id -> {"parsed": ..., "nights": [...]}


def _get_service() -> SleepTwinService:
    global _service
    if _service is None:
        _service = SleepTwinService()
    return _service


@app.on_event("startup")
def _startup() -> None:
    _get_service()  # eager-load so first request is fast


# ---------------------------------------------------------------------------
# PSG subject endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, Any]:
    svc = _get_service()
    return {
        "status": "ok",
        "n_subjects": len(svc.list_subjects()),
        "sources": list(svc.probs.keys()),
        "n_uploads": len(_uploads),
    }


@app.get("/api/subjects")
def list_subjects() -> list[dict[str, Any]]:
    return _get_service().list_subjects()


@app.get("/api/subjects/{subject_id}")
def get_subject(subject_id: str) -> dict[str, Any]:
    try:
        return _get_service().analyze(subject_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


class WhatIfRequest(BaseModel):
    subject_id: str
    lost_minutes: float = Field(0.0, ge=0.0, le=480.0)
    extra_steps: float = Field(0.0, ge=0.0, le=30000.0)


@app.post("/api/whatif")
def whatif(req: WhatIfRequest) -> dict[str, Any]:
    # Route to either the PSG subject service or a previously-uploaded night
    if req.subject_id.startswith("upload:"):
        upload_id, night_id = _parse_upload_subject(req.subject_id)
        analysis = _analyze_upload_night(upload_id, night_id)
        return _whatif_from_analysis(analysis, req.lost_minutes, req.extra_steps)
    try:
        return _get_service().whatif(
            subject_id=req.subject_id,
            lost_minutes=req.lost_minutes,
            extra_steps=req.extra_steps,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/leaderboard")
def leaderboard() -> list[dict[str, Any]]:
    return _get_service().leaderboard()


# ---------------------------------------------------------------------------
# Apple Health upload endpoints
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload exceeds {_MAX_UPLOAD_BYTES} bytes")
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="empty upload")

    import io
    try:
        parsed = apple_health.parse_apple_export(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not parse export: {exc}")

    nights = apple_health.list_nights(parsed)
    if not nights:
        raise HTTPException(
            status_code=400,
            detail="no sleep sessions found in the export — does this watch record sleep?",
        )

    upload_id = uuid.uuid4().hex[:12]
    _uploads[upload_id] = {"parsed": parsed, "nights": nights}

    default_night = nights[0]
    default_analysis = apple_health.analyze_night(parsed, default_night.night_id)
    default_analysis["upload_id"] = upload_id
    default_analysis["subject_id"] = f"upload:{upload_id}:{default_night.night_id}"

    return {
        "upload_id": upload_id,
        "nights": [_night_dict(n) for n in nights],
        "selected_night_id": default_night.night_id,
        "analysis": default_analysis,
    }


@app.get("/api/uploads/{upload_id}/nights")
def upload_nights(upload_id: str) -> list[dict[str, Any]]:
    if upload_id not in _uploads:
        raise HTTPException(status_code=404, detail="unknown upload_id")
    return [_night_dict(n) for n in _uploads[upload_id]["nights"]]


@app.get("/api/uploads/{upload_id}/nights/{night_id}")
def upload_night_analysis(upload_id: str, night_id: str) -> dict[str, Any]:
    if upload_id not in _uploads:
        raise HTTPException(status_code=404, detail="unknown upload_id")
    try:
        analysis = apple_health.analyze_night(_uploads[upload_id]["parsed"], night_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    analysis["upload_id"] = upload_id
    analysis["subject_id"] = f"upload:{upload_id}:{night_id}"
    return analysis


def _night_dict(n: apple_health.Night) -> dict[str, Any]:
    return {
        "id": n.night_id,
        "start": n.start.isoformat(),
        "end": n.end.isoformat(),
        "duration_hours": n.duration_hours,
        "source": n.source,
        "has_apple_stages": n.has_apple_stages,
        "n_sleep_records": n.n_sleep_records,
    }


def _parse_upload_subject(subject_id: str) -> tuple[str, str]:
    # Format: upload:<upload_id>:<night_id>
    parts = subject_id.split(":", 2)
    if len(parts) != 3:
        raise HTTPException(status_code=400, detail="malformed upload subject_id")
    return parts[1], parts[2]


def _analyze_upload_night(upload_id: str, night_id: str) -> dict[str, Any]:
    if upload_id not in _uploads:
        raise HTTPException(status_code=404, detail="unknown upload_id")
    try:
        return apple_health.analyze_night(_uploads[upload_id]["parsed"], night_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


def _whatif_from_analysis(analysis: dict[str, Any], lost_minutes: float, extra_steps: float) -> dict[str, Any]:
    """Re-score an Apple Health night under a hypothetical perturbation."""
    from sleep_twin.digital_twin import (
        HumanState,
        PhysioSummary,
        SleepSummary,
        estimate_human_state,
        simulate_sleep_less,
        simulate_train_harder,
    )

    sleep = SleepSummary(**analysis["sleep_summary"])
    physio_dict = analysis["physio_summary"]
    physio = PhysioSummary(**physio_dict) if physio_dict else None
    baseline_steps = float(analysis["steps_total"])
    base = estimate_human_state(sleep, physio, prior_day_steps=baseline_steps)

    modified = base
    if lost_minutes > 0:
        modified = simulate_sleep_less(modified, lost_minutes)
    if extra_steps > 0:
        modified = simulate_train_harder(modified, extra_steps)

    def _hs(s: HumanState) -> dict[str, Any]:
        return {
            "sleep_quality_score": float(s.sleep_quality_score),
            "sleep_debt_score": float(s.sleep_debt_score),
            "recovery_score": float(s.recovery_score),
            "stress_index": float(s.stress_index),
            "fatigue_score": float(s.fatigue_score),
            "energy_level": float(s.energy_level),
            "components": s.components,
        }

    base_d = _hs(base)
    mod_d = _hs(modified)
    return {
        "subject_id": analysis["subject_id"],
        "baseline": base_d,
        "scenario": mod_d,
        "delta": {k: mod_d[k] - base_d[k] for k in (
            "sleep_quality_score", "sleep_debt_score", "recovery_score",
            "stress_index", "fatigue_score", "energy_level",
        )},
    }


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/styles.css")
    def styles() -> FileResponse:
        return FileResponse(_STATIC_DIR / "styles.css", media_type="text/css")

    @app.get("/app.js")
    def app_js() -> FileResponse:
        return FileResponse(_STATIC_DIR / "app.js", media_type="application/javascript")
else:
    @app.get("/")
    def root() -> JSONResponse:
        return JSONResponse({"detail": "Static frontend not found. Build app/index.html first."})
