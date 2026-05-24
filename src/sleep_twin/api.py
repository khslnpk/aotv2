"""FastAPI app exposing the Sleep Twin analysis service to the front-end.

Endpoints:
    GET  /api/health
    GET  /api/subjects                 -> list of subjects (id, n_epochs, split)
    GET  /api/subjects/{id}            -> full analysis: hypnogram + scores + signals
    POST /api/whatif                   -> re-score under a hypothetical scenario
    GET  /api/leaderboard              -> headline model leaderboard rows
    GET  /                             -> static front-end (app/index.html)

The service is instantiated once at startup and held in module state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sleep_twin.inference import SleepTwinService


_APP_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = _APP_ROOT / "app"

app = FastAPI(title="Sleep Twin", version="0.2.0")
_service: SleepTwinService | None = None


def _get_service() -> SleepTwinService:
    global _service
    if _service is None:
        _service = SleepTwinService()
    return _service


@app.on_event("startup")
def _startup() -> None:
    _get_service()  # eager-load so first request is fast


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, Any]:
    svc = _get_service()
    return {
        "status": "ok",
        "n_subjects": len(svc.list_subjects()),
        "sources": list(svc.probs.keys()),
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
