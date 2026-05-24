"""Launch the Sleep Twin web app.

Usage:
    .venv/Scripts/python.exe scripts/run_app.py
    .venv/Scripts/python.exe scripts/run_app.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Launch the Sleep Twin web app.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--reload", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print(f"\n  Sleep Twin -> http://{args.host}:{args.port}\n")
    uvicorn.run(
        "sleep_twin.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
