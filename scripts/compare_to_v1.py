"""Generate a v1-vs-v2 comparison report from the saved metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V2_ROOT.parent
V1_MODELS = REPO_ROOT / "artifacts" / "models"
V2_MODELS = V2_ROOT / "artifacts" / "models"
REPORT_DIR = V2_ROOT / "artifacts" / "reports"


def _load_v1_metrics() -> list[dict]:
    rows = []
    for json_path in sorted(V1_MODELS.glob("*/*_metrics.json")):
        with json_path.open("r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append(
            {
                "version": "v1",
                "family": json_path.parent.name,
                "model": json_path.stem.removesuffix("_metrics"),
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "cohen_kappa": m["cohen_kappa"],
            }
        )
    return rows


def _load_v2_ensemble() -> tuple[list[dict], dict]:
    ens_path = V2_MODELS / "ensemble" / "ensemble_metrics.json"
    if not ens_path.exists():
        return [], {}
    with ens_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = []
    for row in payload.get("rows", []):
        rows.append({"version": "v2", "family": "ensemble", **row})
    return rows, payload.get("stacked", {})


def _format_table(rows: list[dict]) -> str:
    header = "| Version | Family | Model | Acc | Bal Acc | Macro F1 | Wtd F1 | Kappa |"
    sep = "|---|---|---|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row['version']} | {row['family']} | {row['model']} | "
            f"{row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['weighted_f1']:.4f} | {row['cohen_kappa']:.4f} |"
        )
    return "\n".join(lines)


def _delta(v2_best: dict, v1_best: dict) -> str:
    deltas = {
        "accuracy": v2_best["accuracy"] - v1_best["accuracy"],
        "balanced_accuracy": v2_best["balanced_accuracy"] - v1_best["balanced_accuracy"],
        "macro_f1": v2_best["macro_f1"] - v1_best["macro_f1"],
        "cohen_kappa": v2_best["cohen_kappa"] - v1_best["cohen_kappa"],
    }
    return ", ".join(f"{k}: {'+' if v >= 0 else ''}{v * 100:.2f}pp" for k, v in deltas.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=REPORT_DIR / "v1_vs_v2_summary.md")
    args = ap.parse_args()

    v1_rows = _load_v1_metrics()
    v2_rows, v2_stacked = _load_v2_ensemble()
    all_rows = sorted(v1_rows + v2_rows, key=lambda r: r["macro_f1"], reverse=True)

    if not v2_rows:
        print("WARN: no v2 ensemble metrics yet; report will only show v1.")

    v1_best = max(v1_rows, key=lambda r: r["macro_f1"]) if v1_rows else None
    v2_best = max(v2_rows, key=lambda r: r["macro_f1"]) if v2_rows else None

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    text = ["# v1 vs v2 Sleep-Stage Model Comparison", ""]
    if v1_best and v2_best:
        text.append(f"**v1 leader**: `{v1_best['model']}` (macroF1={v1_best['macro_f1']:.4f}, kappa={v1_best['cohen_kappa']:.4f})")
        text.append(f"**v2 leader**: `{v2_best['model']}` (macroF1={v2_best['macro_f1']:.4f}, kappa={v2_best['cohen_kappa']:.4f})")
        text.append(f"**Delta**: {_delta(v2_best, v1_best)}")
        text.append("")
    text.append("## Full Leaderboard (sorted by macro F1)")
    text.append("")
    text.append(_format_table(all_rows))
    text.append("")
    args.output.write_text("\n".join(text), encoding="utf-8")
    print(f"wrote {args.output}")
    print("\n".join(text[: max(8, len(text))]))


if __name__ == "__main__":
    main()
