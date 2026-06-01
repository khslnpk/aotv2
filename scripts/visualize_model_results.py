"""Render a model-comparison HTML report.

Loads per-model metrics from the trained-model directories under
`artifacts/models/`, then produces:

    - 6 single-topic analytical panels
        * Model skill fingerprint (heatmap of recall / precision per class)
        * Minority-stage frontier (Wake recall vs REM recall scatter)
        * Confusion fingerprint (off-diagonal error-pattern heatmap)
        * Metric constellation (PCA scatter of model behavior)
        * Rank ribbons (parallel coordinates of metric ranks)
        * Top-model radar (per-class F1 + balanced acc + kappa)
    - A normalized confusion-matrix grid for the 5 base learners
    - A dark-themed HTML report (matches the app's visual language) that
      embeds the panels, a summary table, the per-class F1 table, AND the
      full 16-row ensemble + HMM leaderboard from the stacking module

Usage:
    .venv/Scripts/python.exe scripts/visualize_model_results.py
    .venv/Scripts/python.exe scripts/visualize_model_results.py --output-dir somewhere/else
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Model registry (display names + family colors)
# ---------------------------------------------------------------------------

MODEL_NAMES = {
    "xgboost_cuda":     "XGBoost (CUDA)",
    "xgboost_cpu":      "XGBoost (CPU)",
    "lightgbm":         "LightGBM",
    "catboost":         "CatBoost",
    "catboost_tuned":   "CatBoost (Optuna-tuned)",
    "bilstm_attention": "BiLSTM + attention",
}

MODEL_SHORT_NAMES = {
    "XGBoost (CUDA)":           "XGB",
    "XGBoost (CPU)":            "XGB CPU",
    "LightGBM":                 "LGBM",
    "CatBoost":                 "CB",
    "CatBoost (Optuna-tuned)":  "CB tuned",
    "BiLSTM + attention":       "BiLSTM",
}

FAMILY_COLORS = {
    "Boosting":  "#8b5cf6",   # violet, matches the app's sleep accent
    "Sequence":  "#00d9c0",   # teal, matches the recovery accent
}

# Default file locations for the metrics JSONs (relative to repo root).
DEFAULT_METRIC_FILES: list[tuple[str, Path]] = [
    ("xgboost_cuda",     Path("artifacts/models/boosters/xgboost_cuda_metrics.json")),
    ("lightgbm",         Path("artifacts/models/boosters/lightgbm_metrics.json")),
    ("catboost",         Path("artifacts/models/boosters/catboost_metrics.json")),
    ("catboost_tuned",   Path("artifacts/models/catboost_tuned/catboost_tuned_metrics.json")),
    ("bilstm_attention", Path("artifacts/models/sequence/bilstm_attention_metrics.json")),
]

ENSEMBLE_METRICS_PATH = Path("artifacts/models/ensemble/ensemble_metrics.json")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadedMetrics:
    model_id: str
    model: str
    family: str
    metrics: dict


def _family_of(model_id: str) -> str:
    if model_id.startswith("bilstm") or model_id.startswith("tcnn") or model_id.startswith("mlp"):
        return "Sequence"
    return "Boosting"


def load_base_metrics(root: Path) -> list[LoadedMetrics]:
    loaded: list[LoadedMetrics] = []
    for model_id, rel in DEFAULT_METRIC_FILES:
        path = root / rel
        if not path.exists():
            print(f"  skipping {path} (missing)")
            continue
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        loaded.append(
            LoadedMetrics(
                model_id=model_id,
                model=MODEL_NAMES.get(model_id, model_id),
                family=_family_of(model_id),
                metrics=metrics,
            )
        )
    if not loaded:
        raise FileNotFoundError(
            "No base-learner metrics JSONs found. Run scripts/train_all.py first."
        )
    return loaded


def load_ensemble_rows(root: Path) -> list[dict]:
    path = root / ENSEMBLE_METRICS_PATH
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        bundle = json.load(f)
    return list(bundle.get("rows", []))


# ---------------------------------------------------------------------------
# Tabular reshaping
# ---------------------------------------------------------------------------

def _label_names(loaded: list[LoadedMetrics]) -> list[str]:
    report = loaded[0].metrics["classification_report"]
    labels = [name for name, value in report.items() if isinstance(value, dict)]
    labels = [name for name in labels if name not in {"macro avg", "weighted avg"}]
    preferred = [label for label in ["wake", "nrem", "rem"] if label in labels]
    return preferred or labels


def build_tables(loaded: list[LoadedMetrics]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = _label_names(loaded)
    summary_rows: list[dict] = []
    class_rows: list[dict] = []
    error_rows: list[dict] = []

    for item in loaded:
        report = item.metrics["classification_report"]
        row = {
            "model": item.model,
            "model_id": item.model_id,
            "family": item.family,
            "accuracy": item.metrics["accuracy"],
            "balanced_accuracy": item.metrics["balanced_accuracy"],
            "macro_f1": item.metrics["macro_f1"],
            "weighted_f1": item.metrics["weighted_f1"],
            "cohen_kappa": item.metrics["cohen_kappa"],
            "imbalance_gap": item.metrics["weighted_f1"] - item.metrics["macro_f1"],
        }
        recalls = []
        for label in labels:
            label_report = report[label]
            recalls.append(label_report["recall"])
            row[f"{label}_precision"] = label_report["precision"]
            row[f"{label}_recall"] = label_report["recall"]
            row[f"{label}_f1"] = label_report["f1-score"]
            class_rows.append(
                {
                    "model": item.model,
                    "family": item.family,
                    "label": label,
                    "precision": label_report["precision"],
                    "recall": label_report["recall"],
                    "f1": label_report["f1-score"],
                    "support": label_report["support"],
                }
            )

        row["minority_recall"] = float(np.mean([row["wake_recall"], row["rem_recall"]]))
        row["recall_range"] = float(np.max(recalls) - np.min(recalls))
        row["recall_balance"] = float(1.0 - row["recall_range"])
        summary_rows.append(row)

        cm = np.asarray(item.metrics["confusion_matrix"], dtype=float)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        for i, true_label in enumerate(labels):
            for j, predicted_label in enumerate(labels):
                error_rows.append(
                    {
                        "model": item.model,
                        "family": item.family,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "cell": f"{true_label}->{predicted_label}",
                        "count": int(cm[i, j]),
                        "row_rate": float(cm_norm[i, j]),
                        "is_error": i != j,
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    class_df = pd.DataFrame(class_rows)
    error_df = pd.DataFrame(error_rows)
    return summary_df, class_df, error_df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _annotate_points(ax: plt.Axes, df: pd.DataFrame, x: str, y: str) -> None:
    for idx, row in df.iterrows():
        dx = 0.006 if idx % 2 == 0 else -0.020
        dy = 0.008 if idx % 3 != 0 else -0.018
        ax.text(
            row[x] + dx, row[y] + dy,
            MODEL_SHORT_NAMES.get(row["model"], row["model"]),
            fontsize=9, weight="bold",
        )


def _plot_skill_heatmap(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    table = summary_df.set_index("model")[
        [
            "wake_recall", "wake_precision",
            "rem_recall",  "rem_precision",
            "nrem_recall",
            "minority_recall", "recall_balance",
            "macro_f1", "cohen_kappa",
        ]
    ].rename(columns={
        "wake_recall": "Wake recall",
        "wake_precision": "Wake precision",
        "rem_recall": "REM recall",
        "rem_precision": "REM precision",
        "nrem_recall": "NREM recall",
        "minority_recall": "Wake/REM recall",
        "recall_balance": "Recall balance",
        "macro_f1": "Macro F1",
        "cohen_kappa": "Kappa",
    })
    sns.heatmap(
        table, ax=ax, cmap="YlGnBu", vmin=0.30, vmax=0.95,
        annot=True, fmt=".2f", linewidths=0.7, cbar_kws={"label": "score"},
    )
    ax.set_title("Model skill fingerprint", loc="left", fontsize=13, weight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")


def _plot_minority_frontier(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    for family, group in summary_df.groupby("family"):
        ax.scatter(
            group["wake_recall"], group["rem_recall"],
            s=250 + 350 * group["nrem_recall"],
            c=FAMILY_COLORS.get(family, "#666666"),
            label=family, alpha=0.82, edgecolor="white", linewidth=1.4,
        )
    _annotate_points(ax, summary_df, "wake_recall", "rem_recall")
    lo, hi = 0.30, 0.85
    ax.plot([lo, hi], [lo, hi], color="#888888", linestyle="--", linewidth=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Wake recall"); ax.set_ylabel("REM recall")
    ax.set_title("Minority-stage frontier", loc="left", fontsize=13, weight="bold")
    ax.text(lo + 0.01, hi - 0.04, "bubble size = NREM recall", fontsize=9, color="#444444")
    ax.legend(frameon=False, loc="lower right")


def _plot_error_fingerprint(ax: plt.Axes, error_df: pd.DataFrame, labels: list[str]) -> None:
    off_diagonal = error_df[error_df["is_error"]].copy()
    off_diagonal["cell"] = pd.Categorical(
        off_diagonal["cell"],
        categories=[f"{true}->{pred}" for true in labels for pred in labels if true != pred],
        ordered=True,
    )
    pivot = off_diagonal.pivot(index="model", columns="cell", values="row_rate").fillna(0.0)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    sns.heatmap(
        pivot, ax=ax, cmap="rocket_r",
        vmin=0.0, vmax=max(0.35, float(pivot.to_numpy().max())),
        annot=True, fmt=".2f", linewidths=0.7,
        cbar_kws={"label": "row-normalized error rate"},
    )
    ax.set_title("Confusion fingerprint", loc="left", fontsize=13, weight="bold")
    ax.set_xlabel("error path"); ax.set_ylabel("")


def _metric_matrix(summary_df: pd.DataFrame, error_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa",
        "wake_precision", "wake_recall", "wake_f1",
        "nrem_precision", "nrem_recall", "nrem_f1",
        "rem_precision", "rem_recall", "rem_f1",
        "minority_recall", "recall_balance",
    ]
    matrix = summary_df.set_index("model")[metric_cols].copy()
    off_diagonal = error_df[error_df["is_error"]].pivot(index="model", columns="cell", values="row_rate").fillna(0.0)
    matrix = matrix.join(off_diagonal, how="left").fillna(0.0)
    return matrix


def _plot_metric_constellation(ax: plt.Axes, summary_df: pd.DataFrame, error_df: pd.DataFrame) -> None:
    matrix = _metric_matrix(summary_df, error_df)
    X = matrix.to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-8)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    coords = X @ vt[:2].T
    coords_df = summary_df.set_index("model").loc[matrix.index].copy()
    coords_df["pc1"] = coords[:, 0]
    coords_df["pc2"] = coords[:, 1]

    for family, group in coords_df.groupby("family"):
        ax.scatter(
            group["pc1"], group["pc2"],
            s=2500 * group["macro_f1"],
            c=FAMILY_COLORS.get(family, "#666666"),
            alpha=0.80, edgecolor="white", linewidth=1.4, label=family,
        )
    _annotate_points(ax, coords_df.reset_index(), "pc1", "pc2")
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.axvline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_xlabel("behavior component 1"); ax.set_ylabel("behavior component 2")
    ax.set_title("Metric constellation", loc="left", fontsize=13, weight="bold")
    ax.text(0.02, 0.96, "distance reflects whole behavior vector",
            transform=ax.transAxes, fontsize=9, color="#444444")
    ax.legend(frameon=False, loc="best")


def _plot_rank_ribbons(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "wake_recall", "rem_recall", "cohen_kappa"]
    names = ["Accuracy", "Balanced", "Macro F1", "Wake recall", "REM recall", "Kappa"]
    ranks = summary_df.set_index("model")[metrics].rank(ascending=False, method="min")
    x = np.arange(len(metrics))
    for model, row in ranks.iterrows():
        family = summary_df.loc[summary_df["model"] == model, "family"].iloc[0]
        ax.plot(x, row.to_numpy(), marker="o", linewidth=2.2,
                color=FAMILY_COLORS.get(family, "#666666"), alpha=0.78)
        ax.text(x[-1] + 0.08, row.iloc[-1],
                MODEL_SHORT_NAMES.get(model, model),
                va="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_yticks(range(1, len(summary_df) + 1))
    ax.set_ylim(len(summary_df) + 0.5, 0.5)
    ax.set_ylabel("rank")
    ax.set_title("Rank ribbons across competing goals", loc="left", fontsize=13, weight="bold")
    ax.grid(axis="y", color="#dddddd")


def _plot_radar(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    axes = ["wake_f1", "nrem_f1", "rem_f1", "balanced_accuracy", "cohen_kappa"]
    names = ["Wake F1", "NREM F1", "REM F1", "Balanced", "Kappa"]
    top = summary_df.sort_values("macro_f1", ascending=False).head(4)
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0.0, 0.90); ax.set_yticks([0.3, 0.6, 0.9])
    ax.set_yticklabels(["0.3", "0.6", "0.9"], fontsize=8)
    for _, row in top.iterrows():
        values = [row[col] for col in axes]; values += values[:1]
        color = FAMILY_COLORS.get(row["family"], "#666666")
        ax.plot(angles, values, color=color, linewidth=2.0,
                label=MODEL_SHORT_NAMES.get(row["model"], row["model"]))
        ax.fill(angles, values, color=color, alpha=0.10)
    ax.set_title("Top-model radar", loc="left", fontsize=13, weight="bold", pad=20)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2)


def plot_confusion_matrices(loaded: list[LoadedMetrics], labels: list[str], output_path: Path) -> None:
    sns.set_theme(style="white")
    cols = 3
    rows = int(np.ceil(len(loaded) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.6 * rows), squeeze=False)
    for plot_idx, (ax, item) in enumerate(zip(axes.ravel(), loaded)):
        cm = np.asarray(item.metrics["confusion_matrix"], dtype=float)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        annotations = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f"{cm_norm[i, j]:.0%}\n{int(cm[i, j])}"
        sns.heatmap(
            cm_norm, ax=ax, cmap="YlGnBu", vmin=0, vmax=1,
            annot=annotations, fmt="",
            xticklabels=[label.upper() for label in labels],
            yticklabels=[label.upper() for label in labels],
            cbar=False, linewidths=0.8, linecolor="white",
        )
        ax.set_title(item.model, fontsize=12, weight="bold")
        ax.set_xlabel("predicted" if plot_idx // cols == rows - 1 else "")
        ax.set_ylabel("actual")
    for ax in axes.ravel()[len(loaded):]:
        ax.axis("off")
    fig.suptitle("Normalized Confusion Matrices: Percent and Count",
                 fontsize=18, weight="bold", x=0.01, ha="left")
    fig.subplots_adjust(hspace=0.58, wspace=0.32, top=0.92, bottom=0.06)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_panel(output_path: Path, figsize: tuple[float, float], plotter, *args,
                projection: str | None = None) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection=projection)
    plotter(ax, *args)
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_individual_panels(summary_df: pd.DataFrame, error_df: pd.DataFrame,
                            labels: list[str], output_dir: Path) -> dict[str, Path]:
    return {
        "skill":         _save_panel(output_dir / "panel_skill_fingerprint.png",   (15.5, 8.0), _plot_skill_heatmap, summary_df),
        "frontier":      _save_panel(output_dir / "panel_minority_frontier.png",   (11.5, 8.0), _plot_minority_frontier, summary_df),
        "errors":        _save_panel(output_dir / "panel_confusion_fingerprint.png", (15.5, 8.0), _plot_error_fingerprint, error_df, labels),
        "constellation": _save_panel(output_dir / "panel_metric_constellation.png",(11.5, 8.0), _plot_metric_constellation, summary_df, error_df),
        "ranks":         _save_panel(output_dir / "panel_rank_ribbons.png",        (14.0, 7.5), _plot_rank_ribbons, summary_df),
        "radar":         _save_panel(output_dir / "panel_top_model_radar.png",     (9.5, 8.5), _plot_radar, summary_df, projection="polar"),
    }


# ---------------------------------------------------------------------------
# HTML report (dark theme, matches the app)
# ---------------------------------------------------------------------------

def _ensemble_rows_table(rows: list[dict]) -> str:
    if not rows:
        return "<p style='color:#888'>No ensemble metrics found at "\
               f"<code>{ENSEMBLE_METRICS_PATH}</code>.</p>"
    sorted_rows = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)
    head = "<tr><th>Rank</th><th>Model</th><th>Accuracy</th><th>Bal acc</th>"\
           "<th>Macro F1</th><th>Wtd F1</th><th>Kappa</th></tr>"
    body = []
    for i, r in enumerate(sorted_rows, 1):
        row_class = "ensemble-row-top" if i <= 2 else ""
        body.append(
            f"<tr class='{row_class}'>"
            f"<td>{i}</td>"
            f"<td><code>{r['model']}</code></td>"
            f"<td>{r['accuracy']:.4f}</td>"
            f"<td>{r['balanced_accuracy']:.4f}</td>"
            f"<td>{r['macro_f1']:.4f}</td>"
            f"<td>{r['weighted_f1']:.4f}</td>"
            f"<td>{r['cohen_kappa']:.4f}</td>"
            f"</tr>"
        )
    return f"<table class='leaderboard-table'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


def write_html_report(
    output_path: Path,
    confusion_path: Path,
    panel_paths: dict[str, Path],
    summary_df: pd.DataFrame,
    class_df: pd.DataFrame,
    ensemble_rows: list[dict],
) -> None:
    best_overall = summary_df.iloc[0]
    best_wake = summary_df.sort_values("wake_recall", ascending=False).iloc[0]
    best_rem = summary_df.sort_values("rem_recall", ascending=False).iloc[0]
    most_balanced = summary_df.sort_values("recall_balance", ascending=False).iloc[0]
    summary_table = summary_df[
        ["model", "family", "accuracy", "balanced_accuracy", "macro_f1",
         "cohen_kappa", "wake_recall", "rem_recall", "recall_balance"]
    ].copy()
    for col in summary_table.columns:
        if col not in {"model", "family"}:
            summary_table[col] = summary_table[col].map(lambda value: f"{value:.3f}")
    class_pivot = class_df.pivot(index="model", columns="label", values="f1").loc[summary_df["model"]]
    class_pivot = class_pivot.rename(columns={col: f"{col.upper()} F1" for col in class_pivot.columns})
    class_pivot = class_pivot.map(lambda value: f"{value:.3f}")

    ensemble_html = _ensemble_rows_table(ensemble_rows)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sleep Twin — Model Comparison Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-deep: #06070d;
      --bg-canvas: #0a0c1a;
      --bg-card: rgba(255, 255, 255, 0.04);
      --border: rgba(255, 255, 255, 0.10);
      --border-hi: rgba(255, 255, 255, 0.18);
      --text: #e8e9f3;
      --text-muted: #8c8fa6;
      --text-faint: #5a5d75;
      --c-recovery: #00d9c0;
      --c-sleep: #8b5cf6;
      --c-stress: #ec4899;
      --c-fatigue: #f59e0b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 0;
      background: var(--bg-deep);
      color: var(--text);
      font-family: "Inter", system-ui, sans-serif;
      font-size: 15px; line-height: 1.55;
      color-scheme: dark;
    }}
    body::before {{
      content: ""; position: fixed; inset: 0; z-index: -1;
      background:
        radial-gradient(ellipse at 20% 0%, rgba(139,92,246,0.08), transparent 50%),
        radial-gradient(ellipse at 80% 100%, rgba(0,217,192,0.06), transparent 50%),
        radial-gradient(ellipse at top, #0d1130 0%, #06070d 60%);
    }}
    .page {{ max-width: 1440px; margin: 0 auto; padding: 56px 32px; }}
    .nav {{
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 28px;
    }}
    .brand {{
      font-family: "Fraunces", serif;
      font-weight: 500; font-size: 22px;
      letter-spacing: -0.01em;
    }}
    .brand small {{
      display: inline-block; margin-left: 12px;
      font-family: "JetBrains Mono", monospace;
      font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase;
      color: var(--text-faint);
    }}
    h1 {{
      font-family: "Fraunces", serif;
      font-weight: 500;
      font-size: clamp(42px, 6vw, 64px);
      letter-spacing: -0.025em;
      line-height: 1.05;
      margin: 0 0 14px;
      background: linear-gradient(180deg, #ffffff 0%, var(--c-recovery) 100%);
      -webkit-background-clip: text; background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .lede {{
      max-width: 720px; color: var(--text-muted);
      font-family: "Fraunces", serif;
      font-weight: 400; font-size: 17px; line-height: 1.65;
      margin: 0 0 40px;
    }}
    h2 {{
      font-family: "Fraunces", serif;
      font-weight: 500; font-size: 26px;
      letter-spacing: -0.012em;
      margin: 0 0 18px;
    }}
    h3 {{
      font-family: "Fraunces", serif;
      font-weight: 500; font-size: 18px;
      margin: 0 0 14px;
    }}
    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 28px 32px;
      margin: 24px 0;
      backdrop-filter: blur(30px) saturate(150%);
      box-shadow: 0 1px 0 rgba(255,255,255,0.06) inset, 0 24px 64px -32px rgba(0,0,0,0.7);
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
      margin: 28px 0 40px;
    }}
    .stat {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 20px;
    }}
    .stat .stat-label {{
      font: 500 11px "JetBrains Mono", monospace;
      letter-spacing: 0.18em; text-transform: uppercase;
      color: var(--text-faint);
      margin-bottom: 8px;
    }}
    .stat .stat-value {{
      font-family: "Fraunces", serif;
      font-size: 22px; letter-spacing: -0.01em;
    }}
    .stat .stat-detail {{
      font-family: "JetBrains Mono", monospace;
      font-size: 11px; color: var(--c-recovery);
      margin-top: 2px;
    }}
    .viz-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
      gap: 22px;
      margin: 18px 0;
    }}
    .viz-card {{
      background: white;
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 12px 32px -16px rgba(0,0,0,0.6);
    }}
    .viz-card h3 {{
      color: #1a1d2e;
      font-family: "Inter", sans-serif;
      font-weight: 600;
      font-size: 14px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin: 0 0 12px;
    }}
    .viz-card img {{ width: 100%; height: auto; display: block; border-radius: 8px; }}
    .confusion-img {{
      width: 100%; max-width: 980px;
      margin: 0 auto;
      display: block; border-radius: 12px;
      background: white; padding: 16px;
    }}
    .table-wrap {{ overflow-x: auto; margin: 16px 0 8px; }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 13.5px;
      font-family: "JetBrains Mono", monospace;
      min-width: 760px;
    }}
    th, td {{
      padding: 11px 14px;
      text-align: right;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }}
    th {{
      background: rgba(255,255,255,0.03);
      color: var(--text-faint);
      font-weight: 500;
      font-size: 10.5px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    table.leaderboard-table tr.ensemble-row-top td {{
      background: linear-gradient(90deg, rgba(0,217,192,0.10), rgba(139,92,246,0.06));
      color: var(--text);
    }}
    table.leaderboard-table tr.ensemble-row-top td:nth-child(2)::after {{
      content: " ★";
      color: var(--c-recovery);
    }}
    code {{
      font-family: "JetBrains Mono", monospace;
      background: rgba(255,255,255,0.05);
      padding: 1px 6px;
      border-radius: 4px;
      color: var(--text);
      font-size: 12.5px;
    }}
    .footer {{
      margin-top: 60px;
      padding-top: 24px;
      border-top: 1px solid var(--border);
      font: 500 11px "JetBrains Mono", monospace;
      color: var(--text-faint);
      letter-spacing: 0.12em;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="nav">
      <div class="brand">Sleep Twin <small>Model Comparison Report</small></div>
      <div class="brand"><small>v0.2 · {len(summary_df)} base learners · {len(ensemble_rows)} ensemble configs</small></div>
    </div>

    <h1>How the models actually behave</h1>
    <p class="lede">
      A behavior-first comparison of the five trained base learners and the {len(ensemble_rows)}
      ensemble + HMM configurations. Each panel separates a different question — minority-stage
      recall, confusion patterns, ranking trade-offs across competing metrics. The final ensemble
      leaderboard at the bottom is the headline result; the per-model panels above explain why
      different models earn their places on it.
    </p>

    <div class="stats-grid">
      <div class="stat">
        <div class="stat-label">Best base macro F1</div>
        <div class="stat-value">{best_overall['model']}</div>
        <div class="stat-detail">{best_overall['macro_f1']:.3f}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Best Wake recall</div>
        <div class="stat-value">{best_wake['model']}</div>
        <div class="stat-detail">{best_wake['wake_recall']:.3f}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Best REM recall</div>
        <div class="stat-value">{best_rem['model']}</div>
        <div class="stat-detail">{best_rem['rem_recall']:.3f}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Most even recall</div>
        <div class="stat-value">{most_balanced['model']}</div>
        <div class="stat-detail">{most_balanced['recall_balance']:.3f}</div>
      </div>
    </div>

    <div class="card">
      <h2>Behavior visuals</h2>
      <p style="color: var(--text-muted); margin: 0 0 18px;">
        Six panels separated by question. Each card embeds a single panel so the labels stay readable.
      </p>
      <div class="viz-grid">
        <div class="viz-card"><h3>Model skill fingerprint</h3><img src="{panel_paths['skill'].name}" alt="Model skill fingerprint"></div>
        <div class="viz-card"><h3>Minority-stage frontier</h3><img src="{panel_paths['frontier'].name}" alt="Minority-stage frontier"></div>
        <div class="viz-card"><h3>Confusion fingerprint</h3><img src="{panel_paths['errors'].name}" alt="Confusion fingerprint"></div>
        <div class="viz-card"><h3>Metric constellation</h3><img src="{panel_paths['constellation'].name}" alt="Metric constellation"></div>
        <div class="viz-card"><h3>Rank ribbons</h3><img src="{panel_paths['ranks'].name}" alt="Rank ribbons"></div>
        <div class="viz-card"><h3>Top-model radar</h3><img src="{panel_paths['radar'].name}" alt="Top model radar"></div>
      </div>
    </div>

    <div class="card">
      <h2>Confusion matrices</h2>
      <p style="color: var(--text-muted); margin: 0 0 18px;">
        Each cell shows row-normalized percent and raw count. Rows are actual stages; columns are predicted stages.
      </p>
      <img class="confusion-img" src="{confusion_path.name}" alt="Normalized confusion matrices">
    </div>

    <div class="card">
      <h2>Base-learner summary</h2>
      <div class="table-wrap">{summary_table.to_html(index=False, escape=False)}</div>
    </div>

    <div class="card">
      <h2>Per-class F1</h2>
      <div class="table-wrap">{class_pivot.to_html(escape=False)}</div>
    </div>

    <div class="card">
      <h2>Ensemble + HMM leaderboard <small style="color: var(--text-faint); font-size: 13px; font-weight: 400;">— headline result, sorted by macro F1</small></h2>
      <p style="color: var(--text-muted); margin: 0 0 14px;">
        The full ablation across every base model, every ensemble head (arithmetic mean, geometric mean,
        logistic-regression stack), and HMM/Viterbi smoothing applied per subject. The top two rows are
        the project's co-winners depending on which metric you optimize.
      </p>
      <div class="table-wrap">{ensemble_html}</div>
    </div>

    <div class="card">
      <h2>Interpretation</h2>
      <p>
        <strong style="color: var(--text);">CatBoost is the standout single learner.</strong>
        Its raw probabilities have the highest balanced accuracy among single boosters and
        the highest minority-class recall. HMM smoothing then converts that minority-class
        strength into the project's best macro F1 (0.6993) and best balanced accuracy
        (0.7159) — without sacrificing the gains.
      </p>
      <p>
        <strong style="color: var(--text);">The geometric-mean ensemble + HMM (avg_geo_hmm) is the headline number.</strong>
        Log-averaging the base probabilities and then Viterbi-decoding the result crosses
        80 % accuracy with Cohen's κ = 0.5695 — the strongest end-to-end configuration in
        the project. Geometric mean edges arithmetic mean by a hair because log-averaging
        is more robust to one model being confidently wrong.
      </p>
      <p>
        <strong style="color: var(--text);">The logistic-regression stack underperforms simple averaging.</strong>
        Base models perfectly memorize their training set, so the meta-learner fits on
        optimistic in-sample probabilities. Proper out-of-fold stacking would fix this at
        the cost of 5× the boosting compute — deferred work.
      </p>
      <p>
        <strong style="color: var(--text);">BiLSTM is the weakest single learner but a useful ensemble ingredient.</strong>
        Macro F1 0.6315 alone, but it contributes meaningfully to <code>avg_geo_hmm</code> —
        its different inductive bias gives the ensemble information the boosters lack.
      </p>
    </div>

    <div class="footer">
      Generated from <code>artifacts/models/</code> · scripts/visualize_model_results.py · Sleep Twin v0.2
    </div>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Sleep Twin model-comparison HTML report.")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Project root (defaults to two levels up from this script)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "visualizations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("loading base-learner metrics...")
    loaded = load_base_metrics(args.root)
    print(f"  {len(loaded)} base learner(s) loaded")
    ensemble_rows = load_ensemble_rows(args.root)
    print(f"  {len(ensemble_rows)} ensemble row(s) loaded from {ENSEMBLE_METRICS_PATH}")

    labels = _label_names(loaded)
    summary_df, class_df, error_df = build_tables(loaded)

    print("rendering panels...")
    confusion_path = args.output_dir / "confusion_matrices_normalized.png"
    html_path = args.output_dir / "model_comparison_report.html"
    summary_csv = args.output_dir / "model_comparison_summary.csv"
    class_csv = args.output_dir / "class_metric_table.csv"
    error_csv = args.output_dir / "error_fingerprint_table.csv"

    plot_confusion_matrices(loaded, labels, confusion_path)
    panel_paths = plot_individual_panels(summary_df, error_df, labels, args.output_dir)
    summary_df.to_csv(summary_csv, index=False)
    class_df.to_csv(class_csv, index=False)
    error_df.to_csv(error_csv, index=False)
    write_html_report(html_path, confusion_path, panel_paths, summary_df, class_df, ensemble_rows)

    print()
    print(f"wrote {confusion_path}")
    for p in panel_paths.values():
        print(f"wrote {p}")
    print(f"wrote {summary_csv}")
    print(f"wrote {class_csv}")
    print(f"wrote {error_csv}")
    print(f"wrote {html_path}")
    print()
    print(f"open file://{html_path.resolve()} in a browser")


if __name__ == "__main__":
    main()
