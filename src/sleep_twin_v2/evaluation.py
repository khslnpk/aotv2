from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True
        ),
    }


def metrics_row(name: str, metrics: dict) -> dict:
    return {
        "model": name,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "cohen_kappa": metrics["cohen_kappa"],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def format_metrics_table(rows: list[dict]) -> str:
    header = "| Model | Acc | Bal Acc | Macro F1 | Wtd F1 | Kappa |"
    sep = "|---|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['weighted_f1']:.4f} | {row['cohen_kappa']:.4f} |"
        )
    return "\n".join(lines)
