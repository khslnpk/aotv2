"""Sequence model trainer: BiLSTM with attention + focal loss + wider context.

Improvements vs v1:
- Sequence radius 15 (31-epoch windows ~15min context each side, was 10).
- 3-layer BiLSTM with hidden size 128, dropout 0.3, layer norm.
- Multi-head additive attention pooling.
- Focal cross-entropy (gamma=1.5) to push the model on minority Wake epochs.
- AdamW + OneCycleLR with cosine annealing.
- Probabilities persisted for stacking and HMM smoothing.
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sleep_twin_v2.evaluation import evaluate_predictions, metrics_row, write_json
from sleep_twin_v2.features import load_feature_cache
from sleep_twin_v2.labels import LABEL_NAMES_3CLASS
from sleep_twin_v2.paths import DEFAULT_FEATURE_DIR, DEFAULT_MODEL_DIR
from sleep_twin_v2.splits import make_group_holdout_split, split_subject_summary


def _resolve_torch():
    import torch

    return torch


def _device(name: str):
    torch = _resolve_torch()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_sequences(
    X: np.ndarray,
    subjects: np.ndarray,
    epoch_times: np.ndarray,
    center_indices: np.ndarray,
    radius: int,
) -> np.ndarray:
    seq_len = radius * 2 + 1
    out = np.zeros((len(center_indices), seq_len, X.shape[1]), dtype=np.float32)
    orders: dict[str, np.ndarray] = {}
    positions = np.full(len(subjects), -1, dtype=np.int64)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        order = idx[np.argsort(epoch_times[idx])]
        orders[subject] = order
        positions[order] = np.arange(len(order))
    for row, gid in enumerate(center_indices):
        subject = str(subjects[gid])
        order = orders[subject]
        pos = int(positions[gid])
        lo = max(0, pos - radius)
        hi = min(len(order), pos + radius + 1)
        dst_lo = radius - (pos - lo)
        dst_hi = dst_lo + (hi - lo)
        out[row, dst_lo:dst_hi, :] = X[order[lo:hi]]
    return out


def _make_model(input_dim: int, num_classes: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.30):
    torch = _resolve_torch()
    from torch import nn

    bi_dim = hidden_size * 2

    class BiLSTMAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_norm = nn.LayerNorm(input_dim)
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.attn = nn.Sequential(
                nn.Linear(bi_dim, max(bi_dim // 2, 16)),
                nn.Tanh(),
                nn.Linear(max(bi_dim // 2, 16), 1),
            )
            self.head = nn.Sequential(
                nn.Linear(bi_dim, max(bi_dim // 2, 16)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(bi_dim // 2, 16), num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layer_norm(x)
            hidden, _ = self.lstm(x)
            weights = torch.softmax(self.attn(hidden).squeeze(-1), dim=1)
            pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
            return self.head(pooled)

    return BiLSTMAttn()


def _focal_ce(logits, targets, class_weights, gamma: float = 1.5):
    torch = _resolve_torch()
    log_probs = torch.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)
    focal = (1.0 - probs) ** gamma
    weighted = class_weights.unsqueeze(0) * focal * one_hot * (-log_probs)
    return weighted.sum(dim=1).mean()


def _class_weights(y: np.ndarray, num_classes: int, device):
    torch = _resolve_torch()
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _make_loader(X, y, batch_size, shuffle, device):
    torch = _resolve_torch()
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _predict_probs(model, loader, device, num_classes: int) -> np.ndarray:
    torch = _resolve_torch()
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            out.append(probs.cpu().numpy())
    return np.concatenate(out, axis=0)


def train_sequence_model(
    feature_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    val_size: float = 0.15,
    seed: int = 42,
    sequence_radius: int = 15,
    batch_size: int = 256,
    epochs: int = 60,
    learning_rate: float = 1.5e-3,
    patience: int = 10,
    device_name: str = "auto",
    amp: bool = True,
    focal_gamma: float = 1.5,
    hidden_size: int = 128,
    num_layers: int = 3,
    dropout: float = 0.30,
) -> dict:
    torch = _resolve_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    fs = load_feature_cache(feature_path)
    labels = LABEL_NAMES_3CLASS
    num_classes = len(labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(device_name)
    print(f"[seq] device={device}")
    if device.type == "cuda":
        print(f"  gpu={torch.cuda.get_device_name(0)}")

    train_idx, val_idx, test_idx = make_group_holdout_split(
        fs.subject_ids, test_size=test_size, val_size=val_size, random_state=seed
    )
    write_json(output_dir / "split_subjects.json", split_subject_summary(fs.subject_ids, train_idx, val_idx, test_idx))

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(fs.X[train_idx])
    X_train_scaled = scaler.fit_transform(X_train_imp).astype(np.float32)
    X_all_scaled = scaler.transform(imputer.transform(fs.X)).astype(np.float32)
    joblib.dump(
        {"imputer": imputer, "scaler": scaler, "feature_names": fs.feature_names, "sequence_radius": sequence_radius},
        output_dir / "preprocessor.joblib",
    )

    y_train = fs.y[train_idx]
    y_val = fs.y[val_idx]
    y_test = fs.y[test_idx]

    train_seq = _build_sequences(X_all_scaled, fs.subject_ids, fs.epoch_times, train_idx, sequence_radius)
    val_seq = _build_sequences(X_all_scaled, fs.subject_ids, fs.epoch_times, val_idx, sequence_radius)
    test_seq = _build_sequences(X_all_scaled, fs.subject_ids, fs.epoch_times, test_idx, sequence_radius)
    all_idx = np.arange(len(fs.y))
    all_seq = _build_sequences(X_all_scaled, fs.subject_ids, fs.epoch_times, all_idx, sequence_radius)

    train_loader = _make_loader(train_seq, y_train, batch_size, True, device)
    val_loader = _make_loader(val_seq, y_val, batch_size, False, device)
    test_loader = _make_loader(test_seq, y_test, batch_size, False, device)
    all_loader = _make_loader(all_seq, fs.y, batch_size, False, device)

    model = _make_model(fs.X.shape[1], num_classes, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    class_weights = _class_weights(y_train, num_classes, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    use_amp = bool(amp and device.type == "cuda")
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state = None
    best_score = -np.inf
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(xb)
                loss = _focal_ce(logits, yb, class_weights, gamma=focal_gamma)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()
            running_loss += float(loss.item()) * len(yb)
            seen += len(yb)

        val_probs = _predict_probs(model, val_loader, device, num_classes)
        val_pred = np.argmax(val_probs, axis=1)
        val_metrics = evaluate_predictions(y_val, val_pred, labels)
        val_score = val_metrics["macro_f1"]
        print(
            f"  epoch {epoch:03d}: loss={running_loss/max(seen,1):.4f} "
            f"val_macroF1={val_score:.4f} val_bal={val_metrics['balanced_accuracy']:.4f}"
        )
        if val_score > best_score + 1e-4:
            best_score = val_score
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_loader_eval = _make_loader(train_seq, y_train, batch_size, False, device)
    train_probs = _predict_probs(model, train_loader_eval, device, num_classes)
    val_probs = _predict_probs(model, val_loader, device, num_classes)
    test_probs = _predict_probs(model, test_loader, device, num_classes)
    all_probs = _predict_probs(model, all_loader, device, num_classes)

    test_pred = np.argmax(test_probs, axis=1)
    metrics = evaluate_predictions(y_test, test_pred, labels)
    row = metrics_row("bilstm_attention_v2", metrics)
    write_json(output_dir / "bilstm_attention_v2_metrics.json", metrics)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": fs.X.shape[1],
            "num_classes": num_classes,
            "sequence_radius": sequence_radius,
        },
        output_dir / "bilstm_attention_v2.pt",
    )

    np.savez_compressed(
        output_dir / "sequence_probabilities.npz",
        train_probs=train_probs.astype(np.float32),
        val_probs=val_probs.astype(np.float32),
        test_probs=test_probs.astype(np.float32),
        all_probs=all_probs.astype(np.float32),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    leaderboard = output_dir / "leaderboard.csv"
    with leaderboard.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "cohen_kappa"])
        writer.writeheader()
        writer.writerow(row)

    print(
        f"  bilstm_attention_v2: acc={row['accuracy']:.4f} bal={row['balanced_accuracy']:.4f} "
        f"macroF1={row['macro_f1']:.4f} kappa={row['cohen_kappa']:.4f}"
    )
    return {"row": row, "proba_path": output_dir / "sequence_probabilities.npz"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train improved BiLSTM sequence model.")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURE_DIR / "sleep_features_v2.npz")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR / "sequence")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sequence-radius", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--learning-rate", type=float, default=1.5e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.30)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    train_sequence_model(
        feature_path=args.features,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        sequence_radius=args.sequence_radius,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device_name=args.device,
        amp=not args.no_amp,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()
