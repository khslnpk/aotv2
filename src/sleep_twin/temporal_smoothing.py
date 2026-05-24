"""HMM/Viterbi post-processing for sleep-stage probability sequences.

Sleep stages have strong transition structure (you rarely jump from Wake straight
to REM; NREM dominates the interior of a night). Treating model probabilities as
HMM emission likelihoods and decoding with Viterbi cleans up flicker noise and
typically buys +1-3% accuracy and +2-5% Cohen's kappa.

The transition matrix is estimated from training labels and the emission model
uses calibrated classifier probabilities directly.
"""

from __future__ import annotations

import numpy as np


def fit_transition_matrix(y_sequences: list[np.ndarray], num_classes: int, smoothing: float = 1.0) -> np.ndarray:
    """Estimate per-step transition probabilities with Laplace smoothing."""
    counts = np.full((num_classes, num_classes), smoothing, dtype=np.float64)
    for seq in y_sequences:
        if len(seq) < 2:
            continue
        prev = seq[:-1]
        curr = seq[1:]
        np.add.at(counts, (prev, curr), 1.0)
    counts /= counts.sum(axis=1, keepdims=True)
    return counts


def fit_initial_probs(y_sequences: list[np.ndarray], num_classes: int, smoothing: float = 1.0) -> np.ndarray:
    counts = np.full(num_classes, smoothing, dtype=np.float64)
    for seq in y_sequences:
        if seq.size:
            counts[int(seq[0])] += 1.0
    return counts / counts.sum()


def viterbi_decode(
    log_emissions: np.ndarray,
    log_transitions: np.ndarray,
    log_initial: np.ndarray,
) -> np.ndarray:
    """Run Viterbi over log-space emissions and transitions for ONE sequence.

    log_emissions: (T, K) log p(observation_t | state=k)
    log_transitions: (K, K) log p(state_t | state_{t-1})
    log_initial: (K,) log p(state_0)
    Returns: (T,) decoded state sequence.
    """
    T, K = log_emissions.shape
    dp = np.full((T, K), -np.inf, dtype=np.float64)
    back = np.zeros((T, K), dtype=np.int64)
    dp[0] = log_initial + log_emissions[0]
    for t in range(1, T):
        scores = dp[t - 1, :, None] + log_transitions
        back[t] = np.argmax(scores, axis=0)
        dp[t] = scores[back[t], np.arange(K)] + log_emissions[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def smooth_subject_sequences(
    proba: np.ndarray,
    subject_ids: np.ndarray,
    epoch_times: np.ndarray,
    log_transitions: np.ndarray,
    log_initial: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Apply Viterbi smoothing per subject (chronological order)."""
    out = np.zeros(len(proba), dtype=np.int64)
    log_em_all = np.log(np.clip(proba, epsilon, 1.0))
    for subject in np.unique(subject_ids):
        mask = subject_ids == subject
        idx = np.where(mask)[0]
        order = idx[np.argsort(epoch_times[idx])]
        log_em = log_em_all[order]
        decoded = viterbi_decode(log_em, log_transitions, log_initial)
        out[order] = decoded
    return out
