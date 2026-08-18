"""Shared training utilities for Phase 0 baseline reproduction."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_pos_weight(y: np.ndarray) -> float:
    """BCEWithLogitsLoss pos_weight for severe class imbalance."""
    n_pos = max(int(y.sum()), 1)
    n_neg = len(y) - n_pos
    return float(n_neg / n_pos)


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Returns NaN (not an exception) if a fold has only one class present --
    this happens routinely here given ~3-4 positives per outer test fold."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI over a set of (e.g. per-seed or per-fold) scores."""
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(values.mean()), float(lo), float(hi)
