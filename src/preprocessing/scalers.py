"""
Normalisation applied after gap/artifact processing, at dataset-assembly time
(once train-fold membership for a given CV split is known). See PROTOCOL.md
section 3's interpretive note for why FHR and UC are normalised differently.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class FHRScaler:
    """Per-channel z-score scaler for FHR, fit by pooling all training-fold
    records together. Must be re-fit for every outer/inner CV fold -- never
    shared across folds, or it leaks test-fold statistics."""
    mean: float
    std: float

    @classmethod
    def fit(cls, fhr_windows: list) -> "FHRScaler":
        """
        Args:
            fhr_windows: list of 1D arrays (1 Hz, gap-filled), train-fold only.
        """
        pooled = np.concatenate([w for w in fhr_windows if len(w) > 0])
        mean = float(pooled.mean())
        std = float(pooled.std())
        if std == 0.0:
            std = 1.0
        return cls(mean=mean, std=std)

    def transform(self, fhr_window: np.ndarray) -> np.ndarray:
        return (fhr_window - self.mean) / self.std


def uc_per_record_minmax(uc_window: np.ndarray) -> np.ndarray:
    """
    Self-referential min-max normalisation to [0, 1] for a single record's UC
    window. Fold-independent by construction -- no leakage possible, since
    each record only ever uses its own min/max.
    """
    if len(uc_window) == 0:
        return uc_window
    lo, hi = float(uc_window.min()), float(uc_window.max())
    if hi - lo < 1e-8:
        return np.zeros_like(uc_window)
    return (uc_window - lo) / (hi - lo)
