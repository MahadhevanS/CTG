"""
Lightweight classical CTG feature extraction for the Phase 0 trivial
baselines (plan section 0.3: "a day of work" sanity-check baselines, not
the rigorous validated Phase 1 FIGO extractor). Operates directly on the
1 Hz gap-filled signal already produced by the Phase 0 pipeline.

Deliberately simplified vs. the real FIGO baseline/STV/LTV definitions
(which need finer-than-1Hz resolution and multi-window iteration) --
Phase 1 will replace this with the validated, unit-tested extractor. This
module exists only to answer "does the Crossformer even beat a handful of
hand-crafted numbers", per the plan's own reasoning.
"""
import numpy as np


def extract_classical_features(fhr: np.ndarray, uc: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Args:
        fhr, uc: (T,) 1 Hz signals (already gap-filled where possible).
        mask: (T,) bool, True = still missing/padded.
    Returns:
        (5,) float32 feature vector:
        [baseline_fhr, stv_approx, ltv_approx, n_accel, n_decel]
    """
    valid = ~mask
    if valid.sum() < 30:  # need at least ~30s of real signal to say anything
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    fhr_valid = fhr[valid]

    baseline = float(np.median(fhr_valid))

    # STV approx: mean absolute sample-to-sample difference (1 Hz proxy for
    # the true 3.75s-epoch Dawes-Redman STV).
    diffs = np.abs(np.diff(fhr_valid))
    stv = float(diffs.mean()) if len(diffs) > 0 else 0.0

    # LTV approx: mean of per-60s-window (range = max-min), on valid samples only.
    ltv_vals = []
    t = np.where(valid)[0]
    for start in range(0, len(fhr), 60):
        end = start + 60
        window_valid_mask = valid[start:end]
        if window_valid_mask.sum() < 10:
            continue
        window_vals = fhr[start:end][window_valid_mask]
        ltv_vals.append(window_vals.max() - window_vals.min())
    ltv = float(np.mean(ltv_vals)) if ltv_vals else 0.0

    # Accel/decel counts: threshold crossings of +/- 15 bpm from baseline,
    # sustained >= 15s (approximated at 1Hz as >=15 consecutive samples).
    above = (fhr >= baseline + 15) & valid
    below = (fhr <= baseline - 15) & valid
    n_accel = _count_sustained_runs(above, min_len=15)
    n_decel = _count_sustained_runs(below, min_len=15)

    return np.array([baseline, stv, ltv, float(n_accel), float(n_decel)], dtype=np.float32)


def _count_sustained_runs(binary_mask: np.ndarray, min_len: int) -> int:
    count = 0
    run = 0
    for v in binary_mask:
        if v:
            run += 1
        else:
            if run >= min_len:
                count += 1
            run = 0
    if run >= min_len:
        count += 1
    return count


def extract_features_batch(fhr: np.ndarray, uc: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Batched version. fhr/uc/mask: (N, T). Returns (N, 5)."""
    n = fhr.shape[0]
    out = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        out[i] = extract_classical_features(fhr[i], uc[i], mask[i])
    return out
