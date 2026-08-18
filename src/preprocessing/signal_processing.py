"""
Phase 0 preprocessing pipeline (PROTOCOL.md section 3).

    raw FHR (4 Hz)
      -> clip to physiological range; out-of-range -> gap
      -> gap detection
      -> gaps <= 15s: PCHIP interpolation
      -> gaps > 15s: leave as gap, carry missingness mask forward
      -> artifact removal (|delta| > 25 bpm/sample-step), re-interpolate
      -> downsample to 1 Hz by 4-sample median
      -> (z-norm / min-max applied later, at dataset-assembly time, once
         train-fold membership is known -- see scalers.py)

UC channel follows the same gap/artifact/downsample logic with its own
clip range and no z-normalisation.
"""
from typing import Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

NATIVE_FS = 4.0
MAX_GAP_SECONDS = 15.0
MAX_GAP_SAMPLES = int(MAX_GAP_SECONDS * NATIVE_FS)  # 60 samples

FHR_MIN, FHR_MAX = 50.0, 200.0
UC_MIN, UC_MAX = 0.0, 100.0

FHR_MAX_RATE_BPM_PER_SAMPLE = 25.0 / NATIVE_FS  # 25 bpm/sec -> per-sample delta at 4Hz


def _contiguous_runs(mask: np.ndarray):
    """Yields (start, end) [end exclusive] for each contiguous True run in mask."""
    if len(mask) == 0:
        return
    changes = np.diff(mask.astype(np.int8))
    starts = list(np.where(changes == 1)[0] + 1)
    ends = list(np.where(changes == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    for s, e in zip(starts, ends):
        yield s, e


def _interpolate_short_gaps(signal: np.ndarray, missing: np.ndarray,
                             lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCHIP-interpolates contiguous missing runs of length <= MAX_GAP_SAMPLES.
    Runs longer than that are left as gaps (signal set to 0.0, mask stays True).

    Returns:
        signal: filled signal (gaps > 15s left at 0.0).
        still_missing: mask, True where the gap remains unfilled (> 15s).
    """
    signal = signal.copy()
    still_missing = missing.copy()
    valid_idx = np.where(~missing)[0]

    if len(valid_idx) < 4:
        # Not enough anchor points for a cubic Hermite fit; nothing fillable.
        signal[missing] = 0.0
        return signal, missing.copy()

    interpolator = PchipInterpolator(valid_idx, signal[valid_idx], extrapolate=True)

    for start, end in _contiguous_runs(missing):
        length = end - start
        if length <= MAX_GAP_SAMPLES:
            fill_idx = np.arange(start, end)
            filled = np.clip(interpolator(fill_idx), lo, hi)
            signal[start:end] = filled
            still_missing[start:end] = False
        else:
            signal[start:end] = 0.0
            still_missing[start:end] = True

    return signal, still_missing


def process_channel(raw: np.ndarray, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the full gap-detect -> short-gap-interpolate -> artifact-reject ->
    re-interpolate chain for one channel (FHR or UC), at native 4 Hz.

    Args:
        raw: 1D raw signal at 4 Hz. Missing samples are 0.0 (dataset convention).
        lo, hi: physiological clip range for this channel.

    Returns:
        signal: processed signal at 4 Hz, gaps > 15s left at 0.0.
        missingness_mask: bool array, True where the sample is still missing
                           (unfilled gap > 15s) after the full chain.
    """
    signal = raw.copy().astype(np.float64)

    # Step 1: clip to physiological range; out-of-range -> gap.
    out_of_range = (signal < lo) | (signal > hi)
    # Original dropout convention in this dataset: exact 0.0 == missing.
    original_missing = (raw == 0.0)
    missing = out_of_range | original_missing
    signal[missing] = 0.0

    # Step 2/3: interpolate gaps <= 15s; gaps > 15s stay masked.
    signal, missing = _interpolate_short_gaps(signal, missing, lo, hi)

    # Step 4: artifact removal -- reject samples with an unphysiological
    # sample-to-sample jump, treat as new (short) gaps, re-interpolate once.
    # Only evaluated on currently-valid (non-missing) samples so we don't
    # flag the boundary of an already-known gap as an "artifact".
    valid = ~missing
    deltas = np.abs(np.diff(signal))
    jump_mask = np.zeros_like(missing)
    both_valid = valid[:-1] & valid[1:]
    artifact_positions = np.where((deltas > FHR_MAX_RATE_BPM_PER_SAMPLE) & both_valid)[0] + 1
    jump_mask[artifact_positions] = True

    if jump_mask.any():
        missing = missing | jump_mask
        signal[jump_mask] = 0.0
        signal, missing = _interpolate_short_gaps(signal, missing, lo, hi)

    return signal, missing


def downsample_median(signal: np.ndarray, factor: int = 4) -> np.ndarray:
    """4-sample median downsample (4 Hz -> 1 Hz for factor=4). Trailing remainder dropped."""
    n = (len(signal) // factor) * factor
    if n == 0:
        return np.array([])
    return np.median(signal[:n].reshape(-1, factor), axis=1)


def downsample_mask_any(mask: np.ndarray, factor: int = 4) -> np.ndarray:
    """Downsamples a boolean missingness mask: a downsampled sample is 'missing'
    if ANY of its constituent native-rate samples were missing (conservative)."""
    n = (len(mask) // factor) * factor
    if n == 0:
        return np.array([], dtype=bool)
    return mask[:n].reshape(-1, factor).any(axis=1)


def process_window_1hz(fhr_4hz: np.ndarray, uc_4hz: np.ndarray) -> dict:
    """
    Full Phase-0 pipeline for one FHR+UC window at native 4 Hz, producing the
    1 Hz, 3-channel (FHR, UC, missingness-mask) representation used as model
    input. FHR z-norm and UC per-record min-max are NOT applied here -- see
    scalers.py, since FHR's scaler must be fit on train-fold data only.

    Returns dict with keys: fhr (1Hz, gap-filled/masked), uc (1Hz),
    missingness_mask (1Hz, bool -- OR of FHR and UC missingness), fhr_missing_frac.
    """
    fhr_proc, fhr_missing = process_channel(fhr_4hz, FHR_MIN, FHR_MAX)
    uc_proc, uc_missing = process_channel(uc_4hz, UC_MIN, UC_MAX)

    fhr_1hz = downsample_median(fhr_proc, factor=4)
    uc_1hz = downsample_median(uc_proc, factor=4)
    fhr_missing_1hz = downsample_mask_any(fhr_missing, factor=4)
    uc_missing_1hz = downsample_mask_any(uc_missing, factor=4)
    combined_missing_1hz = fhr_missing_1hz | uc_missing_1hz

    return {
        "fhr": fhr_1hz,
        "uc": uc_1hz,
        "missingness_mask": combined_missing_1hz,
        "fhr_missing_frac": float(fhr_missing.mean()) if len(fhr_missing) else 1.0,
        "uc_missing_frac": float(uc_missing.mean()) if len(uc_missing) else 1.0,
    }
