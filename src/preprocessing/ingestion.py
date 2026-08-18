"""
Raw signal + clinical metadata loading for CTU-CHB records.
"""
import os
from typing import Tuple

import numpy as np
import pandas as pd
import wfdb

NATIVE_FS: float = 4.0
"""CTU-CHB native sampling frequency (Hz), per PROTOCOL.md decision #5."""


def load_ctu_chb_record(record_path: str, strict: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Loads a single CTU-CHB record (FHR + UC) at its native 4 Hz rate.

    Args:
        record_path: Path to the record without extension.
        strict: If True, re-raise load failures. If False (default), log and
                return empty arrays with fs=0.0 so bulk-processing callers can
                skip bad records without aborting.

    Returns:
        fhr, uc: 1D float64 arrays. Missing/absent samples are 0.0.
        fs: Sampling frequency (4.0 on success, 0.0 on failure).
    """
    try:
        record = wfdb.rdrecord(record_path)
        signals = record.p_signal
        fhr = signals[:, 0].copy()
        uc = signals[:, 1].copy()
        fs = float(record.fs)

        fhr = np.where(np.isnan(fhr), 0.0, fhr)
        uc = np.where(np.isnan(uc), 0.0, uc)

        if abs(fs - NATIVE_FS) > 1e-6:
            raise ValueError(
                f"Record '{record_path}' has fs={fs} Hz, expected {NATIVE_FS} Hz. "
                "CTU-CHB is documented at a fixed native rate; an unexpected fs "
                "indicates a corrupt or non-standard file, not something to "
                "silently resample past."
            )
        return fhr, uc, fs

    except Exception as e:
        if strict:
            raise RuntimeError(f"Failed to load CTU-CHB record '{record_path}': {e}") from e
        print(f"  [ERROR] Failed to load record '{record_path}': {e}")
        return np.array([]), np.array([]), 0.0


def load_clinical_metadata(metadata_path: str) -> pd.DataFrame:
    """Loads clinical_metadata.csv (produced by scripts/extract_metadata.py)."""
    df = pd.read_csv(metadata_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["record_id"] = df["record_id"].astype(str)
    return df.set_index("record_id")


def record_path_for(raw_dir: str, record_id: str) -> str:
    return os.path.join(raw_dir, str(record_id))
