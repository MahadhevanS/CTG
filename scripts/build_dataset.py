"""
Assembles the Phase-0 modelling cache: for every eligible record, runs the
frozen preprocessing chain (PROTOCOL.md section 3) on the final 30-min
window and saves a single consolidated .npz with UN-NORMALISED signals
(FHR z-norm and UC min-max are fold-conditional / fold-independent
respectively -- see src/preprocessing/scalers.py -- and are applied at
training time, never baked into this cache, to avoid leaking any fold's
statistics into another).

Short recordings (< 30 min total) are left-padded (start of the array) with
zeros and marked missing in the mask, since the window is "last-aligned".

Usage:
    python scripts/build_dataset.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from preprocessing.ingestion import load_ctu_chb_record, load_clinical_metadata, record_path_for
from preprocessing.signal_processing import NATIVE_FS, process_window_1hz

RAW_DIR = os.path.join(ROOT, "data", "raw", "ctu-chb-intrapartum-cardiotocography-database-1.0.0")
ELIGIBLE_CSV = os.path.join(ROOT, "data", "processed", "eligible_records.csv")
OUT_PATH = os.path.join(ROOT, "data", "processed", "phase0_dataset.npz")

WINDOW_MINUTES = 30
WINDOW_SAMPLES_4HZ = int(WINDOW_MINUTES * 60 * NATIVE_FS)  # 7200
T_1HZ = WINDOW_MINUTES * 60  # 1800


def final_window_4hz(signal: np.ndarray) -> np.ndarray:
    if len(signal) >= WINDOW_SAMPLES_4HZ:
        return signal[-WINDOW_SAMPLES_4HZ:]
    return signal


def left_pad_to(arr: np.ndarray, target_len: int, pad_value: float) -> np.ndarray:
    if len(arr) >= target_len:
        return arr[-target_len:]
    pad = np.full(target_len - len(arr), pad_value, dtype=arr.dtype)
    return np.concatenate([pad, arr])


def main():
    eligible = pd.read_csv(ELIGIBLE_CSV)
    eligible["record_id"] = eligible["record_id"].astype(str)
    n = len(eligible)
    print(f"Building Phase-0 dataset cache for {n} eligible records...")

    fhr_all = np.zeros((n, T_1HZ), dtype=np.float32)
    uc_all = np.zeros((n, T_1HZ), dtype=np.float32)
    mask_all = np.ones((n, T_1HZ), dtype=bool)  # True = missing/padded by default
    record_ids = []
    y_primary = np.zeros(n, dtype=np.int64)
    y_secondary = np.zeros(n, dtype=np.int64)
    ph = np.full(n, np.nan, dtype=np.float64)
    bdecf = np.full(n, np.nan, dtype=np.float64)
    stage2_min = np.full(n, np.nan, dtype=np.float64)
    stage2_known = np.zeros(n, dtype=bool)

    for i, row in eligible.iterrows():
        rid = row["record_id"]
        path = record_path_for(RAW_DIR, rid)
        fhr, uc, fs = load_ctu_chb_record(path, strict=True)

        fhr_win = final_window_4hz(fhr)
        uc_win = final_window_4hz(uc)
        out = process_window_1hz(fhr_win, uc_win)

        actual_t = len(out["fhr"])
        fhr_1hz = left_pad_to(out["fhr"], T_1HZ, 0.0)
        uc_1hz = left_pad_to(out["uc"], T_1HZ, 0.0)
        mask_1hz = left_pad_to(out["missingness_mask"].astype(np.float32), T_1HZ, 1.0).astype(bool)

        fhr_all[i] = fhr_1hz
        uc_all[i] = uc_1hz
        mask_all[i] = mask_1hz
        record_ids.append(rid)

        y_primary[i] = int(row["ph"] <= 7.05)
        y_secondary[i] = int(row["bdecf"] > 12) if not pd.isna(row["bdecf"]) else 0
        ph[i] = row["ph"]
        bdecf[i] = row["bdecf"]
        if not pd.isna(row.get("stage2_min", np.nan)):
            stage2_min[i] = row["stage2_min"]
            stage2_known[i] = True

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        fhr=fhr_all, uc=uc_all, missingness_mask=mask_all,
        record_id=np.array(record_ids),
        y_primary=y_primary, y_secondary=y_secondary,
        ph=ph, bdecf=bdecf,
        stage2_min=stage2_min, stage2_known=stage2_known,
    )

    print(f"Saved -> {OUT_PATH}")
    print(f"  Shapes: fhr {fhr_all.shape}, uc {uc_all.shape}, mask {mask_all.shape}")
    print(f"  y_primary positives: {y_primary.sum()} / {n} ({100*y_primary.mean():.1f}%)")
    print(f"  y_secondary positives: {y_secondary.sum()} / {n} ({100*y_secondary.mean():.1f}%)")
    print(f"  Mean missingness fraction: {mask_all.mean():.3f}")
    print(f"  Stage-II duration known: {stage2_known.sum()} / {n}")


if __name__ == "__main__":
    main()
