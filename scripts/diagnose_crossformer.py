"""
Phase 0 reproduction-gap diagnostic (NOT part of the frozen Phase 0 harness
-- run_baseline_reproduction.py and its results stay untouched). Investigates
whether the 0.604-vs-0.822 Crossformer gap (docs/baseline_reproduction_report.md)
is a fixable training/hyperparameter problem or a hard ceiling from N_pos=17,
before deciding whether to invest more in the base model or move to Phase 1.

Two parts, both restricted to a small diagnostic fold set (seed=0's 5 outer
folds -- cheap to run at ~1s/epoch on this GPU, still gives real fold-to-fold
spread):

1. --curves: per-epoch train loss / val AUROC / test AUROC logged with NO
   early stopping (100 epochs), to see whether training is under- or
   over-fitting relative to the frozen 40-epoch/patience-8 cutoff.
2. --sweep: a curated (not exhaustive) set of hyperparameter configs, each
   evaluated with early stopping exactly like the frozen harness, mean test
   AUROC compared against the frozen defaults on the same 5 folds.

Usage:
    python scripts/diagnose_crossformer.py --curves
    python scripts/diagnose_crossformer.py --sweep
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cv.nested_cv import outer_splits
from preprocessing.scalers import FHRScaler, uc_per_record_minmax
from models.crossformer import CTGCrossformer
from training.train_utils import set_seed, compute_pos_weight, safe_auroc

DATA_PATH = os.path.join(ROOT, "data", "processed", "phase0_dataset.npz")
OUT_DIR = os.path.join(ROOT, "docs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIAG_SEED = 0  # diagnostic fold set: all 5 outer folds of seed 0


def load_data():
    d = np.load(DATA_PATH, allow_pickle=True)
    return d["fhr"], d["uc"], d["missingness_mask"], d["y_primary"], d["record_id"]


def prep_signals(fhr, uc, mask, train_idx):
    train_fhr_windows = [fhr[i][~mask[i]] for i in train_idx if (~mask[i]).any()]
    scaler = FHRScaler.fit(train_fhr_windows)
    fhr_norm = np.stack([scaler.transform(fhr[i]) for i in range(len(fhr))])
    uc_norm = np.stack([uc_per_record_minmax(uc[i]) for i in range(len(uc))])
    fhr_norm = np.where(mask, 0.0, fhr_norm)
    uc_norm = np.where(mask, 0.0, uc_norm)
    x = np.stack([fhr_norm, uc_norm], axis=-1).astype(np.float32)
    return x, mask.astype(np.float32)


def diagnostic_folds(fhr, uc, mask, y, groups):
    """Same outer-split + val-carve logic as run_baseline_reproduction.py,
    restricted to seed=0's 5 folds."""
    from sklearn.model_selection import train_test_split
    folds = outer_splits(y, groups, n_splits=5, seed=DIAG_SEED)
    out = []
    for fold_i, fold in enumerate(folds):
        train_idx_full, test_idx = fold.train_idx, fold.test_idx
        if y[train_idx_full].sum() >= 2:
            tr_idx, val_idx = train_test_split(
                train_idx_full, test_size=0.15, stratify=y[train_idx_full],
                random_state=DIAG_SEED,
            )
        else:
            tr_idx, val_idx = train_test_split(
                train_idx_full, test_size=0.15, random_state=DIAG_SEED,
            )
        x_all, mask_f_all = prep_signals(fhr, uc, mask, tr_idx)
        out.append({
            "fold_i": fold_i,
            "tr_idx": tr_idx, "val_idx": val_idx, "test_idx": test_idx,
            "x_all": x_all, "mask_f_all": mask_f_all,
        })
    return out


def run_curves(epochs=100):
    """Part 1: log per-epoch train loss / val AUROC / test AUROC, no early
    stopping, on all 5 diagnostic folds."""
    fhr, uc, mask, y, groups = load_data()
    folds = diagnostic_folds(fhr, uc, mask, y, groups)

    all_curves = {}
    for f in folds:
        fold_i = f["fold_i"]
        tr_idx, val_idx, test_idx = f["tr_idx"], f["val_idx"], f["test_idx"]
        x_all, mask_f_all = f["x_all"], f["mask_f_all"]
        y_test = y[test_idx]

        set_seed(DIAG_SEED)
        model = CTGCrossformer(seq_len=x_all.shape[1]).to(DEVICE)
        pos_weight = torch.tensor([compute_pos_weight(y[tr_idx])], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

        xt = torch.tensor(x_all[tr_idx], device=DEVICE)
        mt = torch.tensor(mask_f_all[tr_idx], device=DEVICE)
        yt = torch.tensor(y[tr_idx], dtype=torch.float32, device=DEVICE)
        xv = torch.tensor(x_all[val_idx], device=DEVICE)
        mv = torch.tensor(mask_f_all[val_idx], device=DEVICE)
        xtest = torch.tensor(x_all[test_idx], device=DEVICE)
        mtest = torch.tensor(mask_f_all[test_idx], device=DEVICE)

        curve = []
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(xt))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(xt), 32):
                idx = perm[start:start + 32]
                optimizer.zero_grad()
                logits = model(xt[idx], mt[idx]).squeeze(-1)
                loss = criterion(logits, yt[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_loss = epoch_loss / n_batches

            model.eval()
            with torch.no_grad():
                val_logits = model(xv, mv).squeeze(-1)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                test_logits = model(xtest, mtest).squeeze(-1)
                test_probs = torch.sigmoid(test_logits).cpu().numpy()
            val_auroc = safe_auroc(y[val_idx], val_probs)
            test_auroc = safe_auroc(y_test, test_probs)
            curve.append({"epoch": epoch, "train_loss": train_loss,
                           "val_auroc": val_auroc, "test_auroc": test_auroc})
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"[fold {fold_i}] epoch {epoch:3d} train_loss={train_loss:.4f} "
                      f"val_auroc={val_auroc:.3f} test_auroc={test_auroc:.3f}", flush=True)

        all_curves[f"fold_{fold_i}"] = curve
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    out_path = os.path.join(OUT_DIR, "diagnostic_training_curves.json")
    with open(out_path, "w") as fp:
        json.dump(all_curves, fp, indent=2)
    print(f"\nSaved curves to {out_path}")


CONFIGS = {
    "default":          dict(lr=3e-4, d_model=128, dropout=0.1, weight_decay=1e-4, n_stages=3, epochs=40, patience=8),
    "lr_low":           dict(lr=1e-4, d_model=128, dropout=0.1, weight_decay=1e-4, n_stages=3, epochs=40, patience=8),
    "lr_high":          dict(lr=1e-3, d_model=128, dropout=0.1, weight_decay=1e-4, n_stages=3, epochs=40, patience=8),
    "small_model":      dict(lr=3e-4, d_model=64,  dropout=0.1, weight_decay=1e-4, n_stages=3, epochs=40, patience=8),
    "small_reg":        dict(lr=3e-4, d_model=64,  dropout=0.3, weight_decay=1e-3, n_stages=3, epochs=40, patience=8),
    "high_dropout":     dict(lr=3e-4, d_model=128, dropout=0.3, weight_decay=1e-4, n_stages=3, epochs=40, patience=8),
    "high_wd":          dict(lr=3e-4, d_model=128, dropout=0.1, weight_decay=1e-3, n_stages=3, epochs=40, patience=8),
    "shallow":          dict(lr=3e-4, d_model=128, dropout=0.1, weight_decay=1e-4, n_stages=2, epochs=40, patience=8),
    "more_epochs":      dict(lr=3e-4, d_model=128, dropout=0.1, weight_decay=1e-4, n_stages=3, epochs=100, patience=20),
    "small_reg_long":   dict(lr=3e-4, d_model=64,  dropout=0.3, weight_decay=1e-3, n_stages=3, epochs=100, patience=20),
}


def train_one(x_all, mask_f_all, y, tr_idx, val_idx, test_idx, cfg):
    set_seed(DIAG_SEED)
    model = CTGCrossformer(
        seq_len=x_all.shape[1], d_model=cfg["d_model"], dropout=cfg["dropout"],
        n_stages=cfg["n_stages"],
    ).to(DEVICE)
    pos_weight = torch.tensor([compute_pos_weight(y[tr_idx])], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    xt = torch.tensor(x_all[tr_idx], device=DEVICE)
    mt = torch.tensor(mask_f_all[tr_idx], device=DEVICE)
    yt = torch.tensor(y[tr_idx], dtype=torch.float32, device=DEVICE)
    xv = torch.tensor(x_all[val_idx], device=DEVICE)
    mv = torch.tensor(mask_f_all[val_idx], device=DEVICE)

    best_val_auroc = -1.0
    best_state = None
    epochs_no_improve = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(xt))
        for start in range(0, len(xt), 32):
            idx = perm[start:start + 32]
            optimizer.zero_grad()
            logits = model(xt[idx], mt[idx]).squeeze(-1)
            loss = criterion(logits, yt[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(xv, mv).squeeze(-1)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_auroc = safe_auroc(y[val_idx], val_probs)
        if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= cfg["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    xtest = torch.tensor(x_all[test_idx], device=DEVICE)
    mtest = torch.tensor(mask_f_all[test_idx], device=DEVICE)
    with torch.no_grad():
        test_logits = model(xtest, mtest).squeeze(-1)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
    test_auroc = safe_auroc(y[test_idx], test_probs)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return test_auroc


def run_sweep():
    fhr, uc, mask, y, groups = load_data()
    folds = diagnostic_folds(fhr, uc, mask, y, groups)

    results = {}
    t0 = time.time()
    for name, cfg in CONFIGS.items():
        fold_aurocs = []
        for f in folds:
            auc = train_one(f["x_all"], f["mask_f_all"], y, f["tr_idx"], f["val_idx"], f["test_idx"], cfg)
            fold_aurocs.append(auc)
        mean_auc = float(np.nanmean(fold_aurocs))
        results[name] = {"cfg": cfg, "fold_aurocs": fold_aurocs, "mean_auroc": mean_auc}
        elapsed = time.time() - t0
        print(f"[{name:15s}] fold_aurocs={[round(a,3) for a in fold_aurocs]} "
              f"mean={mean_auc:.4f} | elapsed={elapsed:.0f}s", flush=True)

    out_path = os.path.join(OUT_DIR, "diagnostic_sweep_results.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nSaved sweep results to {out_path}")

    print("\n=== SWEEP SUMMARY (seed=0, 5 folds, mean test AUROC) ===")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["mean_auroc"]):
        print(f"{name:15s} mean={r['mean_auroc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    if args.curves:
        run_curves()
    if args.sweep:
        run_sweep()
    if not args.curves and not args.sweep:
        print("Pass --curves and/or --sweep")
