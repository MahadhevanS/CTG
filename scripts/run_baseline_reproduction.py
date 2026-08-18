"""
Phase 0 baseline reproduction (PROTOCOL.md section 4): trains the true
CTG-Crossformer plus the three trivial baselines (logistic regression on
classical features, 1D-ResNet on raw FHR+UC, gradient-boosted trees on
classical features) under the frozen protocol's 5-fold CV x N-seed
evaluation, and checks the reproduction gate against published 0.822+-0.03.

No inner-loop hyperparameter search here -- Phase 0 uses fixed, documented
default hyperparameters (nothing to leak/overfit via inner CV yet; that
machinery is exercised starting Phase 3's ablations). A small held-out
slice of each outer-train fold is used only for early stopping.

Usage:
    python scripts/run_baseline_reproduction.py --n_seeds 10 --epochs 40
    python scripts/run_baseline_reproduction.py --smoke   # 1 seed, 3 epochs, sanity check only
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cv.nested_cv import outer_splits
from preprocessing.scalers import FHRScaler, uc_per_record_minmax
from baselines.classical_features import extract_features_batch
from baselines.resnet1d import ResNet1D
from models.crossformer import CTGCrossformer
from training.train_utils import set_seed, compute_pos_weight, safe_auroc, bootstrap_ci

DATA_PATH = os.path.join(ROOT, "data", "processed", "phase0_dataset.npz")
OUT_DIR = os.path.join(ROOT, "docs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prep_signals(fhr, uc, mask, train_idx):
    """Applies FHR z-norm (fit on train_idx only) and UC per-record min-max
    (fold-independent) to the whole array, returning model-ready (N, T, 2)
    signal + (N, T) mask-as-float, matching PROTOCOL.md section 3."""
    train_fhr_windows = [fhr[i][~mask[i]] for i in train_idx if (~mask[i]).any()]
    scaler = FHRScaler.fit(train_fhr_windows)

    fhr_norm = np.stack([scaler.transform(fhr[i]) for i in range(len(fhr))])
    uc_norm = np.stack([uc_per_record_minmax(uc[i]) for i in range(len(uc))])
    # Zero out padded/missing region post-normalisation so it carries no
    # spurious signal beyond what the mask channel already communicates.
    fhr_norm = np.where(mask, 0.0, fhr_norm)
    uc_norm = np.where(mask, 0.0, uc_norm)

    x = np.stack([fhr_norm, uc_norm], axis=-1).astype(np.float32)  # (N, T, 2)
    mask_f = mask.astype(np.float32)
    return x, mask_f


def train_torch_model(model, x_train, mask_train, y_train, x_val, mask_val, y_val,
                       epochs, lr, patience, is_crossformer: bool):
    model.to(DEVICE)
    pos_weight = torch.tensor([compute_pos_weight(y_train)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    xt = torch.tensor(x_train, device=DEVICE)
    mt = torch.tensor(mask_train, device=DEVICE)
    yt = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    xv = torch.tensor(x_val, device=DEVICE)
    mv = torch.tensor(mask_val, device=DEVICE)

    best_val_auroc = -1.0
    best_state = None
    epochs_no_improve = 0
    batch_size = 32

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for start in range(0, len(xt), batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            if is_crossformer:
                logits = model(xt[idx], mt[idx]).squeeze(-1)
            else:
                logits = model(xt[idx].permute(0, 2, 1)).squeeze(-1)
                # ResNet1D wants (B, C, T) with C=3 (fhr, uc, mask) -- handled by caller stacking mask in.
            loss = criterion(logits, yt[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if is_crossformer:
                val_logits = model(xv, mv).squeeze(-1)
            else:
                val_logits = model(xv.permute(0, 2, 1)).squeeze(-1)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_auroc = safe_auroc(y_val, val_probs)
        if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_torch(model, x, mask, is_crossformer: bool):
    model.eval()
    xt = torch.tensor(x, device=DEVICE)
    mt = torch.tensor(mask, device=DEVICE)
    if is_crossformer:
        logits = model(xt, mt).squeeze(-1)
    else:
        logits = model(xt.permute(0, 2, 1)).squeeze(-1)
    return torch.sigmoid(logits).cpu().numpy()


def run(n_seeds: int, epochs: int, patience: int, base_seed: int = 0):
    d = np.load(DATA_PATH, allow_pickle=True)
    fhr, uc, mask = d["fhr"], d["uc"], d["missingness_mask"]
    y = d["y_primary"]
    groups = d["record_id"]
    n = len(y)

    feats_cache = extract_features_batch(fhr, uc, mask)

    results = {"crossformer": [], "resnet1d": [], "logreg": [], "gbt": []}
    fold_log = []

    t0 = time.time()
    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        set_seed(seed)
        folds = outer_splits(y, groups, n_splits=5, seed=seed)

        for fold_i, fold in enumerate(folds):
            train_idx_full, test_idx = fold.train_idx, fold.test_idx
            y_test = y[test_idx]

            # Hold out a small val slice from the outer-train fold for
            # early stopping only (never touches test_idx).
            if y[train_idx_full].sum() >= 2:
                tr_idx, val_idx = train_test_split(
                    train_idx_full, test_size=0.15, stratify=y[train_idx_full],
                    random_state=seed,
                )
            else:
                tr_idx, val_idx = train_test_split(
                    train_idx_full, test_size=0.15, random_state=seed,
                )

            x_all, mask_f_all = prep_signals(fhr, uc, mask, tr_idx)

            # --- Crossformer ---
            model = CTGCrossformer(seq_len=x_all.shape[1])
            model = train_torch_model(
                model, x_all[tr_idx], mask_f_all[tr_idx], y[tr_idx],
                x_all[val_idx], mask_f_all[val_idx], y[val_idx],
                epochs=epochs, lr=3e-4, patience=patience, is_crossformer=True,
            )
            probs = predict_torch(model, x_all[test_idx], mask_f_all[test_idx], is_crossformer=True)
            auc_cf = safe_auroc(y_test, probs)
            results["crossformer"].append(auc_cf)
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            # --- 1D-ResNet (raw FHR+UC+mask, 3 channels) ---
            x_resnet_all = np.concatenate([x_all, mask_f_all[..., None]], axis=-1)  # (N, T, 3)
            resnet = ResNet1D(in_channels=3)
            resnet = train_torch_model(
                resnet, x_resnet_all[tr_idx], mask_f_all[tr_idx], y[tr_idx],
                x_resnet_all[val_idx], mask_f_all[val_idx], y[val_idx],
                epochs=epochs, lr=1e-3, patience=patience, is_crossformer=False,
            )
            probs_r = predict_torch(resnet, x_resnet_all[test_idx], mask_f_all[test_idx], is_crossformer=False)
            auc_resnet = safe_auroc(y_test, probs_r)
            results["resnet1d"].append(auc_resnet)
            del resnet

            # --- Logistic regression on classical features ---
            fscaler = StandardScaler().fit(feats_cache[tr_idx])
            f_train = fscaler.transform(feats_cache[train_idx_full])
            f_test = fscaler.transform(feats_cache[test_idx])
            lr_clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
            lr_clf.fit(f_train, y[train_idx_full])
            auc_lr = safe_auroc(y_test, lr_clf.predict_proba(f_test)[:, 1])
            results["logreg"].append(auc_lr)

            # --- Gradient-boosted trees on classical features ---
            spw = compute_pos_weight(y[train_idx_full])
            gbt = XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                scale_pos_weight=spw, eval_metric="logloss", verbosity=0,
            )
            gbt.fit(feats_cache[train_idx_full], y[train_idx_full])
            auc_gbt = safe_auroc(y_test, gbt.predict_proba(feats_cache[test_idx])[:, 1])
            results["gbt"].append(auc_gbt)

            elapsed = time.time() - t0
            print(f"[seed {seed} fold {fold_i}] n_test={len(test_idx)} pos={int(y_test.sum())} "
                  f"| crossformer={auc_cf:.3f} resnet={auc_resnet:.3f} logreg={auc_lr:.3f} gbt={auc_gbt:.3f} "
                  f"| elapsed={elapsed:.0f}s", flush=True)
            fold_log.append({
                "seed": seed, "fold": fold_i, "n_test": int(len(test_idx)), "n_pos_test": int(y_test.sum()),
                "crossformer": auc_cf, "resnet1d": auc_resnet, "logreg": auc_lr, "gbt": auc_gbt,
            })

    summary = {}
    for name, vals in results.items():
        mean, lo, hi = bootstrap_ci(np.array(vals))
        summary[name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_folds": len(vals),
                          "n_valid_folds": int(np.sum(~np.isnan(vals)))}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "baseline_reproduction_results.json"), "w") as f:
        json.dump({"summary": summary, "fold_log": fold_log,
                    "config": {"n_seeds": n_seeds, "epochs": epochs, "patience": patience}}, f, indent=2)

    print("\n=== SUMMARY (mean AUROC, bootstrap 95% CI over fold-level scores) ===")
    for name, s in summary.items():
        print(f"{name:12s} mean={s['mean']:.4f}  CI=[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]  "
              f"valid_folds={s['n_valid_folds']}/{s['n_folds']}")

    cf_mean = summary["crossformer"]["mean"]
    gate_lo, gate_hi = 0.822 - 0.03, 0.822 + 0.03
    gate_pass = gate_lo <= cf_mean <= gate_hi
    print(f"\nPhase 0 gate: reproduced Crossformer AUROC must be in [{gate_lo:.3f}, {gate_hi:.3f}] "
          f"(published 0.822 +/- 0.03). Got {cf_mean:.4f}. GATE {'PASSES' if gate_pass else 'FAILS -- write diagnosis in docs/baseline_reproduction_report.md'}.")

    return summary, fold_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="1 seed, 3 epochs, sanity check only")
    args = parser.parse_args()

    if args.smoke:
        run(n_seeds=1, epochs=3, patience=2)
    else:
        run(n_seeds=args.n_seeds, epochs=args.epochs, patience=args.patience)
