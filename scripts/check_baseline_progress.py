"""
Reads data/processed/baseline_reproduction_fold_results.jsonl (appended
incrementally by run_baseline_reproduction.py) and prints current progress
+ a partial summary/gate-check, whether the run is still in flight, was
killed mid-way, or finished. Safe to run at any time.

Usage:
    python scripts/check_baseline_progress.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from training.train_utils import bootstrap_ci

RESULTS_JSONL = os.path.join(ROOT, "data", "processed", "baseline_reproduction_fold_results.jsonl")


def main(n_seeds_target: int = 10, n_outer: int = 5):
    if not os.path.exists(RESULTS_JSONL):
        print(f"No results yet -- {RESULTS_JSONL} does not exist.")
        return

    rows = []
    with open(RESULTS_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    target = n_seeds_target * n_outer
    print(f"Progress: {len(rows)} / {target} folds complete "
          f"({100*len(rows)/target:.1f}%).")
    if not rows:
        return

    seeds_seen = sorted(set(r["seed"] for r in rows))
    print(f"Seeds with >=1 completed fold: {seeds_seen}")

    for name in ["crossformer", "resnet1d", "logreg", "gbt"]:
        vals = np.array([r[name] for r in rows])
        mean, lo, hi = bootstrap_ci(vals)
        n_valid = int(np.sum(~np.isnan(vals)))
        print(f"  {name:12s} mean={mean:.4f}  CI=[{lo:.4f}, {hi:.4f}]  "
              f"(n={len(vals)}, valid={n_valid}) [PARTIAL -- not all folds in yet]"
              if len(rows) < target else
              f"  {name:12s} mean={mean:.4f}  CI=[{lo:.4f}, {hi:.4f}]  (n={len(vals)}, valid={n_valid})")

    if len(rows) == target:
        cf_vals = np.array([r["crossformer"] for r in rows])
        cf_mean = bootstrap_ci(cf_vals)[0]
        gate_lo, gate_hi = 0.822 - 0.03, 0.822 + 0.03
        gate_pass = gate_lo <= cf_mean <= gate_hi
        print(f"\nALL FOLDS COMPLETE. Phase 0 gate: [{gate_lo:.3f}, {gate_hi:.3f}], "
              f"got {cf_mean:.4f} -> {'PASSES' if gate_pass else 'FAILS'}.")
    else:
        print(f"\nRun not yet complete ({target - len(rows)} folds remaining). "
              f"Re-run `python scripts/run_baseline_reproduction.py --n_seeds {n_seeds_target}` "
              f"to resume -- it will skip these {len(rows)} completed folds automatically.")


if __name__ == "__main__":
    main()
