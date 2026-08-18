# Known Limitations (living document)

## 1. Positive-label prevalence collapse under the exclusion cascade

**Status: accepted, documented, not remediated.** Decision made 2026-08-18.

The frozen exclusion cascade (`PROTOCOL.md`) produces N=432 eligible
records (within the plan's expected 430-480 range), but only **17 positive
records (pH ≤ 7.05, 3.9% prevalence)**, down from 44/552 (8.0%) in the full
dataset. See `docs/exclusion_cascade_report.md` for the full breakdown.

**Root cause**: a `2000`-series sub-cohort (46 records) has genuine,
verified 20-30 minute contiguous signal dropout in the final 30 minutes
before delivery, and is disproportionately positive-label. This is a real
property of the data, confirmed directly against raw signals, not a
pipeline defect.

**Decision**: keep the cascade exactly as frozen and proceed. Rationale:
changing an exclusion rule after observing that it removes inconvenient
positives is exactly the protocol-drift failure mode the source plan
explicitly warns against (see plan's closing section, "ON THE 0.86
TARGET"). The alternative — investigating and potentially relaxing Step 2's
threshold — was explicitly declined in favor of documenting and moving
forward.

**Downstream consequences that follow from this, to be honored later
rather than silently worked around:**

1. **Phase 5.2 power analysis must be recomputed for N_pos=17, not the
   plan's assumed ~40.** The plan states "with N≈450 and ~40 positives, the
   minimum detectable ΔAUROC at 80% power will likely land around
   0.04-0.05." With fewer than half the assumed positives, the true
   minimum detectable effect will be meaningfully larger. This needs to be
   computed (not assumed) once Phase 5 is reached, and reported plainly.
2. **5x5 nested CV outer test folds will have ~3-4 positives each.**
   Per-fold AUROC will be high-variance by construction; this is a sample-
   size property, not a sign of a bad model. Report the full per-fold
   spread, not just the mean, everywhere.
3. **The secondary label (BDecf > 12, 7 positives in the eligible set) is
   even thinner** and should be treated as a qualitative sensitivity check
   only, exactly as the plan already intended ("sensitivity analysis only,
   reported once at the end") — never as a target for model selection.
4. Any future comparison to the old `CTG-Fetal-Distress-Prediction` repo's
   numbers (pH ≤ 7.15, 113 positives) must note the ~2.5x difference in
   positive count as a primary reason results are not comparable, separate
   from the CV-scheme and window differences already documented in
   PROTOCOL.md.

## 2. CV fold assignments are not reproducible across environments

**Status: accepted, partially mitigated, not retroactively fixed.**
Decision made 2026-08-18, see `docs/baseline_reproduction_report.md` §2.4.

`src/cv/nested_cv.py`'s `outer_splits()`/`inner_splits()` use
`StratifiedGroupKFold(shuffle=True, random_state=seed)`. This is only
reproducible *within one fixed scikit-learn version* — `random_state`
does not guarantee bit-identical splits across sklearn versions. Because
`run_baseline_reproduction.py`'s resume logic skips any `(seed, fold)`
already present in the results JSONL, the Phase 0 baseline reproduction
run's 50 stored folds are a mixture of splits computed on at least two
different machines/environments (the run was started on a CPU laptop and
finished on a GPU laptop) — **17 of 50 (34%) do not match what the same
seed produces when recomputed fresh in the final (GPU, pinned sklearn
1.7.2) environment.**

This does not invalidate any reported AUROC number (every fold is still a
real, correctly grouped, non-leaking, independently-trained-and-evaluated
split — leakage is asserted on every fold in every environment). It means
"seed=k" cannot be treated as a portable, reproducible fold identifier
across machines or scikit-learn versions, including retroactively for
Phase 0's own results.

**Mitigation so far**: `requirements.txt` now pins `scikit-learn==1.7.2`
exactly, so this stops getting worse from this point forward, and any
single-environment run (including all of Phase 1+, as long as the pin
holds) will be internally reproducible.

**Not yet done**: the existing Phase 0 fold assignments were not
regenerated to be bit-exact within the pinned environment (would require
one more full 50-fold re-run and does not change Phase 0's conclusions).
If bit-exact reproducibility of the *exact* Phase 0 fold membership is
ever required (e.g. supplementary material for publication), either
re-run that job once now under the pin, or — better long-term — freeze
the actual fold assignments (record IDs per fold) to a committed file
once, so future phases don't depend on `StratifiedGroupKFold` regenerating
the same thing across time/environments at all.
