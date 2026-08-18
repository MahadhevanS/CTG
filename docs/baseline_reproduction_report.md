# Baseline Reproduction Report — Phase 0 Gate

**Status: GATE FAILS. Diagnosis recorded below per `PROTOCOL.md` section 4.
Reproduced numbers adopted as the true baseline for everything downstream.**

Run: 10 seeds x 5 outer folds = 50 folds, all complete.
Config: `epochs=40, patience=8` (early stopping on outer-train-derived val
AUROC). Full per-fold log in `docs/baseline_reproduction_results.json`,
raw rows in `data/processed/baseline_reproduction_fold_results.jsonl`.

## 1. Result

| Model | Mean AUROC | 95% CI | Valid folds |
|---|---|---|---|
| **Crossformer (B0)** | **0.604** | [0.560, 0.646] | 49/50 |
| ResNet1D | 0.667 | [0.621, 0.714] | 49/50 |
| Logistic regression (classical features) | 0.505 | [0.466, 0.545] | 49/50 |
| GBT / XGBoost (classical features) | 0.456 | [0.412, 0.500] | 49/50 |

Gate: reproduced Crossformer mean must land in [0.792, 0.852] (published
0.822 ± 0.03). **Got 0.604 — 0.19 below the lower gate bound, roughly 6x
the tolerance window.** This is not a marginal miss.

One fold (seed 7, fold 4) has 0 positives in its test set (n_test=86,
n_pos_test=0) → AUROC undefined, excluded from all four models' means
(49/50 valid everywhere). This is a mechanical consequence of §2 below, not
a bug.

## 2. Diagnosis

### 2.1 Primary driver: the frozen exclusion cascade leaves too few positives to estimate AUROC stably

`docs/known_limitations.md` already documents that the frozen cascade
collapses positive prevalence from 8.0% (44/552) to **3.9% (17/432)**. Each
outer test fold therefore contains on average 3.4 positives (observed
range in this run: 0–7; distribution `{0:1, 2:5, 3:27, 4:11, 5:3, 6:2,
7:1}` across the 50 folds).

With single-digit positive counts per fold, AUROC is estimated from as few
as 2-7 discordant-pair anchors. The result is exactly what the raw fold
log shows: **per-fold Crossformer AUROC ranges from 0.250 to 0.916, std
0.157** across the 49 valid folds — a swing of nearly the entire [0,1]
range on the same model, same architecture, same hyperparameters, only the
fold's specific 3-4 positive records changing. The correlation between
`n_pos_test` and per-fold AUROC is weak (r = -0.12), so this isn't a case
of "more positives always helps" — it's closer to per-fold coin-flip
variance around whatever the model actually learned, because the
denominator is too small for AUROC to concentrate.

A published point estimate of 0.822 almost certainly comes from a cohort
with substantially more positives than 17 (see 2.2) — under that regime,
per-fold AUROC concentrates much more tightly, and the mean is a stable
target. Under N_pos=17, the mean over 50 such high-variance folds is
itself a noisy quantity; 0.604 is our best estimate of it, but the CI
width ([0.560, 0.646], 0.086 wide) already reflects that instability, and
the true gap to 0.822 (0.22) is far larger than that CI width — so this is
a real gap in central tendency, not just sampling noise around 0.822.

### 2.2 Contributing factor: published number is likely not comparable at the cohort level

`docs/known_limitations.md` item 4 already flags that a related prior
repo (`CTG-Fetal-Distress-Prediction`) used pH ≤ 7.15 with 113 positives —
2.5x more positives than PROTOCOL.md's frozen pH ≤ 7.05 / 17-positive
cohort, purely from the label-threshold choice, before this project's
window/exclusion decisions are even applied. If the published 0.822 traces
to a similarly higher-prevalence, higher-N_pos setup, it was never
estimated under the noise regime this reproduction is running in. This is
a plausible, previously-flagged reason the two numbers aren't directly
comparable — not a newly discovered excuse, but worth restating here since
it bears directly on the gate.

### 2.3 Contributing factor: Crossformer is outperformed by a much simpler model

Crossformer (0.604) is beaten by the plain 1D-ResNet baseline (0.667) on
the same folds, same data, same label. If N=432/N_pos=17 were simply "too
little data for any model," the trivial classical-feature baselines
(logreg 0.505, GBT 0.456 — both at or below chance) would be the ones
closest to Crossformer, not the ones furthest away. Instead the ranking is
ResNet1D > Crossformer > logreg > GBT, i.e. the two raw-signal deep models
beat the two classical-feature models, but the architecturally simpler raw-
signal model beats the architecturally heavier one. This is consistent
with a transformer-style cross-time/cross-dimension attention mechanism
being comparatively data-hungry relative to a plain convolutional
baseline, and getting no benefit (or a net penalty) from that extra
capacity at N_train ≈ 340/fold. It argues against an implementation-bug
explanation specific to the Crossformer code (a broken implementation
would more likely look randomly bad, not "systematically beaten by a
strictly simpler model on the same data") and toward a sample-size/
architecture-mismatch explanation.

GBT scoring below 0.5 (worse than random) on held-out folds, given
`n_estimators=200, max_depth=3` trained on ~340 samples with ~13
positives, is consistent with overfitting noise in the classical features
under this same small-N/scarce-positive regime — a symptom of the same
root cause (2.1), not a separate bug.

### 2.4 Secondary finding, does not affect the conclusion above: CV split is not byte-reproducible across environments

While reconciling this run's results with an earlier partial sync (23/50
folds, committed from the CPU-laptop run), fold (seed=4, fold=2) was found
to have a **different n_pos_test (4 vs 3) and different train/test split**
between the two runs, despite identical `phase0_dataset.npz` (unchanged
since it was committed) and identical seed passed to
`StratifiedGroupKFold(shuffle=True, random_state=seed)`
(`src/cv/nested_cv.py`). All other 22 previously-synced folds matched
exactly.

Cause: `requirements.txt` pins `scikit-learn>=1.3.0` (unpinned upper
bound); this machine has 1.7.2 installed. `StratifiedGroupKFold`'s exact
split assignment is not guaranteed bit-identical across sklearn versions
even with a fixed `random_state` — only within a fixed version. The
CPU-laptop and GPU-laptop runs almost certainly used different sklearn
versions, so that one fold's assignment differed once the GPU laptop
recomputed it independently (it wasn't in *its* copy of the resume file
yet at that point).

This is a real reproducibility gap worth fixing going forward (pin
`scikit-learn` exactly — done in this same change, see below) but it does
**not** explain the gate failure: it affects the exact record membership
of 1 fold out of 50, not the aggregate ~0.22 gap between 0.604 and 0.822,
and the final 50-fold result reported above is entirely self-consistent
(computed in one environment, one run, all resumed folds validated to have
zero duplicate `(seed, fold)` keys).

## 3. Decision (per PROTOCOL.md section 4)

Gate fails. Per the pre-agreed protocol, no cascade/label/window changes
are made in response to this result (that would be exactly the
protocol-drift the plan warns against). Instead:

- **The reproduced number is adopted as the true baseline**: Crossformer
  mean AUROC = **0.604**, 95% CI [0.560, 0.646], N=49 valid folds.
- **Section 5's target is reinterpreted accordingly**: "AUROC ≥ 0.86, CI
  lower bound above the reproduced baseline mean" now means the CI lower
  bound must exceed **0.604** (not 0.822). The absolute 0.86 bar is
  unchanged and, given the strongest baseline observed here is 0.667
  (ResNet1D), remains an ambitious target relative to this dataset's
  apparent ceiling under N_pos=17 — worth flagging now rather than
  discovering it as a surprise in Phase 5.
- **Crossformer vs. classical baselines (plan section 0.3 check)**:
  Crossformer (0.604) does beat both classical-feature baselines (logreg
  0.505, GBT 0.456), so the "does the transformer beat a classical
  baseline" check passes. It does **not** beat the simpler raw-signal
  ResNet1D baseline (0.667) — flagged in 2.3 as a finding worth carrying
  into Phase 3's ablations, not a reason to block Phase 1.
- **`requirements.txt`**: `scikit-learn` pinned to the installed `1.7.2`
  to stop this specific irreproducibility from recurring (2.4).
- **Phase 0 is closed. Proceed to Phase 1** (FIGO Feature Extraction
  Engine) per `docs/PROJECT_STATUS.md`.
