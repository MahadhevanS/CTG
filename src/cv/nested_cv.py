"""
Grouped, label-stratified 5x5 nested cross-validation (PROTOCOL.md decision
#7), repeated over multiple seeds for the final evaluation protocol (plan
section 5.1: "5x5 nested CV x 10 seeds").

Phase 0 has exactly one window per record, so "grouped by record ID" is
trivially satisfied by record-level splitting -- but this is implemented
with explicit group-awareness (StratifiedGroupKFold) rather than assumed,
so it stays correct if a later phase introduces multiple windows/record.
"""
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class FoldSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray


def _no_group_leakage(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> bool:
    train_groups = set(groups[train_idx].tolist())
    test_groups = set(groups[test_idx].tolist())
    return len(train_groups & test_groups) == 0


def outer_splits(y: np.ndarray, groups: np.ndarray, n_splits: int = 5,
                  seed: int = 42) -> List[FoldSplit]:
    """
    5-fold outer split, stratified by y, grouped by record ID (groups).

    Args:
        y: (N,) primary label array.
        groups: (N,) record-ID array (str or int), one entry per sample.
        n_splits: number of outer folds.
        seed: RNG seed -- vary this across the plan's 10 seeds.
    """
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in skf.split(np.zeros(len(y)), y, groups):
        assert _no_group_leakage(groups, train_idx, test_idx), (
            "Record-level leakage detected between outer train/test -- "
            "this must never happen per PROTOCOL.md decision #7."
        )
        folds.append(FoldSplit(train_idx=train_idx, test_idx=test_idx))
    return folds


def inner_splits(outer_train_idx: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  n_splits: int = 5, seed: int = 42) -> List[FoldSplit]:
    """
    5-fold inner split within one outer-train fold, for hyperparameter/config
    selection ONLY. Indices returned are into the *original* (full-dataset)
    index space, not re-based to 0..len(outer_train_idx), so they can be used
    directly against the same y/groups/feature arrays as the outer split.

    Args:
        outer_train_idx: indices (into the full dataset) belonging to this
                          outer fold's training set.
    """
    y_sub = y[outer_train_idx]
    groups_sub = groups[outer_train_idx]

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_local, va_local in skf.split(np.zeros(len(y_sub)), y_sub, groups_sub):
        train_idx = outer_train_idx[tr_local]
        val_idx = outer_train_idx[va_local]
        assert _no_group_leakage(groups, train_idx, val_idx), (
            "Record-level leakage detected between inner train/val."
        )
        folds.append(FoldSplit(train_idx=train_idx, test_idx=val_idx))
    return folds


def repeated_nested_cv(y: np.ndarray, groups: np.ndarray, n_seeds: int = 10,
                        n_outer: int = 5, n_inner: int = 5,
                        base_seed: int = 0) -> Iterator[Tuple[int, int, FoldSplit, List[FoldSplit]]]:
    """
    Full plan-spec iterator: for each of n_seeds seeds, for each of the
    n_outer outer folds, yields (seed, outer_fold_idx, outer_fold, inner_folds).

    Usage:
        for seed, outer_i, outer, inners in repeated_nested_cv(y, groups):
            # select hyperparams using `inners` (never touching outer.test_idx)
            # then train final model on outer.train_idx, evaluate on outer.test_idx
            ...
    """
    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        outers = outer_splits(y, groups, n_splits=n_outer, seed=seed)
        for outer_i, outer in enumerate(outers):
            inners = inner_splits(outer.train_idx, y, groups, n_splits=n_inner, seed=seed)
            yield seed, outer_i, outer, inners
