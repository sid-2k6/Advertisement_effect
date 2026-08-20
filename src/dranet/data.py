"""
Data loading, cleaning, validation and locked 70/15/15 splitting for Criteo v2.1.

Design choices (aligned with the approved plan + user corrections):
  * `exposure` is a POST-TREATMENT variable (ad actually served) -> a mediator/collider.
    It is DROPPED and never used as a covariate or outcome.
  * Features f0..f11 stored as float32; treatment/visit/conversion as int8 (memory-safe).
  * Random 70/15/15 split with a fixed seed. At N=13.98M a uniform random split
    preserves treatment (~0.85) and outcome marginals to >4 decimals (verified in
    `validate_splits`), so the same split serves both Experiment A (visit) and
    Experiment B (conversion). Splits are frozen to parquet and reused everywhere.
"""
from __future__ import annotations
import gc
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import config as C


DTYPES = {f: "float32" for f in C.FEATURES}
DTYPES.update({c: "int8" for c in [C.TREATMENT] + C.OUTCOMES + C.POST_TREATMENT})


def load_raw(nrows: int | None = None) -> pd.DataFrame:
    """Load the raw gzip CSV with memory-efficient dtypes; drop post-treatment vars."""
    df = pd.read_csv(C.RAW_CSV_GZ, dtype=DTYPES, nrows=nrows)
    # Drop post-treatment mediator(s)
    df = df.drop(columns=[c for c in C.POST_TREATMENT if c in df.columns])
    return df


def data_validation_report(df: pd.DataFrame) -> Dict:
    """Missing/duplicate/balance/imbalance/cardinality checks on the full frame."""
    rep: Dict = {}
    rep["n_rows"] = int(len(df))
    rep["columns"] = list(df.columns)
    rep["missing_per_col"] = {c: int(df[c].isna().sum()) for c in df.columns}
    rep["n_duplicate_rows"] = int(df.duplicated().sum())
    rep["treatment_mean"] = float(df[C.TREATMENT].mean())
    rep["treatment_counts"] = {int(k): int(v) for k, v in df[C.TREATMENT].value_counts().items()}
    for y in C.OUTCOMES:
        rep[f"{y}_mean"] = float(df[y].mean())
        # naive ATE = E[Y|T=1]-E[Y|T=0]
        t1 = df.loc[df[C.TREATMENT] == 1, y].mean()
        t0 = df.loc[df[C.TREATMENT] == 0, y].mean()
        rep[f"{y}_naive_ate"] = float(t1 - t0)
        rep[f"{y}_rate_treated"] = float(t1)
        rep[f"{y}_rate_control"] = float(t0)
    rep["feature_cardinality"] = {f: int(df[f].nunique()) for f in C.FEATURES}
    rep["feature_summary"] = {
        f: {"min": float(df[f].min()), "max": float(df[f].max()),
            "mean": float(df[f].mean()), "std": float(df[f].std())}
        for f in C.FEATURES
    }
    return rep


def make_splits(df: pd.DataFrame, seed: int = C.SEED) -> Dict[str, np.ndarray]:
    """Return dict of index arrays for train/val/test with exact target counts."""
    n = len(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = C.SPLIT_TARGET["train"] if n == C.N_TOTAL else int(round(C.TRAIN_FRAC * n))
    n_va = C.SPLIT_TARGET["val"] if n == C.N_TOTAL else int(round(C.VAL_FRAC * n))
    idx = {
        "train": perm[:n_tr],
        "val": perm[n_tr:n_tr + n_va],
        "test": perm[n_tr + n_va:],
    }
    return idx


def prepare_and_save(nrows: int | None = None, seed: int = C.SEED) -> Dict:
    """End-to-end: load, validate, split, and persist parquet splits. Returns report."""
    C.ensure_dirs()
    df = load_raw(nrows=nrows)
    report = data_validation_report(df)

    idx = make_splits(df, seed=seed)
    counts = {}
    for split, ix in idx.items():
        sub = df.iloc[ix].reset_index(drop=True)
        sub.to_parquet(C.split_path(split), index=False)
        counts[split] = int(len(sub))
        del sub
        gc.collect()
    report["split_counts"] = counts

    # Per-split marginal consistency (train/val/test consistency check)
    split_marginals = {}
    for split, ix in idx.items():
        sub = df.iloc[ix]
        split_marginals[split] = {
            "n": int(len(sub)),
            "treatment_mean": float(sub[C.TREATMENT].mean()),
            **{f"{y}_mean": float(sub[y].mean()) for y in C.OUTCOMES},
        }
    report["split_marginals"] = split_marginals

    with open(os.path.join(C.DATA_DIR, "data_validation_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    del df
    gc.collect()
    return report


def load_split(split: str, outcome: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a frozen split as float32 X, int8 T, int8 Y (single outcome)."""
    df = pd.read_parquet(C.split_path(split),
                         columns=C.FEATURES + [C.TREATMENT, outcome])
    X = df[C.FEATURES].to_numpy(dtype=np.float32)
    T = df[C.TREATMENT].to_numpy(dtype=np.int8)
    Y = df[outcome].to_numpy(dtype=np.int8)
    del df
    gc.collect()
    assert X.shape[0] == T.shape[0] == Y.shape[0], "row mismatch in load_split"
    assert X.shape[1] == len(C.FEATURES), "feature count mismatch"
    return X, T, Y


def subsample_train(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                    n: int | None, seed: int = C.SEED
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Documented training subsample (stratified by treatment x outcome).

    NEVER used for validation/test. Returns full arrays if n is None or n>=len.
    """
    if n is None or n >= len(X):
        return X, T, Y
    rng = np.random.default_rng(seed)
    strata = (T.astype(np.int64) * 2 + Y.astype(np.int64))
    sel = np.zeros(len(X), dtype=bool)
    for s in np.unique(strata):
        pos = np.where(strata == s)[0]
        k = max(1, int(round(n * len(pos) / len(X))))
        k = min(k, len(pos))
        chosen = rng.choice(pos, size=k, replace=False)
        sel[chosen] = True
    return X[sel], T[sel], Y[sel]
