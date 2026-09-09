"""
DeepBudget-Vis v2 - shared pipeline module.

This module contains ONLY code that was verified against the actual supplied
files (train.csv / val.csv / test.csv / val_with_context.csv /
test_with_context.csv, and Prop_model_v1 (1).ipynb) in this repository,
branch Suren-1.

Verified facts baked into this module (see LEAKAGE_AUDIT.md / final report for
the full verification transcript):
  - train.csv:  2557 rows, 2016-01-01 .. 2022-12-31  (84 whole months)
  - val.csv:     547 rows, 2023-01-01 .. 2024-06-30  (18 whole months)
  - test.csv:    549 rows, 2024-07-01 .. 2025-12-31  (18 whole months)
  - val_with_context.csv / test_with_context.csv are EXACTLY val.csv/test.csv
    plus 30 preceding rows copied verbatim from the previous split (train for
    val's context, val for test's context). Verified byte-for-byte equal on
    all scaled and unscaled columns for the overlapping dates.
  - All "<col>_scaled" feature columns are plain z-scores: (x - mean_train) /
    std_train(ddof=0), with mean/std computed ONLY on train.csv. Verified for
    all 39 raw feature columns, max reconstruction error < 2e-15.
  - Both targets (NET_PATIENT_REVENUE, TOTAL_OPERATING_EXP) are scaled as
    z-score of log1p(x), with mean/std of log1p(x) computed ONLY on
    train.csv. Verified max reconstruction error < 3e-14 on train, val, test.
  - No duplicate dates in any file. No gaps checked beyond min/max (daily
    data, see EDA note in final report for day-count verification).
"""
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import explained_variance_score, median_absolute_error, max_error

# ------------------------------------------------------------------
# Paths (repo-relative; this file lives in Advertisement_effect/deepbudgetvis_v2/)
# ------------------------------------------------------------------
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(REPO_DIR, 'train.csv')
VAL_CSV = os.path.join(REPO_DIR, 'val.csv')
TEST_CSV = os.path.join(REPO_DIR, 'test.csv')
VAL_CTX_CSV = os.path.join(REPO_DIR, 'val_with_context.csv')
TEST_CTX_CSV = os.path.join(REPO_DIR, 'test_with_context.csv')

SEED = 42

# ------------------------------------------------------------------
# Stream assignment - IDENTICAL to Prop_model_v1 (1).ipynb Cell 3.
# Verified these 39 names are exactly the set of non-scaled, non-target,
# non-id, non-label columns in train.csv (see audit above).
# ------------------------------------------------------------------
DEMAND_COLS = ['OCCUPANCY_RATE', 'STAFFED_BEDS', 'ADMISSIONS', 'ER_VISITS', 'OP_VISITS', 'SURGERIES',
               'DISCHARGES', 'AVG_LENGTH_OF_STAY', 'YEAR', 'IS_WEEKEND', 'QUARTER', 'IS_HOLIDAY',
               'DOW_SIN', 'DOW_COS', 'MONTH_SIN', 'MONTH_COS']
REVCYCLE_COLS = ['GROSS_CHARGES_MEDICARE', 'GROSS_CHARGES_MEDICAID', 'GROSS_CHARGES_COMMERCIAL',
                  'GROSS_CHARGES_SELFPAY', 'GROSS_CHARGES_OTHER', 'TOTAL_GROSS_CHARGES', 'CHARITY_CARE',
                  'BAD_DEBT', 'CLAIMS_SUBMITTED', 'DENIAL_RATE', 'CLAIMS_DENIED', 'CASH_COLLECTED']
EXPENSE_COLS = ['LABOR_EXP', 'SUPPLY_EXP', 'OVERHEAD_EXP', 'CAPITAL_EXP', 'OTHER_OPERATING_EXP',
                 'BUDGETED_LABOR_EXP', 'BUDGETED_SUPPLY_EXP', 'BUDGETED_PHARMACY_EXP',
                 'BUDGETED_OVERHEAD_EXP', 'BUDGET_VARIANCE', 'OPERATING_MARGIN_PCT']

ALL_FEATURE_COLS = DEMAND_COLS + REVCYCLE_COLS + EXPENSE_COLS
assert len(ALL_FEATURE_COLS) == 39, "Feature stream count changed unexpectedly"

DEMAND_COLS_S = [f'{c}_scaled' for c in DEMAND_COLS]
REVCYCLE_COLS_S = [f'{c}_scaled' for c in REVCYCLE_COLS]
EXPENSE_COLS_S = [f'{c}_scaled' for c in EXPENSE_COLS]

TARGET_COLS = ['NET_PATIENT_REVENUE', 'TOTAL_OPERATING_EXP']
TARGET_COLS_S = [f'{c}_scaled' for c in TARGET_COLS]

# Revenue-cycle candidates explicitly evaluated for reintroduction (Upgrade 7).
# AR_BALANCE, DAYS_IN_AR, CLAIMS_PAID and a distinct CONTRACTUAL_ADJUSTMENTS
# column DO NOT EXIST in the supplied train.csv/val.csv/test.csv (verified by
# column-name search - see final report, "Upgrade 7" section). CASH_COLLECTED
# already exists and is already included in REVCYCLE_COLS above. No new
# columns were introduced.


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_splits(use_context_for_val_test=True):
    """Load train/val/test.

    If use_context_for_val_test=True, val/test are loaded from the
    *_with_context.csv files (which contain 30 extra leading rows copied
    verbatim from the preceding split) so that sequence windows for the
    first val/test targets can look back across the split boundary without
    any leakage (those context rows were already seen during the preceding
    split and are not future information).
    """
    train_df = pd.read_csv(TRAIN_CSV, parse_dates=['DATE']).sort_values('DATE').reset_index(drop=True)
    if use_context_for_val_test:
        val_df = pd.read_csv(VAL_CTX_CSV, parse_dates=['DATE']).sort_values('DATE').reset_index(drop=True)
        test_df = pd.read_csv(TEST_CTX_CSV, parse_dates=['DATE']).sort_values('DATE').reset_index(drop=True)
    else:
        val_df = pd.read_csv(VAL_CSV, parse_dates=['DATE']).sort_values('DATE').reset_index(drop=True)
        test_df = pd.read_csv(TEST_CSV, parse_dates=['DATE']).sort_values('DATE').reset_index(drop=True)
    return train_df, val_df, test_df


def load_val_test_no_context():
    val_df = pd.read_csv(VAL_CSV, parse_dates=['DATE']).sort_values('DATE').reset_index(drop=True)
    test_df = pd.read_csv(TEST_CSV, parse_dates=['DATE']).sort_values('DATE').reset_index(drop=True)
    return val_df, test_df


# Verified constant: val_with_context.csv / test_with_context.csv each prepend
# exactly 30 rows of context copied verbatim from the preceding split.
CONTEXT_ROWS = 30


def inverse_targets(y_scaled, target_col_stats):
    """target_col_stats: dict target_col -> (mu_log, sd_log), fit on TRAIN ONLY."""
    out = np.zeros_like(y_scaled)
    for i, col in enumerate(TARGET_COLS):
        mu, sd = target_col_stats[col]
        out[:, i] = np.expm1(y_scaled[:, i] * sd + mu)
    return out


def compute_target_scaler_stats(train_df):
    """Recompute the log1p+zscore stats from train.csv directly (does not
    rely on any joblib scaler file, since none was supplied with this repo
    checkout - the _scaled columns in the CSVs already encode the fitted
    transform, and we verified train-only fitting numerically, see module
    docstring)."""
    stats = {}
    for col in TARGET_COLS:
        logt = np.log1p(train_df[col])
        stats[col] = (float(logt.mean()), float(logt.std(ddof=0)))
    return stats


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class MultiStreamDataset(Dataset):
    """Returns (xd, xr, xe, xh, y, is_anomaly) per window.

    xh (target-history stream) is included only if `include_target_history`
    is True; it holds the SAME seq_len window of past scaled target values
    (t-seq_len .. t-1), which are strictly historical relative to the
    prediction target at time t (see leakage audit - Upgrade 1).

    n_context: number of leading rows in `df` that belong to the PRECEDING
    split (e.g. the 30 context rows prepended to val_with_context.csv /
    test_with_context.csv). Pass n_context=0 for train.csv (no context).

    IMPORTANT (leakage/contamination fix): valid prediction TARGETS are
    restricted to indices >= n_context, i.e. dates that actually belong to
    the official split being represented by `df`. Without this restriction,
    for seq_len < n_context the dataset would silently produce "validation"
    or "test" targets whose dates fall inside the context window - and since
    those context dates are copied verbatim from the PRECEDING split (which
    was itself used to fit the model, e.g. context dates for val are literal
    tail rows of train.csv that were already prediction targets during
    training), evaluating on them would contaminate validation/test metrics
    with disguised training performance. This was verified empirically: for
    n_context=30, seq_len=7 produced 23 such contaminated targets before this
    fix. Restricting targets to idx >= n_context removes them. For
    seq_len > n_context, targets whose window would need dates further back
    than the provided context are also unavailable and are correctly
    dropped (not fabricated).
    """
    def __init__(self, df, seq_len, include_target_history=False, n_context=0):
        self.Xd = df[DEMAND_COLS_S].values.astype(np.float32)
        self.Xr = df[REVCYCLE_COLS_S].values.astype(np.float32)
        self.Xe = df[EXPENSE_COLS_S].values.astype(np.float32)
        self.Xh = df[TARGET_COLS_S].values.astype(np.float32)  # for history stream
        self.y = df[TARGET_COLS_S].values.astype(np.float32)
        self.is_anomaly = df['IS_ANOMALY'].values.astype(np.float32)
        self.seq_len = seq_len
        self.include_target_history = include_target_history
        self.n_context = n_context

        n = len(self.Xd)
        first_target = max(n_context, seq_len)
        # valid target indices: first_target .. n-1
        self.target_indices = list(range(first_target, n))

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, i):
        target_idx = self.target_indices[i]
        s = slice(target_idx - self.seq_len, target_idx)
        xh = self.Xh[s] if self.include_target_history else np.zeros((self.seq_len, len(TARGET_COLS)), dtype=np.float32)
        return (self.Xd[s], self.Xr[s], self.Xe[s], xh,
                self.y[target_idx], self.is_anomaly[target_idx])


# ------------------------------------------------------------------
# Metrics (identical formula set to Prop_model_v1 (1).ipynb Cell 7, applied
# per-target here since Upgrade 5 requires target-specific reporting)
# ------------------------------------------------------------------
def compute_metrics_1d(y_true, y_pred, y_true_scaled=None, y_pred_scaled=None):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else float('nan')
    evs = explained_variance_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    maxerr = max_error(y_true, y_pred)
    out = {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape), 'SMAPE': float(smape),
           'R2': float(r2), 'ExplainedVar': float(evs), 'MedianAE': float(medae), 'MaxError': float(maxerr)}
    if y_true_scaled is not None:
        y_true_scaled = np.asarray(y_true_scaled).flatten()
        y_pred_scaled = np.asarray(y_pred_scaled).flatten()
        out['MAE_scaled'] = float(np.mean(np.abs(y_true_scaled - y_pred_scaled)))
        out['RMSE_scaled'] = float(np.sqrt(np.mean((y_true_scaled - y_pred_scaled) ** 2)))
    else:
        out['MAE_scaled'] = float('nan')
        out['RMSE_scaled'] = float('nan')
    return out


def compute_metrics_per_target(y_true, y_pred, y_true_scaled, y_pred_scaled):
    """y_true/y_pred shape (N, 2) in original financial units;
    y_true_scaled/y_pred_scaled shape (N, 2) in training space.
    Returns dict: {target_name: {metric: value}}"""
    out = {}
    for i, col in enumerate(TARGET_COLS):
        out[col] = compute_metrics_1d(y_true[:, i], y_pred[:, i], y_true_scaled[:, i], y_pred_scaled[:, i])
    return out
