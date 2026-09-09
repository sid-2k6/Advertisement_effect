"""
Builds DeepBudgetVis_Proposed_Final_v2.ipynb programmatically with nbformat.

Every code cell here is the ACTUAL code that was run to produce the results
reported in this project (taken from pipeline.py / models.py / train_utils.py
/ run_*.py in this directory). Every output embedded below is the ACTUAL
captured stdout from running that code in this sandbox during this session
(copied from experiments/*.log / *.json) - nothing is fabricated.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(source, outputs_text=None):
    c = nbf.v4.new_code_cell(source)
    if outputs_text:
        c['outputs'] = [nbf.v4.new_output(output_type='stream', name='stdout', text=outputs_text)]
    cells.append(c)


# ============================================================
# 0. Title / overview
# ============================================================
md("""# DeepBudget-Vis Proposed Model — Final v2

Trains and evaluates the upgraded DeepBudget-Vis forecasting model for
`NET_PATIENT_REVENUE` and `TOTAL_OPERATING_EXP`, under a strict **84/18/18
month** chronological split with a leakage-safe protocol.

All decisions below (window length, loss function, hyperparameters) were
made using **only** `train.csv` and `val.csv`. The 18-month `test.csv` is
evaluated **exactly once**, at the very end, after every decision was frozen.

`Prop_model_v1 (1).ipynb` (the original proposed model) is **not modified**.
Its architecture is faithfully reimplemented here and retrained on the same
84/18/18 split, purely as a fair baseline comparison (Section 10).

**Honesty note (read first):** the upgraded model improves over the
retrained-v1 baseline on the VALIDATION set, but does **not** improve over it
on the held-out TEST set (Section 11). This is reported in full — see
"RESEARCH INTERPRETATION" at the end.

**Files that were requested but not found in this repository/branch
(`Suren-1`):** `DeepBudgetVis_Preprocessing_v2.ipynb`, `EDA on 10 yrs
data.ipynb`, `DeepBudgetVis_Baselines(1).ipynb`, and the TODO screenshot. Only
`Prop_model_v1 (1).ipynb` and the five CSVs (`train.csv`, `val.csv`,
`test.csv`, `val_with_context.csv`, `test_with_context.csv`) were present.
Everything in this notebook is grounded in those actually-supplied files;
anywhere this notebook needed information from a missing file, it says so
explicitly instead of guessing.
""")

# ============================================================
# 1. Imports
# ============================================================
md("## 1. Imports")
code("""import os, json, random, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import explained_variance_score, median_absolute_error, max_error
import matplotlib.pyplot as plt

print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
""", outputs_text="torch: 2.8.0+cu128 | cuda available: False\n")

# ============================================================
# 2. Configuration / reproducibility
# ============================================================
md("## 2. Configuration and reproducibility")
code("""SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", DEVICE)

REPO_DIR = os.path.abspath('..')          # Advertisement_effect/
OUT_DIR = 'outputs'
EXP_DIR = 'experiments'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(REPO_DIR, 'train.csv')
VAL_CSV = os.path.join(REPO_DIR, 'val.csv')
TEST_CSV = os.path.join(REPO_DIR, 'test.csv')
VAL_CTX_CSV = os.path.join(REPO_DIR, 'val_with_context.csv')
TEST_CTX_CSV = os.path.join(REPO_DIR, 'test_with_context.csv')
CONTEXT_ROWS = 30   # verified below: *_with_context.csv prepend exactly 30 rows from the preceding split
""", outputs_text="Device: cpu\n")

# ============================================================
# 3. Data loading and verification
# ============================================================
md("""## 3. Data loading and split verification

**Verified facts about the supplied files** (measured directly, not assumed):

| file | rows | cols | date range | duplicate dates |
|---|---|---|---|---|
| train.csv | 2557 | 85 | 2016-01-01 .. 2022-12-31 (84 months) | 0 |
| val.csv | 547 | 85 | 2023-01-01 .. 2024-06-30 (18 months) | 0 |
| test.csv | 549 | 85 | 2024-07-01 .. 2025-12-31 (18 months) | 0 |

The supplied split is **already** exactly 84/18/18 months, chronologically
ordered — no re-splitting was needed or performed.""")

code("""def load_splits():
    train_df = pd.read_csv(TRAIN_CSV, parse_dates=['DATE']).sort_values('DATE').reset_index(drop=True)
    val_df = pd.read_csv(VAL_CTX_CSV, parse_dates=['DATE']).sort_values('DATE').reset_index(drop=True)
    test_df = pd.read_csv(TEST_CTX_CSV, parse_dates=['DATE']).sort_values('DATE').reset_index(drop=True)
    return train_df, val_df, test_df

train_df, val_df, test_df = load_splits()

for name, df in [('train', pd.read_csv(TRAIN_CSV, parse_dates=['DATE'])),
                 ('val', pd.read_csv(VAL_CSV, parse_dates=['DATE'])),
                 ('test', pd.read_csv(TEST_CSV, parse_dates=['DATE']))]:
    n_months = (df['DATE'].max().to_period('M') - df['DATE'].min().to_period('M')).n + 1
    print(f"{name:6s}: rows={len(df):5d} cols={df.shape[1]:3d} "
          f"range={df['DATE'].min().date()}..{df['DATE'].max().date()} "
          f"months={n_months} dup_dates={df['DATE'].duplicated().sum()}")
""", outputs_text=(
    "train : rows= 2557 cols= 85 range=2016-01-01..2022-12-31 months=84 dup_dates=0\n"
    "val   : rows=  547 cols= 85 range=2023-01-01..2024-06-30 months=18 dup_dates=0\n"
    "test  : rows=  549 cols= 85 range=2024-07-01..2025-12-31 months=18 dup_dates=0\n"
))

md("""### 3.1 Verify `*_with_context.csv` are leak-free

`val_with_context.csv` / `test_with_context.csv` add extra leading rows so
that sequence windows for the FIRST few targets of val/test can look back
across the split boundary. These extra rows must be byte-identical copies of
already-legitimate historical data (not re-fit, not from the future).""")

code("""val_official = pd.read_csv(VAL_CSV, parse_dates=['DATE'])
merged = val_official.merge(val_df, on='DATE', suffixes=('_off', '_ctx'))
mismatch = [c for c in val_official.columns if c != 'DATE' and not (merged[f'{c}_off'] == merged[f'{c}_ctx']).all()]
n_context_rows_val = int((val_df['DATE'] < val_official['DATE'].min()).sum())
print(f"val_with_context extra leading rows: {n_context_rows_val} | mismatched cols vs val.csv: {mismatch}")

test_official = pd.read_csv(TEST_CSV, parse_dates=['DATE'])
merged2 = test_official.merge(test_df, on='DATE', suffixes=('_off', '_ctx'))
mismatch2 = [c for c in test_official.columns if c != 'DATE' and not (merged2[f'{c}_off'] == merged2[f'{c}_ctx']).all()]
n_context_rows_test = int((test_df['DATE'] < test_official['DATE'].min()).sum())
print(f"test_with_context extra leading rows: {n_context_rows_test} | mismatched cols vs test.csv: {mismatch2}")
""", outputs_text=(
    "val_with_context extra leading rows: 30 | mismatched cols vs val.csv: []\n"
    "test_with_context extra leading rows: 30 | mismatched cols vs test.csv: []\n"
))

# ============================================================
# 4. Scaling / leakage audit
# ============================================================
md("""## 4. Leakage audit — feature and target scaling

**Claim to verify:** every `<col>_scaled` column in the CSVs is a z-score
fit **only** on `train.csv` (features) or a log1p+z-score fit **only** on
`train.csv` (targets), and the SAME train-fit statistics are applied to
val/test (no separate fitting on val/test).""")

code("""DEMAND_COLS = ['OCCUPANCY_RATE','STAFFED_BEDS','ADMISSIONS','ER_VISITS','OP_VISITS','SURGERIES',
               'DISCHARGES','AVG_LENGTH_OF_STAY','YEAR','IS_WEEKEND','QUARTER','IS_HOLIDAY',
               'DOW_SIN','DOW_COS','MONTH_SIN','MONTH_COS']
REVCYCLE_COLS = ['GROSS_CHARGES_MEDICARE','GROSS_CHARGES_MEDICAID','GROSS_CHARGES_COMMERCIAL',
                 'GROSS_CHARGES_SELFPAY','GROSS_CHARGES_OTHER','TOTAL_GROSS_CHARGES','CHARITY_CARE',
                 'BAD_DEBT','CLAIMS_SUBMITTED','DENIAL_RATE','CLAIMS_DENIED','CASH_COLLECTED']
EXPENSE_COLS = ['LABOR_EXP','SUPPLY_EXP','OVERHEAD_EXP','CAPITAL_EXP','OTHER_OPERATING_EXP',
                'BUDGETED_LABOR_EXP','BUDGETED_SUPPLY_EXP','BUDGETED_PHARMACY_EXP',
                'BUDGETED_OVERHEAD_EXP','BUDGET_VARIANCE','OPERATING_MARGIN_PCT']
ALL_FEATURE_COLS = DEMAND_COLS + REVCYCLE_COLS + EXPENSE_COLS
TARGET_COLS = ['NET_PATIENT_REVENUE', 'TOTAL_OPERATING_EXP']
print("n feature cols:", len(ALL_FEATURE_COLS), "| demand:", len(DEMAND_COLS),
      "| revcycle:", len(REVCYCLE_COLS), "| expense:", len(EXPENSE_COLS))

train_raw = pd.read_csv(TRAIN_CSV, parse_dates=['DATE'])
val_raw = pd.read_csv(VAL_CSV, parse_dates=['DATE'])
test_raw = pd.read_csv(TEST_CSV, parse_dates=['DATE'])

max_diff_val, max_diff_test = 0.0, 0.0
for col in ALL_FEATURE_COLS:
    mu, sd = train_raw[col].mean(), train_raw[col].std(ddof=0)
    if sd == 0:
        continue
    max_diff_val = max(max_diff_val, np.max(np.abs((val_raw[col]-mu)/sd - val_raw[col+'_scaled'])))
    max_diff_test = max(max_diff_test, np.max(np.abs((test_raw[col]-mu)/sd - test_raw[col+'_scaled'])))
print(f"Max reconstruction error, ALL 39 features, train-fit z-score applied to val: {max_diff_val:.2e}")
print(f"Max reconstruction error, ALL 39 features, train-fit z-score applied to test: {max_diff_test:.2e}")

for tcol in TARGET_COLS:
    logtr = np.log1p(train_raw[tcol]); mu, sd = logtr.mean(), logtr.std(ddof=0)
    dv = np.max(np.abs((np.log1p(val_raw[tcol])-mu)/sd - val_raw[tcol+'_scaled']))
    dt = np.max(np.abs((np.log1p(test_raw[tcol])-mu)/sd - test_raw[tcol+'_scaled']))
    print(f"{tcol}: log1p+zscore (train-fit) reconstruction error -> val={dv:.2e}, test={dt:.2e}")
""", outputs_text=(
    "n feature cols: 39 | demand: 16 | revcycle: 12 | expense: 11\n"
    "Max reconstruction error, ALL 39 features, train-fit z-score applied to val: 1.78e-15\n"
    "Max reconstruction error, ALL 39 features, train-fit z-score applied to test: 1.78e-15\n"
    "NET_PATIENT_REVENUE: log1p+zscore (train-fit) reconstruction error -> val=4.44e-16, test=4.44e-16\n"
    "TOTAL_OPERATING_EXP: log1p+zscore (train-fit) reconstruction error -> val=2.31e-14, test=2.31e-14\n"
))

md("""**Conclusion:** all scaling (39 features + 2 targets) is verified fit
strictly on `train.csv` and applied identically to val/test (reconstruction
error is floating-point noise, <3e-14 in every case). **No preprocessing
leakage was found.**""")

# ============================================================
# 5. CAPEX / derived-column integrity audit
# ============================================================
md("""## 5. CAPEX / derived-column integrity audit

The task asked to verify `CAPITAL_EXP`, `TOTAL_OPERATING_EXP`,
`OPERATING_MARGIN_PCT`, `BUDGET_VARIANCE` (and any other dependent columns
discovered) for accounting consistency, and to correct any real
inconsistency found — but **not** to invent a correction if the true formula
cannot be reconstructed from the supplied files.""")

code("""tr = pd.read_csv(TRAIN_CSV, parse_dates=['DATE'])

# (a) OPERATING_MARGIN_PCT
computed_margin = (tr['NET_PATIENT_REVENUE'] - tr['TOTAL_OPERATING_EXP']) / tr['NET_PATIENT_REVENUE'] * 100
print("OPERATING_MARGIN_PCT vs (Rev-Exp)/Rev*100, max abs diff:",
      float((computed_margin - tr['OPERATING_MARGIN_PCT']).abs().max()), "-> VERIFIED (floating point noise only)")

# (b) TOTAL_OPERATING_EXP vs sum of the 5 visible expense components
sum5 = tr['LABOR_EXP'] + tr['SUPPLY_EXP'] + tr['OVERHEAD_EXP'] + tr['CAPITAL_EXP'] + tr['OTHER_OPERATING_EXP']
gap = tr['TOTAL_OPERATING_EXP'] - sum5
print("TOTAL_OPERATING_EXP - (LABOR+SUPPLY+OVERHEAD+CAPITAL+OTHER): mean=%.1f max=%.1f (NOT zero -> identity REJECTED)"
      % (gap.mean(), gap.max()))

# (c) BUDGET_VARIANCE vs (TOTAL_OPERATING_EXP - 4 budget lines)
budget4 = tr['BUDGETED_LABOR_EXP'] + tr['BUDGETED_SUPPLY_EXP'] + tr['BUDGETED_PHARMACY_EXP'] + tr['BUDGETED_OVERHEAD_EXP']
resid = (tr['TOTAL_OPERATING_EXP'] - budget4) - tr['BUDGET_VARIANCE']
print("BUDGET_VARIANCE vs TOTAL_OPERATING_EXP - budget4: mean resid=%.1f, corr(resid, CAPITAL_EXP)=%.4f"
      % (resid.mean(), resid.corr(tr['CAPITAL_EXP'])))
coef = np.polyfit(tr['CAPITAL_EXP'], resid, 1)
fit_resid = np.max(np.abs(np.polyval(coef, tr['CAPITAL_EXP']) - resid))
print("Linear fit resid ~ a*CAPITAL_EXP + b: slope=%.4f intercept=%.1f | max residual after fit=%.1f"
      % (coef[0], coef[1], fit_resid))
""", outputs_text=(
    "OPERATING_MARGIN_PCT vs (Rev-Exp)/Rev*100, max abs diff: 2.1316282072803006e-14 -> VERIFIED (floating point noise only)\n"
    "TOTAL_OPERATING_EXP - (LABOR+SUPPLY+OVERHEAD+CAPITAL+OTHER): mean=14255.6 max=17880.8 (NOT zero -> identity REJECTED)\n"
    "BUDGET_VARIANCE vs TOTAL_OPERATING_EXP - budget4: mean resid=3608.4, corr(resid, CAPITAL_EXP)=0.9969\n"
    "Linear fit resid ~ a*CAPITAL_EXP + b: slope=1.0005 intercept=2992.5 | max residual after fit=172.8\n"
))

md("""**Findings (full detail in `experiments/capex_accounting_audit.md`):**

1. **`OPERATING_MARGIN_PCT` — VERIFIED.** Matches `(Revenue - Expense) / Revenue * 100`
   exactly (floating-point noise only). No correction needed.
2. **`TOTAL_OPERATING_EXP` — CANNOT VERIFY the exact generating formula.**
   It is consistently ~14,250 larger than the sum of the 5 visible expense
   components (LABOR, SUPPLY, OVERHEAD, CAPITAL, OTHER_OPERATING). This gap's
   magnitude (mean ≈14,256, std ≈1,612) is close to `BUDGETED_PHARMACY_EXP`
   (mean ≈12,077 per the CSVs), consistent with an **unobserved actual
   pharmacy-expense column** that has no raw counterpart in the supplied
   CSVs — but this cannot be confirmed without that column.
3. **`BUDGET_VARIANCE` — CANNOT FULLY VERIFY.** `TOTAL_OPERATING_EXP - budget4`
   minus `BUDGET_VARIANCE` correlates 0.997 with `CAPITAL_EXP` (i.e. CAPEX
   appears excluded from the budget-variance comparison, slope≈1.0), but a
   residual constant offset (mean≈2,993, std≈48) remains unexplained — likely
   related to the same missing pharmacy-actual column as (2), but not
   confirmed.
4. **Decision: no correction was applied.** Per the task's explicit
   instruction, a "correction" would require inventing an unobserved
   PHARMACY_EXP series that does not exist in the supplied files — this
   violates the strict non-hallucination rule. `TOTAL_OPERATING_EXP` and
   `BUDGET_VARIANCE` are used exactly as delivered in the CSVs, unmodified.
5. The preprocessing notebook that generated these CSVs
   (`DeepBudgetVis_Preprocessing_v2.ipynb`) was **not found** in this
   repository/branch, so the true generating formula cannot be inspected
   directly — this is stated rather than guessed.""")

# ============================================================
# 6. Feature/target/stream verification
# ============================================================
md("""## 6. Feature / target / stream verification (unchanged from v1)

Stream assignment is **identical** to `Prop_model_v1 (1).ipynb`: 39 raw
feature columns split into Demand (16), Revenue-Cycle (12), Expense (11)
streams. This was NOT changed, because no data/EDA evidence was found that
the grouping itself is wrong — the grouping follows mechanistic hospital
concepts (census/demand, revenue-cycle/claims, expense/budget), which
remains valid.

**Revenue-cycle candidate columns checked for possible reintroduction
(Upgrade 7):** `AR_BALANCE`, `DAYS_IN_AR`, `CLAIMS_PAID`,
`CONTRACTUAL_ADJUSTMENTS` — **none of these columns exist** in the supplied
`train.csv`/`val.csv`/`test.csv` (verified by direct column-name search
below). `CASH_COLLECTED` already exists and was already part of
`REVCYCLE_COLS` in v1. **No new columns were added** — there was nothing
available to add.""")

code("""candidates = ['AR_BALANCE', 'DAYS_IN_AR', 'CLAIMS_PAID', 'CONTRACTUAL_ADJUSTMENTS', 'CASH_COLLECTED']
cols = list(train_raw.columns)
for name in candidates:
    print(f"{name:28s} -> {'EXISTS' if name in cols else 'DOES NOT EXIST'}")

assert set(ALL_FEATURE_COLS) == set([c for c in train_raw.columns
    if not c.endswith('_scaled') and c not in ['DATE', 'IS_ANOMALY', 'ANOMALY_TYPE'] + TARGET_COLS]), \\
    "Stream column lists do not match the actual pruned feature set"
print("\\nStream assignment verified against actual train.csv columns: OK")
""", outputs_text=(
    "AR_BALANCE                   -> DOES NOT EXIST\n"
    "DAYS_IN_AR                    -> DOES NOT EXIST\n"
    "CLAIMS_PAID                   -> DOES NOT EXIST\n"
    "CONTRACTUAL_ADJUSTMENTS       -> DOES NOT EXIST\n"
    "CASH_COLLECTED                -> EXISTS\n"
    "\n"
    "Stream assignment verified against actual train.csv columns: OK\n"
))

# ============================================================
# 7. EDA evidence used to justify upgrades
# ============================================================
md("""## 7. EDA evidence used to justify the upgrades below

`EDA on 10 yrs data.ipynb` was **not found** in this repository/branch, so
the EDA evidence used to justify architectural changes was measured directly
from `train.csv` in this notebook (not assumed, not copied from a missing
file).""")

code("""from scipy import stats as sstats

# (a) Revenue-to-cash collection lag (justifies keeping the 14-21 day lag-attention window)
tgc, cash = train_raw['TOTAL_GROSS_CHARGES'], train_raw['CASH_COLLECTED']
lag_corrs = {lag: tgc.shift(lag).corr(cash) for lag in [0, 7, 10, 13, 14, 15, 21, 28]}
best_lag = max(lag_corrs, key=lag_corrs.get)
print("TOTAL_GROSS_CHARGES -> CASH_COLLECTED correlation by lag (days):")
for lag, c in lag_corrs.items():
    print(f"  lag={lag:2d}: corr={c:.4f}")
print(f"Peak correlation at lag={best_lag} days (corr={lag_corrs[best_lag]:.4f})")

# (b) Target skew/kurtosis (tests the "heavy tail -> use Huber loss" hypothesis)
print()
for col in TARGET_COLS:
    sk, ku = sstats.skew(train_raw[col]), sstats.kurtosis(train_raw[col])
    print(f"{col}: skew={sk:.3f} kurtosis={ku:.3f} (near-symmetric, NOT heavy-tailed)")
for col in ['SUPPLY_EXP']:
    sk, ku = sstats.skew(train_raw[col]), sstats.kurtosis(train_raw[col])
    print(f"{col} (a FEATURE, not a target): skew={sk:.3f} kurtosis={ku:.3f} (heavy-tailed)")

# (c) Target autocorrelation by lag (justifies window-length search range)
print()
for col in TARGET_COLS:
    print(f"{col} autocorrelation:", {lag: round(train_raw[col].autocorr(lag), 3) for lag in [1, 7, 14, 21, 30, 45, 60]})
""", outputs_text=(
    "TOTAL_GROSS_CHARGES -> CASH_COLLECTED correlation by lag (days):\n"
    "  lag= 0: corr=0.7477\n"
    "  lag= 7: corr=0.7573\n"
    "  lag=10: corr=0.3780\n"
    "  lag=13: corr=0.6156\n"
    "  lag=14: corr=0.8728\n"
    "  lag=15: corr=0.6153\n"
    "  lag=21: corr=0.7558\n"
    "  lag=28: corr=0.7518\n"
    "Peak correlation at lag=14 days (corr=0.8728)\n"
    "\n"
    "NET_PATIENT_REVENUE: skew=0.075 kurtosis=-0.510 (near-symmetric, NOT heavy-tailed)\n"
    "TOTAL_OPERATING_EXP: skew=0.238 kurtosis=0.703 (near-symmetric, NOT heavy-tailed)\n"
    "SUPPLY_EXP (a FEATURE, not a target): skew=11.173 kurtosis=150.631 (heavy-tailed)\n"
    "\n"
    "NET_PATIENT_REVENUE autocorrelation: {1: 0.695, 7: 0.862, 14: 0.856, 21: 0.853, 30: 0.43, 45: 0.393, 60: 0.376}\n"
    "TOTAL_OPERATING_EXP autocorrelation: {1: 0.68, 7: 0.858, 14: 0.839, 21: 0.832, 30: 0.449, 45: 0.408, 60: 0.368}\n"
))

md("""**What this EDA supports:**
- The ~14-day lag between `TOTAL_GROSS_CHARGES` and `CASH_COLLECTED` is real
  and measured directly — it justifies **keeping** v1's lag-attention module
  in the Revenue-Cycle encoder (its `lag_window≈21` default already covers
  the 14-day peak with margin). No redesign of this module is justified.
- Target autocorrelation drops sharply between lag 21 and lag 30 — this
  motivates testing window lengths around 14-21 days rather than assuming
  30 is optimal (Section 9, Upgrade 8).
- **The heavy-tail evidence is on a FEATURE (`SUPPLY_EXP`), not on either
  forecasting TARGET.** Both targets are near-symmetric with negative-to-mild
  kurtosis. This means the original stated justification for testing a
  robust loss (Huber) does not actually apply to what the loss function
  operates on — this is tested empirically anyway in Section 9 rather than
  assumed, and the result is reported honestly either way.""")

# ============================================================
# 8. Dataset construction
# ============================================================
md("""## 8. Sequence dataset construction (Upgrade 1 groundwork + leakage-safe target restriction)

**Bug found and fixed during this project (reported honestly):** a naive
implementation (`idx` from `0` to `len(df)-seq_len`, identical to v1's
`MultiStreamDataset.__getitem__`) works correctly for `train.csv` (no
context rows), but when applied to `val_with_context.csv` /
`test_with_context.csv` with `seq_len < CONTEXT_ROWS` (=30), it would
silently produce "validation"/"test" targets whose DATE falls **inside**
the copied context window — i.e. dates that were already legitimate
training/validation targets in the preceding split. Evaluating on those
would contaminate validation/test metrics with disguised training/validation
performance. This was caught empirically (23 contaminated targets for
`seq_len=7`, `n_context=30`) before it could bias any reported result, and
fixed by restricting valid target indices to `>= max(n_context, seq_len)`.
""")

code("""TARGET_COLS_S = [f'{c}_scaled' for c in TARGET_COLS]
DEMAND_COLS_S = [f'{c}_scaled' for c in DEMAND_COLS]
REVCYCLE_COLS_S = [f'{c}_scaled' for c in REVCYCLE_COLS]
EXPENSE_COLS_S = [f'{c}_scaled' for c in EXPENSE_COLS]

class MultiStreamDataset(Dataset):
    \"\"\"Returns (xd, xr, xe, xh, y, is_anomaly) per window.

    xh (Upgrade 1 - target-history stream): the SAME seq_len window of past
    scaled target values (t-seq_len .. t-1) - strictly historical relative to
    the prediction target at time t. Never includes the future target.

    n_context: rows in `df` belonging to the PRECEDING split (0 for train.csv,
    CONTEXT_ROWS for *_with_context.csv). Valid targets are restricted to
    indices >= max(n_context, seq_len) - this is the leakage/contamination
    fix described above.
    \"\"\"
    def __init__(self, df, seq_len, include_target_history=True, n_context=0):
        self.Xd = df[DEMAND_COLS_S].values.astype(np.float32)
        self.Xr = df[REVCYCLE_COLS_S].values.astype(np.float32)
        self.Xe = df[EXPENSE_COLS_S].values.astype(np.float32)
        self.Xh = df[TARGET_COLS_S].values.astype(np.float32)
        self.y = df[TARGET_COLS_S].values.astype(np.float32)
        self.is_anomaly = df['IS_ANOMALY'].values.astype(np.float32)
        self.seq_len = seq_len
        self.include_target_history = include_target_history
        n = len(self.Xd)
        first_target = max(n_context, seq_len)
        self.target_indices = list(range(first_target, n))

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, i):
        target_idx = self.target_indices[i]
        s = slice(target_idx - self.seq_len, target_idx)
        xh = self.Xh[s] if self.include_target_history else np.zeros((self.seq_len, len(TARGET_COLS)), dtype=np.float32)
        return (self.Xd[s], self.Xr[s], self.Xe[s], xh, self.y[target_idx], self.is_anomaly[target_idx])

# Demonstrate the contamination bug and its fix
CONTEXT_ROWS = 30
ds_buggy_targets = list(range(0, len(val_df) - 7))                    # naive v1-style indexing
ds_fixed = MultiStreamDataset(val_df, seq_len=7, n_context=CONTEXT_ROWS)
n_contaminated = sum(1 for i in ds_buggy_targets if i + 7 < CONTEXT_ROWS)
print(f"seq_len=7, n_context=30: naive indexing would allow {n_contaminated} contaminated targets "
      f"(dates inside the copied context window). Fixed dataset produces {len(ds_fixed)} targets, "
      f"all >= max(n_context, seq_len)={max(CONTEXT_ROWS,7)}.")
""", outputs_text=(
    "seq_len=7, n_context=30: naive indexing would allow 23 contaminated targets "
    "(dates inside the copied context window). Fixed dataset produces 547 targets, "
    "all >= max(n_context, seq_len)=30.\n"
))

md("""**Note on `IS_ANOMALY`:** it is loaded here strictly as the auxiliary
supervision label returned alongside `y` — it is **never** placed into `xd`,
`xr`, `xe`, or `xh`, and is never used as a model input anywhere in this
notebook.""")

# ============================================================
# 9. Model definitions
# ============================================================
md("""## 9. Model definitions

### 9.1 What was already present in v1 (kept unchanged)

- `DemandEncoder`: 1D-CNN (kernel=7) + 2-layer GRU.
- `RevCycleEncoder`: 2-layer GRU + lag-attention over the last `lag_window`
  steps (kept — justified by the measured 14-day lag in Section 7).
- `ExpenseEncoder`: plain 2-layer GRU.
- Cross-stream fusion via multi-head attention + residual + LayerNorm.
- An anomaly auxiliary head trained with `IS_ANOMALY` as the label (never as
  input) via a joint BCE loss term (weight 0.1).

### 9.2 What was changed, and why (see Section 14 "WHAT CHANGED AND WHY" for full detail)

- **Upgrade 1 (History encoder):** a new small GRU stream over past target
  values, fused via the *same* cross-stream attention mechanism as a 4th
  token — not duplicated into other streams.
- **Upgrade 2 (Attention pooling):** v1 pools each stream by
  `out[:, -1, :]` (last hidden state only). Replaced with a learned additive
  attention pool over the whole valid sequence, for every stream.
- **Upgrade 3 (Target-specific heads):** v1 uses one shared linear layer for
  both targets. Replaced with two separate small MLP heads (Revenue,
  Expense) off the same shared fused representation.
- **Upgrade 6 (Explicit anomaly-conditioned gate):** v1's gate is a
  `Linear+Sigmoid` fed the *same* fused representation as the anomaly head,
  but the anomaly probability itself never explicitly touches the gate — it
  is only implicitly, jointly trained. Replaced with a gate that is
  literally a function of the anomaly probability
  (`gate = sigmoid(W @ (1 - anomaly_prob))`), so the forecast path is
  explicitly modulated by how anomalous the model currently believes the
  input is. `IS_ANOMALY` itself is still never an input anywhere.""")

code("""class DemandEncoder(nn.Module):
    def __init__(self, n_features, hidden=48, dropout=0.25):
        super().__init__()
        self.conv = nn.Conv1d(n_features, 32, kernel_size=7, padding=3)
        self.act = nn.ReLU()
        self.gru = nn.GRU(32, hidden, batch_first=True, dropout=dropout, num_layers=2)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = x.transpose(1, 2); x = self.act(self.conv(x)); x = x.transpose(1, 2)
        out, _ = self.gru(x)
        return self.drop(out)

class RevCycleEncoder(nn.Module):
    def __init__(self, n_features, hidden=48, lag_window=21, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True, dropout=dropout, num_layers=2)
        self.lag_window = lag_window
        self.lag_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        out, _ = self.gru(x)
        w = min(self.lag_window, out.size(1))
        recent = out[:, -w:, :]
        attn_out, _ = self.lag_attn(recent, recent, recent)
        fused = out.clone()
        fused[:, -w:, :] = fused[:, -w:, :] + attn_out
        return self.drop(fused)

class ExpenseEncoder(nn.Module):
    def __init__(self, n_features, hidden=48, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True, dropout=dropout, num_layers=2)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.drop(out)

class HistoryEncoder(nn.Module):
    \"\"\"Upgrade 1: dedicated encoder over PAST target history only (t-seq_len..t-1).\"\"\"
    def __init__(self, n_targets=2, hidden=48, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(n_targets, hidden, batch_first=True, dropout=dropout, num_layers=1)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.drop(out)

class AttnPool(nn.Module):
    \"\"\"Upgrade 2: learned attention pooling, replaces out[:, -1, :].\"\"\"
    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Linear(hidden, 1)
    def forward(self, seq):
        scores = self.score(seq).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), seq).squeeze(1)
        return pooled, weights

print("Shared encoder blocks defined.")
""", outputs_text="Shared encoder blocks defined.\n")

code("""class BaselineForecaster(nn.Module):
    \"\"\"Faithful port of Prop_model_v1 (1).ipynb's DeepBudgetVisForecaster.
    Used ONLY to retrain v1 on the new 84/18/18 split (Section 10).\"\"\"
    def __init__(self, n_demand, n_rev, n_exp, n_targets, hidden=48, dropout=0.25, lag_window=21):
        super().__init__()
        self.demand_enc = DemandEncoder(n_demand, hidden, dropout)
        self.rev_enc = RevCycleEncoder(n_rev, hidden, lag_window, dropout)
        self.exp_enc = ExpenseEncoder(n_exp, hidden, dropout)
        self.fusion_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.anomaly_head = nn.Linear(hidden, 1)
        self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_targets)
        )
    def forward(self, xd, xr, xe, xh=None):
        hd = self.demand_enc(xd); hr = self.rev_enc(xr); he = self.exp_enc(xe)
        stacked = torch.stack([hd[:, -1, :], hr[:, -1, :], he[:, -1, :]], dim=1)
        fused, _ = self.fusion_attn(stacked, stacked, stacked)
        fused = self.fusion_norm(fused + stacked)
        fused_pooled = fused.mean(dim=1)
        anomaly_logit = self.anomaly_head(fused_pooled).squeeze(-1)
        gate_weights = self.gate(fused_pooled)
        gated = fused_pooled * gate_weights
        forecast = self.forecast_head(gated)
        return forecast, anomaly_logit


class UpgradedForecaster(nn.Module):
    \"\"\"Upgrades 1, 2, 3, 6 applied on top of v1's DemandEncoder / RevCycleEncoder /
    ExpenseEncoder / fusion-attention, which are kept unchanged.\"\"\"
    def __init__(self, n_demand, n_rev, n_exp, n_targets=2, hidden=48, dropout=0.25,
                 lag_window=21, use_history_stream=True):
        super().__init__()
        self.use_history_stream = use_history_stream
        self.demand_enc = DemandEncoder(n_demand, hidden, dropout)
        self.rev_enc = RevCycleEncoder(n_rev, hidden, lag_window, dropout)
        self.exp_enc = ExpenseEncoder(n_exp, hidden, dropout)
        if use_history_stream:
            self.hist_enc = HistoryEncoder(n_targets, hidden, dropout)

        self.pool_demand = AttnPool(hidden)
        self.pool_rev = AttnPool(hidden)
        self.pool_exp = AttnPool(hidden)
        if use_history_stream:
            self.pool_hist = AttnPool(hidden)

        self.fusion_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden)

        self.anomaly_head = nn.Linear(hidden, 1)
        self.anomaly_gate_proj = nn.Linear(1, hidden)
        self.forecast_dropout = nn.Dropout(dropout)

        self.revenue_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.expense_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, xd, xr, xe, xh=None):
        hd = self.demand_enc(xd); hr = self.rev_enc(xr); he = self.exp_enc(xe)
        pd_, _ = self.pool_demand(hd); pr_, _ = self.pool_rev(hr); pe_, _ = self.pool_exp(he)
        tokens = [pd_, pr_, pe_]
        if self.use_history_stream:
            hh = self.hist_enc(xh)
            ph_, _ = self.pool_hist(hh)
            tokens.append(ph_)
        stacked = torch.stack(tokens, dim=1)
        fused, _ = self.fusion_attn(stacked, stacked, stacked)
        fused = self.fusion_norm(fused + stacked)
        fused_pooled = fused.mean(dim=1)

        anomaly_logit = self.anomaly_head(fused_pooled).squeeze(-1)
        anomaly_prob = torch.sigmoid(anomaly_logit).unsqueeze(-1)
        gate = torch.sigmoid(self.anomaly_gate_proj(1.0 - anomaly_prob))  # Upgrade 6
        gated = self.forecast_dropout(fused_pooled * gate)

        revenue = self.revenue_head(gated)   # Upgrade 3
        expense = self.expense_head(gated)   # Upgrade 3
        forecast = torch.cat([revenue, expense], dim=-1)
        return forecast, anomaly_logit

n_targets = len(TARGET_COLS)
_probe = UpgradedForecaster(len(DEMAND_COLS), len(REVCYCLE_COLS), len(EXPENSE_COLS))
print(f"UpgradedForecaster parameter count (default hidden=48): {sum(p.numel() for p in _probe.parameters()):,}")
_probe_base = BaselineForecaster(len(DEMAND_COLS), len(REVCYCLE_COLS), len(EXPENSE_COLS), n_targets)
print(f"BaselineForecaster parameter count (hidden=48, matches original notebook's printed 99,235): "
      f"{sum(p.numel() for p in _probe_base.parameters()):,}")
del _probe, _probe_base
""", outputs_text=(
    "UpgradedForecaster parameter count (default hidden=48): 107,015\n"
    "BaselineForecaster parameter count (hidden=48, matches original notebook's printed 99,235): 99,235\n"
))

# ============================================================
# 10. Metrics + training utilities
# ============================================================
md("""## 10. Metrics and training utilities

Metric set is unchanged from v1 (MAE, RMSE, MAPE, SMAPE, R2, ExplainedVar,
MedianAE, MaxError, MAE_scaled, RMSE_scaled), but computed **per target**
(Upgrade 5) instead of on the two targets flattened together, so
`NET_PATIENT_REVENUE` and `TOTAL_OPERATING_EXP` are always reported
separately.""")

code("""def compute_metrics_1d(y_true, y_pred, y_true_scaled=None, y_pred_scaled=None):
    y_true = np.asarray(y_true).flatten(); y_pred = np.asarray(y_pred).flatten()
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2); ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else float('nan')
    evs = explained_variance_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    maxerr = max_error(y_true, y_pred)
    out = {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape), 'SMAPE': float(smape),
           'R2': float(r2), 'ExplainedVar': float(evs), 'MedianAE': float(medae), 'MaxError': float(maxerr)}
    if y_true_scaled is not None:
        y_true_scaled = np.asarray(y_true_scaled).flatten(); y_pred_scaled = np.asarray(y_pred_scaled).flatten()
        out['MAE_scaled'] = float(np.mean(np.abs(y_true_scaled - y_pred_scaled)))
        out['RMSE_scaled'] = float(np.sqrt(np.mean((y_true_scaled - y_pred_scaled) ** 2)))
    else:
        out['MAE_scaled'] = float('nan'); out['RMSE_scaled'] = float('nan')
    return out

def compute_metrics_per_target(y_true, y_pred, y_true_scaled, y_pred_scaled):
    return {col: compute_metrics_1d(y_true[:, i], y_pred[:, i], y_true_scaled[:, i], y_pred_scaled[:, i])
            for i, col in enumerate(TARGET_COLS)}

def compute_target_scaler_stats(train_df_):
    return {col: (float(np.log1p(train_df_[col]).mean()), float(np.log1p(train_df_[col]).std(ddof=0)))
            for col in TARGET_COLS}

def inverse_targets(y_scaled, target_stats):
    out = np.zeros_like(y_scaled)
    for i, col in enumerate(TARGET_COLS):
        mu, sd = target_stats[col]
        out[:, i] = np.expm1(y_scaled[:, i] * sd + mu)
    return out

print("Metrics utilities defined.")
""", outputs_text="Metrics utilities defined.\n")

code("""def make_loss(name):
    return nn.MSELoss() if name == 'mse' else nn.HuberLoss(delta=1.0)

def run_training(model, train_ds, val_ds, device, loss_name='mse', lr=1e-3, weight_decay=1e-4,
                  batch_size=32, max_epochs=60, patience=10, anomaly_weight=0.1,
                  use_history_stream=True, verbose=False, warmup_epochs=3, return_history=False):
    \"\"\"Train/val loop: warmup(3 epochs) + ReduceLROnPlateau + early stopping on
    validation combined loss (forecast loss + anomaly_weight * anomaly BCE),
    matching v1's training protocol exactly.\"\"\"
    model = model.to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    criterion_forecast = make_loss(loss_name)
    criterion_anomaly = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e+1)/max(1, warmup_epochs)))
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss, best_state, best_epoch, epochs_no_improve, history = float('inf'), None, -1, 0, []
    for epoch in range(max_epochs):
        model.train(); train_losses = []
        for xd, xr, xe, xh, y, is_anom in train_loader:
            xd, xr, xe, xh, y, is_anom = (t.to(device) for t in (xd, xr, xe, xh, y, is_anom))
            optimizer.zero_grad()
            pred, anom_logit = model(xd, xr, xe, xh) if use_history_stream else model(xd, xr, xe)
            loss = criterion_forecast(pred, y) + anomaly_weight * criterion_anomaly(anom_logit, is_anom)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        if epoch < warmup_epochs:
            warmup_scheduler.step()

        model.eval(); val_losses = []
        with torch.no_grad():
            for xd, xr, xe, xh, y, is_anom in val_loader:
                xd, xr, xe, xh, y, is_anom = (t.to(device) for t in (xd, xr, xe, xh, y, is_anom))
                pred, anom_logit = model(xd, xr, xe, xh) if use_history_stream else model(xd, xr, xe)
                loss = criterion_forecast(pred, y) + anomaly_weight * criterion_anomaly(anom_logit, is_anom)
                val_losses.append(loss.item())

        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        if epoch >= warmup_epochs:
            plateau_scheduler.step(val_loss)
        if return_history:
            history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': optimizer.param_groups[0]['lr']})
        if verbose:
            print(f"epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val_loss - 1e-6:
            best_val_loss, best_epoch, epochs_no_improve = val_loss, epoch, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break
    return best_val_loss, best_state, best_epoch, history

@torch.no_grad()
def predict(model, ds, device, use_history_stream=True, batch_size=32):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    all_true, all_pred, all_anom = [], [], []
    for xd, xr, xe, xh, y, is_anom in loader:
        xd, xr, xe, xh = xd.to(device), xr.to(device), xe.to(device), xh.to(device)
        pred, _ = model(xd, xr, xe, xh) if use_history_stream else model(xd, xr, xe)
        all_true.append(y.numpy()); all_pred.append(pred.cpu().numpy()); all_anom.append(is_anom.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_anom)

print("Training utilities defined.")
""", outputs_text="Training utilities defined.\n")

# ============================================================
# 11. Window-length selection (Upgrade 8) - train/val only
# ============================================================
md("""## 11. Window-length selection (Upgrade 8) — train → val only

Candidates: 7, 14, 21, 30, 45, 60 days (per the task spec). Each candidate is
trained on `train.csv` and evaluated on `val.csv` (using
`val_with_context.csv` so short windows are not artificially penalized near
the split boundary), max 40 epochs, patience 8. **`test.csv` is never loaded
in this cell.**

This cell reproduces `run_window_selection.py` in this directory; the code
below is copied verbatim from that script. Because the full sweep takes
several minutes on CPU, the ACTUAL results already produced by running this
exact code in this session are shown as the captured output — re-running
this cell will reproduce the same numbers (seed=42, deterministic
CPU training).""")

code("""candidates = [7, 14, 21, 30, 45, 60]
window_results = []
for seq_len in candidates:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    train_ds = MultiStreamDataset(train_df, seq_len, include_target_history=True, n_context=0)
    val_ds = MultiStreamDataset(val_df, seq_len, include_target_history=True, n_context=CONTEXT_ROWS)
    model = UpgradedForecaster(len(DEMAND_COLS), len(REVCYCLE_COLS), len(EXPENSE_COLS))
    t0 = time.time()
    best_val_loss, best_state, best_epoch, _ = run_training(
        model, train_ds, val_ds, DEVICE, loss_name='mse', lr=1e-3, weight_decay=1e-4, batch_size=32,
        max_epochs=40, patience=8, use_history_stream=True, verbose=False)
    dt = time.time() - t0
    print(f"seq_len={seq_len:3d} | train_windows={len(train_ds):5d} val_windows={len(val_ds):4d} | "
          f"best_val_loss={best_val_loss:.5f} best_epoch={best_epoch} time={dt:.1f}s")
    window_results.append({'seq_len': seq_len, 'best_val_loss': best_val_loss, 'best_epoch': best_epoch,
                            'train_windows': len(train_ds), 'val_windows': len(val_ds), 'time_s': dt})

with open(os.path.join(EXP_DIR, 'window_selection_results.json'), 'w') as f:
    json.dump(window_results, f, indent=2)

best_window = min(window_results, key=lambda r: r['best_val_loss'])
print("\\nBEST WINDOW (lowest val loss, evaluated on identical 547-target val set for seq_len<=30):", best_window)
SEQ_LEN = best_window['seq_len']
""", outputs_text=(
    "seq_len=  7 | train_windows= 2550 val_windows= 547 | best_val_loss=0.12908 best_epoch=13 time=28.6s\n"
    "seq_len= 14 | train_windows= 2543 val_windows= 547 | best_val_loss=0.13226 best_epoch=26 time=75.6s\n"
    "seq_len= 21 | train_windows= 2536 val_windows= 547 | best_val_loss=0.12733 best_epoch=7 time=45.8s\n"
    "seq_len= 30 | train_windows= 2527 val_windows= 547 | best_val_loss=0.13109 best_epoch=16 time=96.3s\n"
    "seq_len= 45 | train_windows= 2512 val_windows= 532 | best_val_loss=0.13309 best_epoch=6 time=80.4s\n"
    "seq_len= 60 | train_windows= 2497 val_windows= 517 | best_val_loss=0.13000 best_epoch=11 time=143.1s\n"
    "\n"
    "BEST WINDOW (lowest val loss, evaluated on identical 547-target val set for seq_len<=30): "
    "{'seq_len': 21, 'best_val_loss': 0.12732616956863138, 'best_epoch': 7, 'train_windows': 2536, "
    "'val_windows': 547, 'time_s': 45.75918984413147}\n"
))

md("""**Decision: `SEQ_LEN = 21`.** This is the best of all 6 candidates, and
among the 4 candidates (`{7,14,21,30}`) that are evaluated on the exact same
547-target validation set (windows 45 and 60 are evaluated on a slightly
smaller validation set — 532/517 targets — because the 30-day context cannot
cover lookback windows longer than 30 days; this is reported honestly rather
than hidden). This also matches the EDA evidence in Section 7: target
autocorrelation drops sharply between lag 21 and lag 30, and the
revenue-to-cash lag peaks at 14 days (well inside a 21-day window). Note this
changes v1's fixed `SEQ_LEN=30` — this is a **verified, data-driven**
change, not an assumption.""")

# ============================================================
# 12. Loss selection (Upgrade 4) - train/val only
# ============================================================
md("""## 12. Loss function selection (Upgrade 4) — train → val only, `SEQ_LEN=21`

Motivation check first: Section 7 already showed the heavy-tail evidence
found in the EDA is on a FEATURE (`SUPPLY_EXP`), not on either target — both
targets are near-symmetric. The comparison below is still run empirically
rather than skipped, and the test set is never used to pick between MSE and
Huber.""")

code("""train_ds = MultiStreamDataset(train_df, SEQ_LEN, include_target_history=True, n_context=0)
val_ds = MultiStreamDataset(val_df, SEQ_LEN, include_target_history=True, n_context=CONTEXT_ROWS)

loss_results = []
for loss_name in ['mse', 'huber']:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = UpgradedForecaster(len(DEMAND_COLS), len(REVCYCLE_COLS), len(EXPENSE_COLS))
    best_val_loss, best_state, best_epoch, _ = run_training(
        model, train_ds, val_ds, DEVICE, loss_name=loss_name, lr=1e-3, weight_decay=1e-4, batch_size=32,
        max_epochs=50, patience=10, use_history_stream=True, verbose=False)
    model.load_state_dict(best_state); model.to(DEVICE)
    y_true_s, y_pred_s, _ = predict(model, val_ds, DEVICE, use_history_stream=True)
    target_stats = compute_target_scaler_stats(train_df)
    y_true, y_pred = inverse_targets(y_true_s, target_stats), inverse_targets(y_pred_s, target_stats)
    per_target = compute_metrics_per_target(y_true, y_pred, y_true_s, y_pred_s)
    print(f"loss={loss_name:6s} | best_val_loss={best_val_loss:.5f} best_epoch={best_epoch}")
    for t, m in per_target.items():
        print(f"   {t}: MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} R2={m['R2']:.4f}")
    loss_results.append({'loss': loss_name, 'best_val_loss': best_val_loss, 'val_metrics_per_target': per_target})

with open(os.path.join(EXP_DIR, 'loss_selection_results.json'), 'w') as f:
    json.dump(loss_results, f, indent=2)
""", outputs_text=(
    "loss=mse    | best_val_loss=0.12733 best_epoch=7\n"
    "   NET_PATIENT_REVENUE: MAE=6864.5 RMSE=8755.5 R2=0.7582\n"
    "   TOTAL_OPERATING_EXP: MAE=1647.0 RMSE=2975.3 R2=0.7447\n"
    "loss=huber  | best_val_loss=0.06483 best_epoch=7\n"
    "   NET_PATIENT_REVENUE: MAE=6849.1 RMSE=8740.8 R2=0.7590\n"
    "   TOTAL_OPERATING_EXP: MAE=1654.8 RMSE=2982.2 R2=0.7435\n"
))

md("""**Decision: `LOSS = 'mse'` (kept, Huber REJECTED).**

The raw `best_val_loss` values are **not** directly comparable between MSE
and Huber (different numeric scale near delta=1.0) — the correct comparison
is the per-target MAE/RMSE/R2 in original financial units, computed
identically for both. Huber improves `NET_PATIENT_REVENUE` MAE by ~0.2%
(6864.5→6849.1) but **worsens** `TOTAL_OPERATING_EXP` MAE by ~0.5%
(1647.0→1654.8); R2 differences are <0.002 on both targets. This is not a
consistent or meaningful improvement in either direction, so — per the
task's stated principle of preferring the simplest choice that
*demonstrably* improves forecasting — **MSE is kept.** This is an upgrade
that was tested and explicitly rejected because the data did not support
it.""")

# ============================================================
# 13. Time-aware hyperparameter search (Upgrade 10) - train only, inner folds
# ============================================================
md("""## 13. Time-aware hyperparameter random search (Upgrade 10)

Design: 3 **expanding inner folds inside `train.csv` only** (never touching
`val.csv` or `test.csv`):

| fold | inner-train months | inner-val months |
|---|---|---|
| 1 | 1-48 (2016-01..2019-12) | 49-60 (2020-01..2020-12) |
| 2 | 1-60 (2016-01..2020-12) | 61-72 (2021-01..2021-12) |
| 3 | 1-72 (2016-01..2021-12) | 73-84 (2022-01..2022-12) |

10 random configurations sampled from `{hidden, dropout, lr, weight_decay,
lag_window, batch_size}`, each trained on all 3 folds (15-epoch budget,
patience 5), scored by mean inner-val loss. **Bayesian optimization was NOT
used** — with only 10 trials and a 6-dimensional discrete grid, a random
search already explores the space reasonably, and there was no clear
evidence a smarter search strategy would change the qualitative outcome
enough to justify the added complexity; this decision is revisited honestly
below once the search's real limitation became apparent.""")

code("""train_df_ym = train_df.copy()
train_df_ym['ym'] = train_df_ym['DATE'].dt.to_period('M')
months = sorted(train_df_ym['ym'].unique())
assert len(months) == 84

FOLD_DEFS = [(48, 60), (60, 72), (72, 84)]
def make_fold_dfs(s, e):
    tr = train_df_ym[train_df_ym['ym'].isin(months[:s])].drop(columns=['ym']).reset_index(drop=True)
    va = train_df_ym[train_df_ym['ym'].isin(months[s:e])].drop(columns=['ym']).reset_index(drop=True)
    return tr, va

SEARCH_SPACE = {'hidden': [32, 48, 64], 'dropout': [0.20, 0.30, 0.40], 'lr': [3e-4, 5e-4, 1e-3, 2e-3],
                'weight_decay': [1e-5, 1e-4, 5e-4], 'lag_window': [14, 21, 28], 'batch_size': [16, 32]}
N_TRIALS, TUNE_MAX_EPOCHS, TUNE_PATIENCE = 10, 15, 5
rng = random.Random(SEED)
def sample_config():
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}

fold_data = [make_fold_dfs(s, e) for (s, e) in FOLD_DEFS]
hp_results = []
for trial in range(1, N_TRIALS + 1):
    cfg = sample_config()
    fold_losses = []
    for (tr_df, va_df) in fold_data:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        tr_ds = MultiStreamDataset(tr_df, SEQ_LEN, include_target_history=True, n_context=0)
        va_ds = MultiStreamDataset(va_df, SEQ_LEN, include_target_history=True, n_context=0)
        model = UpgradedForecaster(len(DEMAND_COLS), len(REVCYCLE_COLS), len(EXPENSE_COLS),
                                    hidden=cfg['hidden'], dropout=cfg['dropout'], lag_window=cfg['lag_window'])
        bvl, _, _, _ = run_training(model, tr_ds, va_ds, DEVICE, loss_name='mse', lr=cfg['lr'],
                                     weight_decay=cfg['weight_decay'], batch_size=cfg['batch_size'],
                                     max_epochs=TUNE_MAX_EPOCHS, patience=TUNE_PATIENCE, use_history_stream=True)
        fold_losses.append(bvl)
    avg_loss = sum(fold_losses) / len(fold_losses)
    print(f"Trial {trial:02d}: {cfg} -> avg_inner_val_loss={avg_loss:.5f}")
    hp_results.append({'trial': trial, **cfg, 'avg_inner_val_loss': avg_loss})

hp_results_sorted = sorted(hp_results, key=lambda r: r['avg_inner_val_loss'])
best_hp = {k: hp_results_sorted[0][k] for k in SEARCH_SPACE}
with open(os.path.join(EXP_DIR, 'hp_search_results.json'), 'w') as f:
    json.dump(hp_results_sorted, f, indent=2)
print("\\nBest inner-fold config:", best_hp, "| avg_inner_val_loss:", hp_results_sorted[0]['avg_inner_val_loss'])
""", outputs_text=(
    "Trial 01: {'hidden': 64, 'dropout': 0.2, 'lr': 0.0003, 'weight_decay': 0.0005, 'lag_window': 21, 'batch_size': 16} -> avg_inner_val_loss=0.15348\n"
    "Trial 02: {'hidden': 32, 'dropout': 0.2, 'lr': 0.0003, 'weight_decay': 0.0005, 'lag_window': 28, 'batch_size': 16} -> avg_inner_val_loss=0.19059\n"
    "Trial 03: {'hidden': 64, 'dropout': 0.3, 'lr': 0.0003, 'weight_decay': 1e-05, 'lag_window': 14, 'batch_size': 16} -> avg_inner_val_loss=0.15382\n"
    "Trial 04: {'hidden': 32, 'dropout': 0.4, 'lr': 0.0003, 'weight_decay': 0.0005, 'lag_window': 14, 'batch_size': 32} -> avg_inner_val_loss=0.23192\n"
    "Trial 05: {'hidden': 32, 'dropout': 0.3, 'lr': 0.001, 'weight_decay': 1e-05, 'lag_window': 14, 'batch_size': 32} -> avg_inner_val_loss=0.16603\n"
    "Trial 06: {'hidden': 48, 'dropout': 0.3, 'lr': 0.0005, 'weight_decay': 1e-05, 'lag_window': 21, 'batch_size': 16} -> avg_inner_val_loss=0.15747\n"
    "Trial 07: {'hidden': 32, 'dropout': 0.3, 'lr': 0.0003, 'weight_decay': 0.0001, 'lag_window': 21, 'batch_size': 32} -> avg_inner_val_loss=0.20368\n"
    "Trial 08: {'hidden': 32, 'dropout': 0.4, 'lr': 0.002, 'weight_decay': 0.0005, 'lag_window': 14, 'batch_size': 32} -> avg_inner_val_loss=0.16798\n"
    "Trial 09: {'hidden': 32, 'dropout': 0.4, 'lr': 0.001, 'weight_decay': 0.0005, 'lag_window': 28, 'batch_size': 32} -> avg_inner_val_loss=0.17249\n"
    "Trial 10: {'hidden': 64, 'dropout': 0.2, 'lr': 0.0003, 'weight_decay': 1e-05, 'lag_window': 28, 'batch_size': 16} -> avg_inner_val_loss=0.15403\n"
    "\n"
    "Best inner-fold config: {'hidden': 64, 'dropout': 0.2, 'lr': 0.0003, 'weight_decay': 0.0005, 'lag_window': 21, 'batch_size': 16} | avg_inner_val_loss: 0.15347640278438726\n"
))

md("""### 13.1 Honest outcome check: does the inner-fold winner transfer to the full-budget/real-val setting?

Before freezing this config for final training, it was validated under the
**actual** final-training protocol (full 84-month train, full training
budget up to 120 epochs, evaluated on the real 18-month `val.csv`) against
the model's **default** hyperparameters as a control, under an identical
budget.""")

code("""DEFAULT_HP = {'hidden': 48, 'dropout': 0.25, 'lr': 1e-3, 'weight_decay': 1e-4, 'lag_window': 21, 'batch_size': 32}
final_val_ds = MultiStreamDataset(val_df, SEQ_LEN, include_target_history=True, n_context=CONTEXT_ROWS)
final_train_ds = MultiStreamDataset(train_df, SEQ_LEN, include_target_history=True, n_context=0)

for label, hp in [('HP-search winner', best_hp), ('Default hyperparameters', DEFAULT_HP)]:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = UpgradedForecaster(len(DEMAND_COLS), len(REVCYCLE_COLS), len(EXPENSE_COLS),
                                hidden=hp['hidden'], dropout=hp['dropout'], lag_window=hp['lag_window'])
    bvl, _, be, _ = run_training(model, final_train_ds, final_val_ds, DEVICE, loss_name='mse', lr=hp['lr'],
                                  weight_decay=hp['weight_decay'], batch_size=hp['batch_size'],
                                  max_epochs=120, patience=15, use_history_stream=True)
    print(f"{label:26s} {hp} -> full-budget real-val best_val_loss={bvl:.5f} (epoch {be})")
""", outputs_text=(
    "HP-search winner           {'hidden': 64, 'dropout': 0.2, 'lr': 0.0003, 'weight_decay': 0.0005, 'lag_window': 21, 'batch_size': 16} -> full-budget real-val best_val_loss=0.13418 (epoch 35)\n"
    "Default hyperparameters    {'hidden': 48, 'dropout': 0.25, 'lr': 0.001, 'weight_decay': 0.0001, 'lag_window': 21, 'batch_size': 32} -> full-budget real-val best_val_loss=0.12733 (epoch 7)\n"
))

md("""**Result: the default hyperparameters beat the HP-search winner** under
the real training budget and real validation set (0.12733 vs 0.13418). The
inner-fold search's short 15-epoch-per-trial budget on small (~365-row)
inner-val folds did not transfer to the full ~120-epoch budget used for
final training — it favored a larger, slower-learning-rate config that
looks competitive early but is out-performed once training is allowed to
run to convergence.

**Decision: use the DEFAULT hyperparameters for final training, NOT the
HP-search result.** This is reported as an executed, honest experiment that
did not produce an improvement, per the task's explicit instruction to
report this outcome rather than force a result. Bayesian optimization was
not attempted afterward either, since there is no reason to believe a
smarter search over this same short-budget/small-fold protocol would fix
the fundamental budget/fold-size mismatch that caused the transfer failure —
the fix would be a longer inner-fold budget, not a smarter search algorithm,
and that was judged not worth the added compute for a search that the
default hyperparameters already outperform.

**Final frozen configuration for the upgraded model:**
`seq_len=21, loss=mse, hidden=48, dropout=0.25, lr=1e-3, weight_decay=1e-4,
lag_window=21, batch_size=32`.""")

code("""FINAL_HP = {'hidden': 48, 'dropout': 0.25, 'lr': 1e-3, 'weight_decay': 1e-4, 'lag_window': 21, 'batch_size': 32}
with open(os.path.join(EXP_DIR, 'final_hyperparameters.json'), 'w') as f:
    json.dump({'seq_len': SEQ_LEN, 'loss': 'mse', **FINAL_HP}, f, indent=2)
print("Frozen final config:", {'seq_len': SEQ_LEN, 'loss': 'mse', **FINAL_HP})
""", outputs_text="Frozen final config: {'seq_len': 21, 'loss': 'mse', 'hidden': 48, 'dropout': 0.25, 'lr': 0.001, 'weight_decay': 0.0001, 'lag_window': 21, 'batch_size': 32}\n")

# ============================================================
# 14. FINAL TRAINING - Upgraded model
# ============================================================
md("""## 14. Final training — Upgraded model (frozen configuration)

All decisions (window=21, loss=MSE, hyperparameters=defaults) are now
frozen. Trains on the full 84-month `train.csv`, checkpoints on the lowest
combined loss (forecast + 0.1×anomaly BCE) on `val.csv`, early stopping
patience=15, max 120 epochs. **`test.csv` is not loaded in this cell.**""")

code("""UPGRADED_OUT_DIR = os.path.join(OUT_DIR, 'upgraded_final')
os.makedirs(UPGRADED_OUT_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
train_ds = MultiStreamDataset(train_df, SEQ_LEN, include_target_history=True, n_context=0)
val_ds = MultiStreamDataset(val_df, SEQ_LEN, include_target_history=True, n_context=CONTEXT_ROWS)
assert len(val_ds) == 547, "Val windows must equal official val.csv row count"
print(f"train windows={len(train_ds)} val windows={len(val_ds)}")

upgraded_model = UpgradedForecaster(len(DEMAND_COLS), len(REVCYCLE_COLS), len(EXPENSE_COLS),
                                     hidden=FINAL_HP['hidden'], dropout=FINAL_HP['dropout'],
                                     lag_window=FINAL_HP['lag_window'])
n_params_upgraded = sum(p.numel() for p in upgraded_model.parameters())
print(f"Model parameters: {n_params_upgraded:,}")

t0 = time.time()
best_val_loss_upg, best_state_upg, best_epoch_upg, history_upg = run_training(
    upgraded_model, train_ds, val_ds, DEVICE, loss_name='mse', lr=FINAL_HP['lr'],
    weight_decay=FINAL_HP['weight_decay'], batch_size=FINAL_HP['batch_size'],
    max_epochs=120, patience=15, use_history_stream=True, verbose=True, return_history=True)
print(f"\\nTraining complete in {time.time()-t0:.1f}s. Best val_loss={best_val_loss_upg:.5f} at epoch {best_epoch_upg}")

torch.save(best_state_upg, os.path.join(UPGRADED_OUT_DIR, 'best_model.pt'))
pd.DataFrame(history_upg).to_csv(os.path.join(UPGRADED_OUT_DIR, 'history.csv'), index=False)
upgraded_config = {'seq_len': SEQ_LEN, 'loss': 'mse', **FINAL_HP, 'n_params': n_params_upgraded,
                    'best_epoch': best_epoch_upg, 'best_val_loss': best_val_loss_upg,
                    'train_windows': len(train_ds), 'val_windows': len(val_ds), 'seed': SEED}
with open(os.path.join(UPGRADED_OUT_DIR, 'config.json'), 'w') as f:
    json.dump(upgraded_config, f, indent=2)
""", outputs_text=(
    "train windows=2536 val windows=547\n"
    "Model parameters: 107,015\n"
    "epoch 000 train_loss=0.6916 val_loss=0.6251\n"
    "epoch 001 train_loss=0.4452 val_loss=0.3079\n"
    "epoch 002 train_loss=0.2395 val_loss=0.2045\n"
    "epoch 003 train_loss=0.2030 val_loss=0.1896\n"
    "epoch 004 train_loss=0.1964 val_loss=0.1900\n"
    "epoch 005 train_loss=0.1934 val_loss=0.2821\n"
    "epoch 006 train_loss=0.1889 val_loss=0.1527\n"
    "epoch 007 train_loss=0.1789 val_loss=0.1273\n"
    "epoch 008 train_loss=0.1786 val_loss=0.1853\n"
    "epoch 009 train_loss=0.1789 val_loss=0.1380\n"
    "epoch 010 train_loss=0.1836 val_loss=0.1320\n"
    "epoch 011 train_loss=0.1799 val_loss=0.1520\n"
    "epoch 012 train_loss=0.1776 val_loss=0.1399\n"
    "epoch 013 train_loss=0.1766 val_loss=0.1522\n"
    "epoch 014 train_loss=0.1743 val_loss=0.1357\n"
    "epoch 015 train_loss=0.1745 val_loss=0.1411\n"
    "epoch 016 train_loss=0.1751 val_loss=0.1430\n"
    "epoch 017 train_loss=0.1687 val_loss=0.1341\n"
    "epoch 018 train_loss=0.1717 val_loss=0.1328\n"
    "epoch 019 train_loss=0.1654 val_loss=0.1315\n"
    "epoch 020 train_loss=0.1719 val_loss=0.1415\n"
    "epoch 021 train_loss=0.1674 val_loss=0.1335\n"
    "epoch 022 train_loss=0.1732 val_loss=0.1346\n"
    "\n"
    "Training complete in 69.0s. Best val_loss=0.12733 at epoch 7\n"
))

code("""upgraded_model.load_state_dict(best_state_upg)
upgraded_model.to(DEVICE)
y_true_s, y_pred_s, anom_flags = predict(upgraded_model, val_ds, DEVICE, use_history_stream=True)
target_stats = compute_target_scaler_stats(train_df)
with open(os.path.join(UPGRADED_OUT_DIR, 'target_scaler_stats.json'), 'w') as f:
    json.dump(target_stats, f, indent=2)
y_true, y_pred = inverse_targets(y_true_s, target_stats), inverse_targets(y_pred_s, target_stats)
upgraded_val_metrics = compute_metrics_per_target(y_true, y_pred, y_true_s, y_pred_s)
print("VALIDATION metrics (per target), upgraded model:")
for t, m in upgraded_val_metrics.items():
    print(f"  {t}: {m}")
with open(os.path.join(UPGRADED_OUT_DIR, 'val_metrics.json'), 'w') as f:
    json.dump(upgraded_val_metrics, f, indent=2)
""", outputs_text=(
    "VALIDATION metrics (per target), upgraded model:\n"
    "  NET_PATIENT_REVENUE: {'MAE': 6864.494140625, 'RMSE': 8755.4853515625, 'MAPE': 3.991209030151367, "
    "'SMAPE': 3.995041608810425, 'R2': 0.7581543922424316, 'ExplainedVar': 0.7598437070846558, "
    "'MedianAE': 5482.046875, 'MaxError': 33040.625, 'MAE_scaled': 0.2702871561050415, 'RMSE_scaled': 0.3457127511501312}\n"
    "  TOTAL_OPERATING_EXP: {'MAE': 1647.0089111328125, 'RMSE': 2975.29296875, 'MAPE': 1.4839214086532593, "
    "'SMAPE': 1.493530035018921, 'R2': 0.7447104454040527, 'ExplainedVar': 0.7460042238235474, "
    "'MedianAE': 1145.4140625, 'MaxError': 28901.953125, 'MAE_scaled': 0.1933348923921585, 'RMSE_scaled': 0.33796027302742004}\n"
))

# ============================================================
# 15. FINAL TRAINING - Baseline v1 retrain (fair comparison)
# ============================================================
md("""## 15. Retrain original v1 architecture on the SAME 84/18/18 split (fair baseline)

`Prop_model_v1 (1).ipynb`'s printed results were produced under a different
(unverified) split, so they are not directly comparable to the numbers
above. `BaselineForecaster` (Section 9, a faithful, unmodified port of v1's
`DeepBudgetVisForecaster`) is retrained here on the identical 84/18/18
split, v1's original `seq_len=30` and original hyperparameters
(`hidden=48, dropout=0.25, lr=1e-3, weight_decay=1e-4, lag_window=21,
batch_size=16`), so the ONLY difference between this and Section 14 is the
architecture.""")

code("""BASELINE_OUT_DIR = os.path.join(OUT_DIR, 'baseline_v1_retrained')
os.makedirs(BASELINE_OUT_DIR, exist_ok=True)

V1_SEQ_LEN = 30
V1_HP = {'hidden': 48, 'dropout': 0.25, 'lr': 1e-3, 'weight_decay': 1e-4, 'lag_window': 21, 'batch_size': 16}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
train_ds_v1 = MultiStreamDataset(train_df, V1_SEQ_LEN, include_target_history=False, n_context=0)
val_ds_v1 = MultiStreamDataset(val_df, V1_SEQ_LEN, include_target_history=False, n_context=CONTEXT_ROWS)
print(f"train windows={len(train_ds_v1)} val windows={len(val_ds_v1)}")

baseline_model = BaselineForecaster(len(DEMAND_COLS), len(REVCYCLE_COLS), len(EXPENSE_COLS), n_targets,
                                     hidden=V1_HP['hidden'], dropout=V1_HP['dropout'], lag_window=V1_HP['lag_window'])
n_params_baseline = sum(p.numel() for p in baseline_model.parameters())
print(f"Model parameters: {n_params_baseline:,} (matches original notebook's printed 99,235)")

t0 = time.time()
best_val_loss_b, best_state_b, best_epoch_b, history_b = run_training(
    baseline_model, train_ds_v1, val_ds_v1, DEVICE, loss_name='mse', lr=V1_HP['lr'],
    weight_decay=V1_HP['weight_decay'], batch_size=V1_HP['batch_size'],
    max_epochs=120, patience=15, use_history_stream=False, verbose=True, return_history=True)
print(f"\\nTraining complete in {time.time()-t0:.1f}s. Best val_loss={best_val_loss_b:.5f} at epoch {best_epoch_b}")

torch.save(best_state_b, os.path.join(BASELINE_OUT_DIR, 'best_model.pt'))
pd.DataFrame(history_b).to_csv(os.path.join(BASELINE_OUT_DIR, 'history.csv'), index=False)
baseline_config = {'seq_len': V1_SEQ_LEN, 'loss': 'mse', **V1_HP, 'n_params': n_params_baseline,
                    'best_epoch': best_epoch_b, 'best_val_loss': best_val_loss_b,
                    'train_windows': len(train_ds_v1), 'val_windows': len(val_ds_v1), 'seed': SEED}
with open(os.path.join(BASELINE_OUT_DIR, 'config.json'), 'w') as f:
    json.dump(baseline_config, f, indent=2)
""", outputs_text=(
    "train windows=2527 val windows=547\n"
    "Model parameters: 99,235 (matches original notebook's printed 99,235)\n"
    "epoch 000 train_loss=0.5777 val_loss=0.3920\n"
    "epoch 001 train_loss=0.2685 val_loss=0.1613\n"
    "epoch 002 train_loss=0.2072 val_loss=0.1479\n"
    "epoch 003 train_loss=0.1891 val_loss=0.1524\n"
    "epoch 004 train_loss=0.1888 val_loss=0.1532\n"
    "epoch 005 train_loss=0.1811 val_loss=0.1441\n"
    "epoch 006 train_loss=0.1839 val_loss=0.1574\n"
    "epoch 007 train_loss=0.1755 val_loss=0.1402\n"
    "epoch 008 train_loss=0.1730 val_loss=0.1466\n"
    "epoch 009 train_loss=0.1769 val_loss=0.1438\n"
    "epoch 010 train_loss=0.1736 val_loss=0.1558\n"
    "epoch 011 train_loss=0.1715 val_loss=0.1501\n"
    "epoch 012 train_loss=0.1726 val_loss=0.1598\n"
    "epoch 013 train_loss=0.1630 val_loss=0.1437\n"
    "epoch 014 train_loss=0.1597 val_loss=0.1517\n"
    "epoch 015 train_loss=0.1580 val_loss=0.1520\n"
    "epoch 016 train_loss=0.1563 val_loss=0.1451\n"
    "epoch 017 train_loss=0.1583 val_loss=0.1334\n"
    "epoch 018 train_loss=0.1577 val_loss=0.1330\n"
    "epoch 019 train_loss=0.1578 val_loss=0.1432\n"
    "epoch 020 train_loss=0.1521 val_loss=0.1478\n"
    "epoch 021 train_loss=0.1539 val_loss=0.1485\n"
    "epoch 022 train_loss=0.1540 val_loss=0.1571\n"
    "epoch 023 train_loss=0.1529 val_loss=0.1375\n"
    "epoch 024 train_loss=0.1501 val_loss=0.1458\n"
    "epoch 025 train_loss=0.1477 val_loss=0.1729\n"
    "epoch 026 train_loss=0.1489 val_loss=0.1528\n"
    "epoch 027 train_loss=0.1470 val_loss=0.1494\n"
    "epoch 028 train_loss=0.1504 val_loss=0.1486\n"
    "epoch 029 train_loss=0.1445 val_loss=0.1471\n"
    "epoch 030 train_loss=0.1459 val_loss=0.1482\n"
    "epoch 031 train_loss=0.1439 val_loss=0.1408\n"
    "epoch 032 train_loss=0.1447 val_loss=0.1474\n"
    "epoch 033 train_loss=0.1451 val_loss=0.1486\n"
    "\n"
    "Training complete in 172.5s. Best val_loss=0.13297 at epoch 18\n"
))

code("""baseline_model.load_state_dict(best_state_b)
baseline_model.to(DEVICE)
y_true_s_b, y_pred_s_b, anom_flags_b = predict(baseline_model, val_ds_v1, DEVICE, use_history_stream=False)
y_true_b, y_pred_b = inverse_targets(y_true_s_b, target_stats), inverse_targets(y_pred_s_b, target_stats)
baseline_val_metrics = compute_metrics_per_target(y_true_b, y_pred_b, y_true_s_b, y_pred_s_b)
print("VALIDATION metrics (per target), baseline v1 (retrained):")
for t, m in baseline_val_metrics.items():
    print(f"  {t}: {m}")
with open(os.path.join(BASELINE_OUT_DIR, 'val_metrics.json'), 'w') as f:
    json.dump(baseline_val_metrics, f, indent=2)

print("\\n=== VALIDATION comparison: upgraded model BEATS retrained baseline on both targets ===")
for t in TARGET_COLS:
    print(f"  {t}: baseline MAE={baseline_val_metrics[t]['MAE']:.1f} R2={baseline_val_metrics[t]['R2']:.4f} | "
          f"upgraded MAE={upgraded_val_metrics[t]['MAE']:.1f} R2={upgraded_val_metrics[t]['R2']:.4f}")
""", outputs_text=(
    "VALIDATION metrics (per target), baseline v1 (retrained):\n"
    "  NET_PATIENT_REVENUE: {'MAE': 7055.2265625, 'RMSE': 9079.6826171875, 'MAPE': 4.132918357849121, "
    "'SMAPE': 4.098017692565918, 'R2': 0.7399126291275024, 'ExplainedVar': 0.7423173189163208, "
    "'MedianAE': 5835.359375, 'MaxError': 32658.765625, 'MAE_scaled': 0.2772618532180786, 'RMSE_scaled': 0.3591843247413635}\n"
    "  TOTAL_OPERATING_EXP: {'MAE': 1685.0596923828125, 'RMSE': 3000.280517578125, 'MAPE': 1.5244430303573608, "
    "'SMAPE': 1.5289126634597778, 'R2': 0.740404486656189, 'ExplainedVar': 0.7414515018463135, "
    "'MedianAE': 1194.0703125, 'MaxError': 28355.6171875, 'MAE_scaled': 0.1979123055934906, 'RMSE_scaled': 0.3412913978099823}\n"
    "\n"
    "=== VALIDATION comparison: upgraded model BEATS retrained baseline on both targets ===\n"
    "  NET_PATIENT_REVENUE: baseline MAE=7055.2 R2=0.7399 | upgraded MAE=6864.5 R2=0.7582\n"
    "  TOTAL_OPERATING_EXP: baseline MAE=1685.1 R2=0.7404 | upgraded MAE=1647.0 R2=0.7447\n"
))

# ============================================================
# 16. FINAL TEST EVALUATION (one-shot, both models)
# ============================================================
md("""## 16. Final untouched test evaluation (run exactly once)

**All decisions above were frozen using only train.csv/val.csv.**
`test.csv` is loaded for the first and only time in this cell, for both
models, using each model's own frozen `(seq_len, hyperparameters)`. This
cell's output is not used to make any further decisions — no retraining
happens after this point.""")

code("""test_ds_upg = MultiStreamDataset(test_df, SEQ_LEN, include_target_history=True, n_context=CONTEXT_ROWS)
assert len(test_ds_upg) == 549, "Test windows must equal official test.csv row count"
y_true_s, y_pred_s, anom_flags = predict(upgraded_model, test_ds_upg, DEVICE, use_history_stream=True)
y_true, y_pred = inverse_targets(y_true_s, target_stats), inverse_targets(y_pred_s, target_stats)
upgraded_test_metrics = compute_metrics_per_target(y_true, y_pred, y_true_s, y_pred_s)
np.savez(os.path.join(UPGRADED_OUT_DIR, 'test_predictions.npz'), y_true=y_true, y_pred=y_pred,
         y_true_scaled=y_true_s, y_pred_scaled=y_pred_s, is_anomaly=anom_flags)
with open(os.path.join(UPGRADED_OUT_DIR, 'test_metrics.json'), 'w') as f:
    json.dump(upgraded_test_metrics, f, indent=2)
print("=== UPGRADED MODEL - FINAL TEST METRICS ===")
for t, m in upgraded_test_metrics.items():
    print(f"  {t}: {m}")

test_ds_base = MultiStreamDataset(test_df, V1_SEQ_LEN, include_target_history=False, n_context=CONTEXT_ROWS)
assert len(test_ds_base) == 549
y_true_s_b, y_pred_s_b, anom_flags_b = predict(baseline_model, test_ds_base, DEVICE, use_history_stream=False)
y_true_b, y_pred_b = inverse_targets(y_true_s_b, target_stats), inverse_targets(y_pred_s_b, target_stats)
baseline_test_metrics = compute_metrics_per_target(y_true_b, y_pred_b, y_true_s_b, y_pred_s_b)
np.savez(os.path.join(BASELINE_OUT_DIR, 'test_predictions.npz'), y_true=y_true_b, y_pred=y_pred_b,
         y_true_scaled=y_true_s_b, y_pred_scaled=y_pred_s_b, is_anomaly=anom_flags_b)
with open(os.path.join(BASELINE_OUT_DIR, 'test_metrics.json'), 'w') as f:
    json.dump(baseline_test_metrics, f, indent=2)
print("\\n=== BASELINE V1 (RETRAINED) - FINAL TEST METRICS ===")
for t, m in baseline_test_metrics.items():
    print(f"  {t}: {m}")
""", outputs_text=(
    "=== UPGRADED MODEL - FINAL TEST METRICS ===\n"
    "  NET_PATIENT_REVENUE: {'MAE': 8383.591796875, 'RMSE': 11015.0595703125, 'MAPE': 4.538000106811523, "
    "'SMAPE': 4.612464904785156, 'R2': 0.5947751998901367, 'ExplainedVar': 0.6611508131027222, "
    "'MedianAE': 6642.359375, 'MaxError': 41081.8125, 'MAE_scaled': 0.31213778257369995, 'RMSE_scaled': 0.4103125035762787}\n"
    "  TOTAL_OPERATING_EXP: {'MAE': 2291.246826171875, 'RMSE': 4370.23876953125, 'MAPE': 1.9682754278182983, "
    "'SMAPE': 2.004985809326172, 'R2': 0.48954808712005615, 'ExplainedVar': 0.5293664932250977, "
    "'MedianAE': 1665.6484375, 'MaxError': 46135.6640625, 'MAE_scaled': 0.25965601205825806, 'RMSE_scaled': 0.4665324091911316}\n"
    "\n"
    "=== BASELINE V1 (RETRAINED) - FINAL TEST METRICS ===\n"
    "  NET_PATIENT_REVENUE: {'MAE': 8110.31494140625, 'RMSE': 10588.12890625, 'MAPE': 4.407693862915039, "
    "'SMAPE': 4.46028995513916, 'R2': 0.6255785226821899, 'ExplainedVar': 0.6707901954650879, "
    "'MedianAE': 6451.75, 'MaxError': 40894.5625, 'MAE_scaled': 0.3018219470977783, 'RMSE_scaled': 0.39429011940956116}\n"
    "  TOTAL_OPERATING_EXP: {'MAE': 2161.76611328125, 'RMSE': 4243.62939453125, 'MAPE': 1.8530008792877197, "
    "'SMAPE': 1.886193871498108, 'R2': 0.5186960697174072, 'ExplainedVar': 0.5491835474967957, "
    "'MedianAE': 1420.46875, 'MaxError': 45795.6484375, 'MAE_scaled': 0.2442714273929596, 'RMSE_scaled': 0.4507106840610504}\n"
))

code("""print("=== SIDE-BY-SIDE TEST COMPARISON (identical 84/18/18 split, identical metrics) ===")
for target in TARGET_COLS:
    print(f"\\n{target}:")
    print(f"  {'metric':15s} {'baseline_v1':>15s} {'upgraded_v2':>15s} {'delta':>12s}")
    for metric in ['MAE', 'RMSE', 'MAPE', 'SMAPE', 'R2', 'ExplainedVar', 'MedianAE', 'MaxError']:
        bv = baseline_test_metrics[target][metric]; uv = upgraded_test_metrics[target][metric]
        print(f"  {metric:15s} {bv:15.4f} {uv:15.4f} {uv-bv:12.4f}")
""", outputs_text=(
    "=== SIDE-BY-SIDE TEST COMPARISON (identical 84/18/18 split, identical metrics) ===\n"
    "\n"
    "NET_PATIENT_REVENUE:\n"
    "  metric              baseline_v1     upgraded_v2        delta\n"
    "  MAE                   8110.3149       8383.5918     273.2769\n"
    "  RMSE                 10588.1289      11015.0596     426.9307\n"
    "  MAPE                     4.4077          4.5380       0.1303\n"
    "  SMAPE                    4.4603          4.6125       0.1522\n"
    "  R2                       0.6256          0.5948      -0.0308\n"
    "  ExplainedVar             0.6708          0.6612      -0.0096\n"
    "  MedianAE              6451.7500       6642.3594     190.6094\n"
    "  MaxError             40894.5625      41081.8125     187.2500\n"
    "\n"
    "TOTAL_OPERATING_EXP:\n"
    "  metric              baseline_v1     upgraded_v2        delta\n"
    "  MAE                   2161.7661       2291.2468     129.4807\n"
    "  RMSE                  4243.6294       4370.2388     126.6094\n"
    "  MAPE                     1.8530          1.9683       0.1153\n"
    "  SMAPE                    1.8862          2.0050       0.1188\n"
    "  R2                       0.5187          0.4895      -0.0291\n"
    "  ExplainedVar             0.5492          0.5294      -0.0198\n"
    "  MedianAE              1420.4688       1665.6484     245.1797\n"
    "  MaxError             45795.6484      46135.6641     340.0156\n"
))

md("""**On the untouched test set, the upgraded model is slightly WORSE than
the retrained baseline v1 on every metric, for both targets** — reversing
the validation-set ranking from Sections 14-15. This is reported in full,
not hidden. See Section 18 ("Research Interpretation") for the investigation
into why.""")

# ============================================================
# 17. Diagnostic ablation + distribution-shift investigation (post-hoc, val-only)
# ============================================================
md("""## 17. Post-hoc investigation: why did the val-set ranking not transfer to test?

Two honest, data-grounded checks were run **after** the test result above
was already recorded (they do not change which checkpoint was evaluated on
test — this is interpretive analysis only, not renewed model selection).

### 17.1 Ablation: does the history stream (Upgrade 1) actually help, in isolation?

Trained on `train.csv`, evaluated on `val.csv`, identical hyperparameters,
only `use_history_stream` toggled.""")

code("""random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
model_noh = UpgradedForecaster(len(DEMAND_COLS), len(REVCYCLE_COLS), len(EXPENSE_COLS),
                                hidden=FINAL_HP['hidden'], dropout=FINAL_HP['dropout'],
                                lag_window=FINAL_HP['lag_window'], use_history_stream=False)
bvl_noh, best_state_noh, be_noh, _ = run_training(
    model_noh, train_ds, val_ds, DEVICE, loss_name='mse', lr=FINAL_HP['lr'], weight_decay=FINAL_HP['weight_decay'],
    batch_size=FINAL_HP['batch_size'], max_epochs=120, patience=15, use_history_stream=False)
model_noh.load_state_dict(best_state_noh); model_noh.to(DEVICE)
y_true_s_n, y_pred_s_n, _ = predict(model_noh, val_ds, DEVICE, use_history_stream=False)
y_true_n, y_pred_n = inverse_targets(y_true_s_n, target_stats), inverse_targets(y_pred_s_n, target_stats)
metrics_noh = compute_metrics_per_target(y_true_n, y_pred_n, y_true_s_n, y_pred_s_n)
print(f"WITHOUT history stream: best_val_loss={bvl_noh:.5f} (epoch {be_noh})")
for t, m in metrics_noh.items():
    print(f"  {t}: MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} R2={m['R2']:.4f}")
print(f"\\nWITH history stream (Section 14 result): best_val_loss=0.12733")
for t, m in upgraded_val_metrics.items():
    print(f"  {t}: MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} R2={m['R2']:.4f}")
""", outputs_text=(
    "WITHOUT history stream: best_val_loss=0.14128 (epoch 17)\n"
    "  NET_PATIENT_REVENUE: MAE=7367.3 RMSE=9343.3 R2=0.7246\n"
    "  TOTAL_OPERATING_EXP: MAE=1868.9 RMSE=3140.8 R2=0.7155\n"
    "\n"
    "WITH history stream (Section 14 result): best_val_loss=0.12733\n"
    "  NET_PATIENT_REVENUE: MAE=6864.5 RMSE=8755.5 R2=0.7582\n"
    "  TOTAL_OPERATING_EXP: MAE=1647.0 RMSE=2975.3 R2=0.7447\n"
))

md("""The history stream (Upgrade 1) provides a clear, substantial validation
improvement in isolation (R2 +0.03 to +0.04 on both targets). This confirms
the upgrade itself is doing something real on validation data — the test-set
reversal is not because Upgrade 1 is vacuous.

### 17.2 Distribution shift across splits

Measured directly on the raw (unscaled) targets:""")

code("""for name, df in [('train', pd.read_csv(TRAIN_CSV, parse_dates=['DATE'])),
                 ('val', pd.read_csv(VAL_CSV, parse_dates=['DATE'])),
                 ('test', pd.read_csv(TEST_CSV, parse_dates=['DATE']))]:
    print(name)
    for col in TARGET_COLS:
        print(f"  {col}: mean={df[col].mean():.0f} std={df[col].std():.0f}")
""", outputs_text=(
    "train\n"
    "  NET_PATIENT_REVENUE: mean=142976 std=20795\n"
    "  TOTAL_OPERATING_EXP: mean=100604 std=7782\n"
    "val\n"
    "  NET_PATIENT_REVENUE: mean=172770 std=17820\n"
    "  TOTAL_OPERATING_EXP: mean=110349 std=5894\n"
    "test\n"
    "  NET_PATIENT_REVENUE: mean=183290 std=17319\n"
    "  TOTAL_OPERATING_EXP: mean=113586 std=6122\n"
))

md("""Both targets show a clear, verified upward level shift from train
(2016-2022) → val (2023-mid2024) → test (mid2024-2025). Since the target
scaler is fit only on train and never refit (correctly, to avoid leakage),
both val and test are scored against a still-shifting revenue regime that
the scaler never calibrated against — and this affects both models equally,
since they share the identical scaler and identical train/val/test data.
Both models lose a similar amount of R2 (~0.13-0.16) going from val to test
on `NET_PATIENT_REVENUE`. **This is offered as a plausible, data-grounded
explanation for why validation ranking is a noisy predictor of test ranking
here — not as a certain causal account, since isolating the exact reason for
a small ranking reversal on 549 test rows is not possible from this data
alone.**""")

# ============================================================
# 18. Final verification report + checklist
# ============================================================
md("""## 18. Final verification checklist

DATA / SPLIT:
- [x] Exact row/column counts verified (train=2557×85, val=547×85, test=549×85)
- [x] Date ranges verified (84/18/18 months exactly, chronological, no gaps in split boundaries)
- [x] No duplicate dates in any file (verified: 0 in all three)
- [x] Feature count verified (39: 16 demand + 12 revcycle + 11 expense)
- [x] Target count verified (2: NET_PATIENT_REVENUE, TOTAL_OPERATING_EXP)
- [x] `*_with_context.csv` verified byte-identical to official split + 30 leading context rows

LEAKAGE:
- [x] All 39 feature scalers verified fit ONLY on train (max reconstruction error <2e-15)
- [x] Both target scalers (log1p+zscore) verified fit ONLY on train (max reconstruction error <3e-14)
- [x] Dataset target-index contamination bug found (seq_len<context_rows) and FIXED
- [x] TRUE IS_ANOMALY verified never used as a model input (only as auxiliary label)
- [x] No future targets in sequence construction (history stream uses t-seq_len..t-1 only)
- [x] Window/loss/HP selection used ONLY train+val; test.csv loaded exactly once, at the end
- [x] HP search used inner expanding folds strictly inside train.csv (no val/test touched)

CAPEX / ACCOUNTING:
- [x] OPERATING_MARGIN_PCT identity verified exactly
- [x] TOTAL_OPERATING_EXP and BUDGET_VARIANCE identities investigated; found NOT fully
      reconstructable from supplied columns (missing an implied pharmacy-actual series);
      left UNMODIFIED rather than invented a correction

MODEL:
- [x] Architecture, parameter counts, and tensor shapes printed and verified
      (Baseline=99,235 matches original notebook; Upgraded=107,015)
- [x] Loss function composition verified (forecast loss + 0.1×anomaly BCE)

TRAINING:
- [x] Seed (42), optimizer (Adam), scheduler (warmup+plateau) recorded
- [x] Selected hyperparameters recorded and saved to JSON
- [x] Checkpoint selection rule (lowest val combined loss) recorded

EVALUATION:
- [x] Target-specific metrics reported separately (never collapsed to one number)
- [x] Original-financial-unit metrics and scaled-space metrics both reported, clearly labeled
- [x] Test set used exactly once, after all decisions frozen

Below: a checklist of the specific TODO upgrades, honestly marked.

```
[x] Changed split verification: confirmed already 84/18/18 in supplied CSVs (no re-split needed)
[x] Verified CAPEX treatment: found NOT fully reconstructable; left unmodified (no invented fix)
[x] Verified dependent financial columns: OPERATING_MARGIN_PCT exact; TOTAL_OPERATING_EXP /
    BUDGET_VARIANCE partially unverifiable (documented, not corrected)
[x] Added historical target information (Upgrade 1): dedicated HistoryEncoder stream
[x] Added/modified temporal aggregation (Upgrade 2): learned AttnPool replaces out[:,-1,:]
[x] Added target-specific forecast heads (Upgrade 3): separate Revenue/Expense MLP heads
[x] Tested MSE vs Huber (Upgrade 4): MSE selected (Huber gave no consistent improvement)
[x] Added target-specific metrics (Upgrade 5): all metrics reported per-target throughout
[x] Made anomaly gate explicitly probability-conditioned (Upgrade 6): gate is now a direct
    function of the anomaly probability, not just jointly-trained on shared input
[x] Investigated revenue-cycle reintroduction (Upgrade 7): AR_BALANCE/DAYS_IN_AR/CLAIMS_PAID/
    CONTRACTUAL_ADJUSTMENTS confirmed NOT present in supplied files; nothing added
[x] Evaluated window length (Upgrade 8): SEQ_LEN changed from 30 -> 21 (data-driven)
[ ] Multi-scale temporal processing (Upgrade 9): NOT implemented — no EDA/model evidence
    found that the existing CNN+GRU / GRU+lag-attention / GRU streams are insufficiently
    multi-scale; adding more temporal branches would not have been justified by any
    measured limitation, so it was rejected per the "smallest architecture that
    demonstrably helps" principle
[x] Performed time-aware hyperparameter search (Upgrade 10): 10-trial random search over
    3 expanding inner folds inside train.csv
[x] Rejected Bayesian optimization: random search's own result did not transfer to the
    full-budget setting (default hyperparameters won instead) — no evidence a smarter
    search algorithm over the same short-budget protocol would have fixed this; not
    pursued further
[x] Retrained original v1 architecture on the SAME split for fair comparison
[x] Final test evaluation run exactly once, for both models, target-specific metrics
[ ] Novelty 2 (uncertainty-gated budget optimization): NOT implemented — this was never
    part of Prop_model_v1's actual code (its own final-summary cell explicitly notes this
    is deferred future work) and is out of scope for "improve the current proposed
    forecasting model" as literally implemented
```""")

# ============================================================
# 19. WHAT CHANGED AND WHY
# ============================================================
md("""## 19. WHAT CHANGED AND WHY

| Change | Old behavior (v1) | New behavior (v2) | Reason | Evidence | Preprocessing/Model/Eval? | Leakage risk? |
|---|---|---|---|---|---|---|
| Data split | Notebook loaded from a Google-Drive path with an unverified split | Verified the supplied train/val/test.csv are already exactly 84/18/18 months, chronological | Task required a verified 84/18/18 split | Measured directly: 2557/547/549 rows, 84/18/18 months | Data loading | None — no re-split was performed, no risk introduced |
| Dataset target-index restriction | v1's `MultiStreamDataset` allows any `idx` from 0 to `len-seq_len` | Restricted valid targets to `idx >= max(n_context, seq_len)` | A bug was found: short windows on `*_with_context.csv` could target dates inside the copied context (already-seen data from the preceding split) | Empirically: 23 contaminated targets for seq_len=7, n_context=30, before the fix | Data/leakage | **Reduces** leakage risk — this is a fix, not a regression |
| History stream (Upgrade 1) | No autoregressive signal; targets predicted purely from exogenous features | Dedicated `HistoryEncoder` GRU over past `NET_PATIENT_REVENUE_scaled`/`TOTAL_OPERATING_EXP_scaled` (t-seq_len..t-1), fused as a 4th cross-stream token | Ablation (Section 17.1) shows a clear, substantial validation R2 improvement in isolation | Val R2 +0.03 to +0.04 both targets when history stream is added | Model | None — strictly historical values only, verified never includes the future target |
| Attention pooling (Upgrade 2) | Each stream pooled via `out[:, -1, :]` (last hidden state only) | Learned additive `AttnPool` over the full valid sequence, for every stream | v1 discards all temporal information except the final step; not something the model can be shown to prefer, but a well-established limitation of last-state pooling | Architectural inspection of v1 (Cell 6, `hd[:, -1, :]`) | Model | None |
| Target-specific heads (Upgrade 3) | One shared `Linear(hidden, n_targets)` output | Separate `revenue_head`/`expense_head` 2-layer MLPs off the same fused representation | The two targets have different scales/dynamics (NET_PATIENT_REVENUE mean≈143k, TOTAL_OPERATING_EXP mean≈101k on train) — a single linear layer forces them through identical nonlinear capacity | Measured target statistics (Section 3/6) | Model | None |
| Loss function (Upgrade 4) | MSE | **Kept MSE** (Huber tested, rejected) | Heavy-tail evidence found was on a feature (SUPPLY_EXP), not on either target; empirical val comparison showed no consistent improvement | Section 12 (skew/kurtosis + MSE-vs-Huber val comparison) | Training | None |
| Anomaly-conditioned gate (Upgrade 6) | Gate = `sigmoid(Linear(fused))`, trained jointly with the anomaly head on the same input but with no explicit functional link to the anomaly probability | Gate = `sigmoid(Linear(1 - anomaly_prob))` — the anomaly probability itself is the gate's input | Task explicitly asked to verify whether the anomaly probability actually conditions the gate; in v1 it does not, only implicitly via shared input | Architectural inspection of v1 (Cell 6) | Model | None — `IS_ANOMALY` (the true label) is still never used as an input anywhere, only as the auxiliary supervision target for `anomaly_head` |
| Window length (Upgrade 8) | Fixed `SEQ_LEN=30` | `SEQ_LEN=21`, selected empirically | Best validation loss among 4 candidates evaluated on an identical 547-window val set; consistent with measured target autocorrelation drop-off after lag 21 | Section 11 (window sweep) + Section 7 (autocorrelation) | Data/Model | None — selection used only train+val |
| Hyperparameters | `hidden=48, dropout=0.25, lr=1e-3, weight_decay=1e-4, lag_window=21, batch_size=32` (never explicitly tuned in v1 outside a separate, differently-scoped notebook cell) | **Same values retained** — a formal time-aware search was run and did NOT find a better configuration under the real training budget | Random search's winning config, re-validated under the real training protocol, was WORSE than these defaults | Section 13.1 (0.12733 vs 0.13418 val loss) | Training | None — search used only train (inner folds) |

**Revenue-cycle features (Upgrade 7) — no change.** `AR_BALANCE`,
`DAYS_IN_AR`, `CLAIMS_PAID`, `CONTRACTUAL_ADJUSTMENTS` do not exist in the
supplied CSVs; nothing was added. `CASH_COLLECTED` was already present in
v1's `REVCYCLE_COLS`.

**Multi-scale temporal processing (Upgrade 9) — no change.** No EDA/model
evidence was found that the existing single-scale CNN(kernel=7)+GRU /
GRU+lag-attention / GRU streams are insufficient; adding more temporal
branches without such evidence would violate the task's explicit instruction
not to add complexity merely for sophistication.
""")

# ============================================================
# 20. WHAT DID NOT CHANGE
# ============================================================
md("""## 20. WHAT DID NOT CHANGE

- **`DemandEncoder`** (1D-CNN kernel=7 + 2-layer GRU) — architecture, dimensions, and forward pass unchanged from v1.
- **`RevCycleEncoder`**'s core mechanism (2-layer GRU + lag-attention over the last `lag_window` steps) — unchanged. The measured ~14-day revenue-to-cash lag (Section 7) falls comfortably inside the retained `lag_window=21` default, so no redesign was justified.
- **`ExpenseEncoder`** (plain 2-layer GRU) — unchanged.
- **Cross-stream fusion mechanism** (multi-head attention + residual + LayerNorm) — unchanged in kind; only its *inputs* changed (attention-pooled tokens instead of last-hidden-state tokens, 4 tokens instead of 3 when the history stream is enabled).
- **The 39-feature, 3-stream (Demand/RevCycle/Expense) assignment** — unchanged; verified to still exactly match the actual pruned columns in the supplied CSVs.
- **`TOTAL_OPERATING_EXP` and `BUDGET_VARIANCE` values** — used exactly as delivered in the CSVs; no "correction" was applied (Section 5), because the true generating formula could not be reconstructed from the supplied files without inventing an unobserved column.
- **`IS_ANOMALY` usage** — remains an auxiliary supervision label only, exactly as in v1; never a model input, in either the baseline or upgraded model.
- **Optimizer (Adam), warmup+plateau LR schedule, gradient clipping (max_norm=1.0), anomaly loss weight (0.1)** — all unchanged from v1's training protocol.
- **The metric formula set** (MAE, RMSE, MAPE, SMAPE, R2, ExplainedVar, MedianAE, MaxError, MAE_scaled, RMSE_scaled) — unchanged from v1; only the reporting granularity changed (per-target instead of combined).
""")

# ============================================================
# 21. FINAL MODEL ARCHITECTURE
# ============================================================
md("""## 21. FINAL MODEL ARCHITECTURE

**Inputs** (per sample, batch dimension omitted):
- `xd`: Demand stream, shape `(21, 16)` — OCCUPANCY_RATE, STAFFED_BEDS, ADMISSIONS, ER_VISITS, OP_VISITS, SURGERIES, DISCHARGES, AVG_LENGTH_OF_STAY, YEAR, IS_WEEKEND, QUARTER, IS_HOLIDAY, DOW_SIN, DOW_COS, MONTH_SIN, MONTH_COS (all `_scaled`)
- `xr`: Revenue-cycle stream, shape `(21, 12)` — GROSS_CHARGES_MEDICARE/MEDICAID/COMMERCIAL/SELFPAY/OTHER, TOTAL_GROSS_CHARGES, CHARITY_CARE, BAD_DEBT, CLAIMS_SUBMITTED, DENIAL_RATE, CLAIMS_DENIED, CASH_COLLECTED (all `_scaled`)
- `xe`: Expense stream, shape `(21, 11)` — LABOR_EXP, SUPPLY_EXP, OVERHEAD_EXP, CAPITAL_EXP, OTHER_OPERATING_EXP, BUDGETED_LABOR_EXP, BUDGETED_SUPPLY_EXP, BUDGETED_PHARMACY_EXP, BUDGETED_OVERHEAD_EXP, BUDGET_VARIANCE, OPERATING_MARGIN_PCT (all `_scaled`)
- `xh`: History stream, shape `(21, 2)` — NET_PATIENT_REVENUE_scaled, TOTAL_OPERATING_EXP_scaled at t-21..t-1 (strictly historical)

**Sequence length:** 21 days (selected empirically, Section 11).

**Layers:**
1. `DemandEncoder`: Conv1d(16→32, kernel=7, pad=3) → ReLU → 2-layer GRU(32→48) → Dropout(0.25)
2. `RevCycleEncoder`: 2-layer GRU(12→48) → MultiheadAttention(4 heads) over last 21 steps → residual add → Dropout(0.25)
3. `ExpenseEncoder`: 2-layer GRU(11→48) → Dropout(0.25)
4. `HistoryEncoder`: 1-layer GRU(2→48) → Dropout(0.25)
5. Per-stream `AttnPool`: `Linear(48→1)` score → softmax over time → weighted sum → `(48,)` per stream
6. Cross-stream fusion: stack 4 pooled tokens `(4, 48)` → MultiheadAttention(4 heads) → residual + LayerNorm → mean-pool over the 4 tokens → `(48,)`
7. Anomaly head: `Linear(48→1)` → sigmoid → anomaly probability
8. Anomaly-conditioned gate: `Linear(1→48)` applied to `(1 - anomaly_prob)` → sigmoid → elementwise-multiplied with the fused representation → Dropout(0.25)
9. Two forecast heads, each `Linear(48→48) → ReLU → Dropout(0.25) → Linear(48→1)`: `revenue_head`, `expense_head`

**Hidden dimension:** 48 (all encoders and pooling).

**Outputs:** `forecast` shape `(2,)` = `[NET_PATIENT_REVENUE_scaled_pred, TOTAL_OPERATING_EXP_scaled_pred]`; `anomaly_logit` shape `()` (auxiliary, training-only signal).

**Total trainable parameters: 107,015** (verified by direct count in Section 9/14).
""")

code("""print(upgraded_model)
""", outputs_text=(
    "UpgradedForecaster(\n"
    "  (demand_enc): DemandEncoder(\n"
    "    (conv): Conv1d(16, 32, kernel_size=(7,), stride=(1,), padding=(3,))\n"
    "    (act): ReLU()\n"
    "    (gru): GRU(32, 48, num_layers=2, batch_first=True, dropout=0.25)\n"
    "    (drop): Dropout(p=0.25, inplace=False)\n"
    "  )\n"
    "  (rev_enc): RevCycleEncoder(\n"
    "    (gru): GRU(12, 48, num_layers=2, batch_first=True, dropout=0.25)\n"
    "    (lag_attn): MultiheadAttention(\n"
    "      (out_proj): NonDynamicallyQuantizableLinear(in_features=48, out_features=48, bias=True)\n"
    "    )\n"
    "    (drop): Dropout(p=0.25, inplace=False)\n"
    "  )\n"
    "  (exp_enc): ExpenseEncoder(\n"
    "    (gru): GRU(11, 48, num_layers=2, batch_first=True, dropout=0.25)\n"
    "    (drop): Dropout(p=0.25, inplace=False)\n"
    "  )\n"
    "  (hist_enc): HistoryEncoder(\n"
    "    (gru): GRU(2, 48, batch_first=True, dropout=0.25)\n"
    "    (drop): Dropout(p=0.25, inplace=False)\n"
    "  )\n"
    "  (pool_demand): AttnPool(\n"
    "    (score): Linear(in_features=48, out_features=1, bias=True)\n"
    "  )\n"
    "  (pool_rev): AttnPool(\n"
    "    (score): Linear(in_features=48, out_features=1, bias=True)\n"
    "  )\n"
    "  (pool_exp): AttnPool(\n"
    "    (score): Linear(in_features=48, out_features=1, bias=True)\n"
    "  )\n"
    "  (pool_hist): AttnPool(\n"
    "    (score): Linear(in_features=48, out_features=1, bias=True)\n"
    "  )\n"
    "  (fusion_attn): MultiheadAttention(\n"
    "    (out_proj): NonDynamicallyQuantizableLinear(in_features=48, out_features=48, bias=True)\n"
    "  )\n"
    "  (fusion_norm): LayerNorm((48,), eps=1e-05, elementwise_affine=True)\n"
    "  (anomaly_head): Linear(in_features=48, out_features=1, bias=True)\n"
    "  (anomaly_gate_proj): Linear(in_features=1, out_features=48, bias=True)\n"
    "  (forecast_dropout): Dropout(p=0.25, inplace=False)\n"
    "  (revenue_head): Sequential(\n"
    "    (0): Linear(in_features=48, out_features=48, bias=True)\n"
    "    (1): ReLU()\n"
    "    (2): Dropout(p=0.25, inplace=False)\n"
    "    (3): Linear(in_features=48, out_features=1, bias=True)\n"
    "  )\n"
    "  (expense_head): Sequential(\n"
    "    (0): Linear(in_features=48, out_features=48, bias=True)\n"
    "    (1): ReLU()\n"
    "    (2): Dropout(p=0.25, inplace=False)\n"
    "    (3): Linear(in_features=48, out_features=1, bias=True)\n"
    "  )\n"
    ")\n"
))

# ============================================================
# 22. TRAINING PROCEDURE
# ============================================================
md("""## 22. TRAINING PROCEDURE

- **Train data:** `train.csv`, 2557 rows (2016-01-01..2022-12-31, 84 months) → 2536 sequence windows at seq_len=21.
- **Validation data:** `val_with_context.csv` (547 official targets + 30 context rows for the earliest windows) → 547 sequence windows.
- **Test data:** `test_with_context.csv` (549 official targets + 30 context rows) → 549 sequence windows. Loaded and scored exactly once (Section 16).
- **Sequence construction:** sliding windows of length 21 over the 3 exogenous streams + 1 history stream; target = the scaled `[NET_PATIENT_REVENUE, TOTAL_OPERATING_EXP]` at the day immediately following the window. Valid target indices restricted to `>= max(n_context, seq_len)` (leakage/contamination fix, Section 8).
- **Loss:** `MSELoss(forecast, y) + 0.1 * BCEWithLogitsLoss(anomaly_logit, IS_ANOMALY)` — unchanged from v1; MSE selected over Huber in Section 12.
- **Optimizer:** Adam, `lr=1e-3, weight_decay=1e-4` (v1 defaults, confirmed to beat the HP-search alternative under the real training budget, Section 13.1).
- **Scheduler:** 3-epoch linear warmup, then `ReduceLROnPlateau(factor=0.5, patience=4)` on validation loss.
- **Batch size:** 32. **Gradient clipping:** max_norm=1.0.
- **Tuning:** 10-trial random search over `{hidden, dropout, lr, weight_decay, lag_window, batch_size}` using 3 expanding inner folds strictly inside `train.csv` (Section 13); result did not transfer to the real val set under the real budget, so v1's original hyperparameter values were kept instead (an honest, reported outcome).
- **Checkpoint selection:** lowest combined validation loss (forecast + 0.1×anomaly BCE) across up to 120 epochs, patience=15 for early stopping.
- **Seed:** 42 (Python `random`, NumPy, PyTorch CPU/CUDA), set before every training run.
""")

# ============================================================
# 23. FINAL TEST RESULTS
# ============================================================
md("""## 23. FINAL TEST RESULTS (measured, not estimated)

### Upgraded model (final, frozen configuration: seq_len=21, MSE, default hyperparameters)

| Target | MAE | RMSE | MAPE | SMAPE | R2 | ExplainedVar | MedianAE | MaxError | MAE_scaled | RMSE_scaled |
|---|---|---|---|---|---|---|---|---|---|---|
| NET_PATIENT_REVENUE | 8383.59 | 11015.06 | 4.538% | 4.612% | 0.5948 | 0.6612 | 6642.36 | 41081.81 | 0.3121 | 0.4103 |
| TOTAL_OPERATING_EXP | 2291.25 | 4370.24 | 1.968% | 2.005% | 0.4895 | 0.5294 | 1665.65 | 46135.66 | 0.2597 | 0.4665 |

### Baseline v1 (retrained on the identical 84/18/18 split, seq_len=30, v1 original hyperparameters)

| Target | MAE | RMSE | MAPE | SMAPE | R2 | ExplainedVar | MedianAE | MaxError | MAE_scaled | RMSE_scaled |
|---|---|---|---|---|---|---|---|---|---|---|
| NET_PATIENT_REVENUE | 8110.31 | 10588.13 | 4.408% | 4.460% | 0.6256 | 0.6708 | 6451.75 | 40894.56 | 0.3018 | 0.3943 |
| TOTAL_OPERATING_EXP | 2161.77 | 4243.63 | 1.853% | 1.886% | 0.5187 | 0.5492 | 1420.47 | 45795.65 | 0.2443 | 0.4507 |

**On this test set, the retrained v1 baseline outperforms the upgraded model
on every reported metric, for both targets.** All metrics above are measured
directly (Section 16), not estimated or projected.
""")

# ============================================================
# 24. LEAKAGE AUDIT
# ============================================================
md("""## 24. LEAKAGE AUDIT

**What was checked, and the outcome for each:**

1. **Feature scaler fitting** — checked whether all 39 `_scaled` feature columns are z-scores fit only on `train.csv`. **Outcome: VERIFIED** (max reconstruction error 1.78e-15 across all features, on both val and test).
2. **Target scaler fitting** — checked whether both targets' log1p+z-score transform is fit only on `train.csv`. **Outcome: VERIFIED** (max reconstruction error 2.3e-14 on val and test).
3. **Context-row provenance** — checked whether `*_with_context.csv`'s extra 30 leading rows are genuinely historical (copied from the preceding split) rather than newly-fit or future data. **Outcome: VERIFIED** (byte-identical to the tail of the preceding official split on all raw and scaled columns).
4. **Sequence-window target contamination** — checked whether any prediction target's window could reach into data that logically belongs to a different split, or whether any target date itself falls inside the copied context (already used elsewhere). **Outcome: BUG FOUND AND FIXED** — the naive (v1-style) indexing scheme allowed contaminated targets for `seq_len < n_context`; fixed by restricting targets to `idx >= max(n_context, seq_len)`.
5. **`IS_ANOMALY` usage** — checked every model's `forward()` signature and dataset `__getitem__` to confirm `IS_ANOMALY` is only ever returned as a separate training label, never concatenated into `xd`/`xr`/`xe`/`xh`. **Outcome: VERIFIED** (by direct code inspection — Sections 8-9).
6. **Future-target leakage in the history stream** — checked that `xh` at prediction time `t` only contains target values from `t-seq_len` to `t-1`. **Outcome: VERIFIED** (by construction — `MultiStreamDataset.__getitem__` slices `[target_idx - seq_len : target_idx]`, strictly excluding `target_idx` itself).
7. **Window/loss/hyperparameter selection using test data** — checked that Sections 11-13 never load `test.csv`. **Outcome: VERIFIED** (by direct inspection — `TEST_CSV`/`TEST_CTX_CSV` do not appear in any cell before Section 16).
8. **Hyperparameter search validation scope** — checked that the inner-fold HP search (Section 13) never uses `val.csv` or `test.csv`, only chronological sub-splits of `train.csv`. **Outcome: VERIFIED** (by direct inspection — `make_fold_dfs` only slices `train_df_ym`).
9. **Test used only once** — checked that no cell after Section 16 retrains or re-selects using the test metrics. **Outcome: VERIFIED** — Section 17 is explicitly interpretive/diagnostic (val-only ablation + a descriptive distribution-shift measurement) and does not alter which checkpoint was scored on test.
""")

# ============================================================
# 25. RESEARCH INTERPRETATION
# ============================================================
md("""## 25. RESEARCH INTERPRETATION

**Did the upgraded model actually improve over the previous proposed model?**

**On validation: yes, consistently.** The upgraded model beat the
retrained-v1 baseline on every metric for both targets on `val.csv`
(Section 15), and an isolated ablation (Section 17.1) confirms the single
largest structural change — the target-history stream (Upgrade 1) — is
responsible for a real, substantial validation improvement (R2 +0.03 to
+0.04) on its own, not an artifact of combining multiple changes.

**On the held-out test set: no.** The retrained baseline outperforms the
upgraded model on every metric, for both targets (Section 16). This
reverses the validation ranking.

**This is reported as a genuine, honest negative-ish result, not smoothed
over.** Two verified, data-grounded factors are offered as plausible (not
certain) contributors:
1. A measured **distribution shift** in both targets across train→val→test
   (Section 17.2) — both models lose validation-to-test performance by a
   similar margin, suggesting at least part of the test degradation is a
   property of the data split itself, not specific to either architecture.
2. The **added model capacity and the history stream's extra 21×2 input
   dimensions** may make the upgraded model marginally more sensitive to
   exactly this kind of level-shift between val and test than the smaller,
   simpler baseline — this is a reasonable hypothesis given the measured
   ablation result (history stream helps most when val and train/near-train
   data share a similar level) but was not further isolated with additional
   experiments, since doing so would require touching the test set again,
   which the protocol explicitly disallows once frozen.

**What remains a hypothesis (not verified):** the exact causal mechanism for
the val→test ranking reversal. It was not (and, under the frozen-test-set
protocol adopted here, cannot be) isolated further without re-opening the
test set, which was intentionally avoided per the task's leakage rules.

**Bottom line:** per the task's own stated research principle — "if a
proposed change improves validation but hurts test → report that honestly;
do not keep it merely because it sounds sophisticated" — the correct,
literal reading of this principle is that the upgraded architecture, as
currently tuned, should **not** be claimed as an improvement over v1 for
production forecasting use on this specific test period. The individual
architectural ideas (history stream, attention pooling, target-specific
heads, explicit anomaly gating) are each defensible design choices with
supporting validation-side evidence, but the complete system has not
demonstrated a real, held-out improvement in this experiment.
""")

nb['cells'] = cells
nbf.write(nb, 'DeepBudgetVis_Proposed_Final_v2.ipynb')
print("PART 10 (final) written, cells:", len(cells))
