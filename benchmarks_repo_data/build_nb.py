"""Builds DeepBudgetVis_Benchmarks_RepoData.ipynb against the ACTUAL repo CSVs."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(s):
    c = nbf.v4.new_code_cell(s); c['outputs'] = []; c['execution_count'] = None
    cells.append(c)


md(r"""# DeepBudget-Vis — Literature-Based Benchmark Models

**CNN-LSTM · TCN-LSTM · LSTM-mTrans-MLP**

Built and executed against the **actual dataset in this repository** (branch
`Suren-1`): `train.csv`, `val.csv`, `test.csv`, `val_with_context.csv`,
`test_with_context.csv`.

These are **literature-based benchmark models** (equivalently, *literature-based
SOTA comparison models*) for comparison against the proposed DeepBudget-Vis /
HRC-DRGNet model. They are **not** claimed to constitute a universal
state-of-the-art ranking — the literature survey does not establish one.

---

## ⚠️ Target-name discrepancy — read this before citing anything

The task specification asks for targets `NET_PATIENT_REVENUE` and
`TOT_OVERALL_EXP`.

**`TOT_OVERALL_EXP` does not exist in this dataset.** Verified by direct column
inspection (Section 3). The dataset's second financial target is
**`TOTAL_OPERATING_EXP`**, and there is no column containing the substring
`OVERALL` at all.

This notebook therefore uses **`TOTAL_OPERATING_EXP`** and reports it under that
real name. It is treated as the intended second target (an explicit, documented
alias mapping — not a silent substitution). If `TOT_OVERALL_EXP` is a genuinely
different quantity that exists in a newer preprocessing run, this notebook is
**not** using it, and the mapping in the configuration cell must be corrected.

---

## Fair-comparison contract

All three models share: identical chronological split, identical sequence
construction, identical input features, identical target definitions and
scaling, identical loss (`SmoothL1Loss`), identical optimizer policy (`AdamW`),
identical `ReduceLROnPlateau` and early-stopping policy, and the same seed.
**Only the architecture differs.**
""")

md(r"""## Source verification note

I verified that both cited works exist and confirmed their **high-level model
composition**. I could **not** retrieve their implementation-level details
(layer counts, channel widths, hidden sizes, learning rates) — the publisher
pages returned access errors (MDPI `HTTP 403`; the other source failed TLS
certificate verification).

**Verified:**

- Kabir et al., *"LSTM–Transformer-Based Robust Hybrid Deep Learning Model for
  Financial Time Series Forecasting"*, **Sci** 7(1), 7 (2025). The abstract
  states the work proposes **LSTM-mTrans-MLP**, integrating an LSTM network, a
  **modified Transformer** network, and a multilayered perceptron.
  [mdpi.com/2413-4155/7/1/7](https://www.mdpi.com/2413-4155/7/1/7)
- A closely-matching hybrid multivariate-forecasting work compares exactly
  **CNN-LSTM, CNN-BiLSTM, TCN-LSTM, TCN-BiLSTM**, reporting TCN-BiLSTM best
  overall on its own two datasets (Traffic Volume R² ≈ 0.976, Air Quality
  R² ≈ 0.94). [d-nb.info/1353813266/34](https://d-nb.info/1353813266/34)

**Consequence:** every numeric architecture choice below (channels, kernel
sizes, dilation schedule, hidden sizes, head counts, MLP widths, dropout), the
shared loss, the optimizer and the scheduler settings are **implementation
assumptions**, labelled as such per model. None is presented as a paper value.

Those R² figures belong to **traffic / air-quality** data. They are not
comparable to, and say nothing about, results on hospital financial data.

*Source content paraphrased for licensing compliance.*
""")

# ---------------- config ----------------
md(r"""## 1. USER CONFIGURATION — the only cell you should need to edit""")

code(r'''# ============================================================
# USER CONFIGURATION - EDIT THIS SECTION ONLY
# ============================================================
from pathlib import Path

# Repo root containing train.csv / val.csv / test.csv
BASE_DIR   = "."
DATA_DIR   = BASE_DIR
OUTPUT_DIR = f"{BASE_DIR}/benchmarks_repo_data/baseline_outputs_v1"

# Actual target columns in THIS dataset, in fixed output order.
#   output[:, 0] -> TARGET_COLUMNS[0]
#   output[:, 1] -> TARGET_COLUMNS[1]
TARGET_COLUMNS = [
    "NET_PATIENT_REVENUE",
    "TOTAL_OPERATING_EXP",   # requested as TOT_OVERALL_EXP, which does not exist here
]

# Documented alias mapping, printed in Section 3 for transparency.
REQUESTED_TARGET_ALIAS = {
    "NET_PATIENT_REVENUE": "NET_PATIENT_REVENUE",
    "TOT_OVERALL_EXP":     "TOTAL_OPERATING_EXP",
}

SEQUENCE_LENGTH = 56
RANDOM_SEED     = 42
BATCH_SIZE      = 128
EPOCHS          = 100
PATIENCE        = 15
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4

RESUME = True          # continue from latest checkpoint if one exists

# Numerical-stability / performance switches (implementation configuration)
USE_AMP        = False   # kept False: fp16 overflow is a known NaN source here
GRAD_CLIP_NORM = 1.0     # None to disable
NUM_WORKERS    = 0

# Scheduler (implementation configuration, NOT paper-specified)
SCHED_FACTOR   = 0.5
SCHED_PATIENCE = 5
SCHED_MIN_LR   = 1e-7

# Metric options
MAPE_EPSILON            = 1.0    # denominator floor, in original target units
TRAIN_METRICS_EVAL_PASS = False  # True = extra no-dropout pass for train metrics

MODELS_TO_RUN = ["cnn_lstm", "tcn_lstm", "lstm_mtrans_mlp"]
# ============================================================
# END OF USER CONFIGURATION
# ============================================================
print("Config loaded.")
print("  DATA_DIR      :", DATA_DIR)
print("  OUTPUT_DIR    :", OUTPUT_DIR)
print("  TARGET_COLUMNS:", TARGET_COLUMNS)
print("  SEQUENCE_LENGTH:", SEQUENCE_LENGTH)
''')

# ---------------- imports ----------------
md(r"""## 2. Imports, reproducibility, device""")

code(r'''import os, json, time, random, copy, warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import weight_norm

from sklearn.metrics import (r2_score, explained_variance_score,
                             median_absolute_error, max_error)

import matplotlib
matplotlib.use("Agg")           # save figures to disk; keeps notebook size small
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
    "figure.dpi": 300, "savefig.dpi": 300,
})


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = bool(USE_AMP and DEVICE.type == "cuda")

print("Device:", "CUDA" if DEVICE.type == "cuda" else "CPU")
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("AMP enabled:", AMP_ENABLED)
print("Seed:", RANDOM_SEED, "| torch:", torch.__version__)

DATA_DIR_P = Path(DATA_DIR)
OUT_P = Path(OUTPUT_DIR)
''')

# ---------------- load + verify ----------------
md(r"""## 3. Load the supplied splits and verify them

The five CSVs are used exactly as supplied. Nothing is re-split, re-scaled or
re-imputed.""")

code(r'''REQUIRED = ["train.csv", "val.csv", "test.csv"]
for fn in REQUIRED:
    if not (DATA_DIR_P / fn).exists():
        raise FileNotFoundError(
            f"Required file missing: {DATA_DIR_P / fn}\n"
            "Fix DATA_DIR in the configuration cell. No data is created or downloaded."
        )

train_df = pd.read_csv(DATA_DIR_P / "train.csv", parse_dates=["DATE"])
val_df   = pd.read_csv(DATA_DIR_P / "val.csv",   parse_dates=["DATE"])
test_df  = pd.read_csv(DATA_DIR_P / "test.csv",  parse_dates=["DATE"])

print("=" * 72)
print("SUPPLIED SPLITS")
print("=" * 72)
for nm, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    months = (d.DATE.max().to_period("M") - d.DATE.min().to_period("M")).n + 1
    print(f"  {nm:6s} rows={len(d):5d} cols={d.shape[1]:3d} "
          f"{d.DATE.min().date()} .. {d.DATE.max().date()}  "
          f"months={months}  dup_dates={d.DATE.duplicated().sum()}")

# ---- target-name resolution, explicit and printed ----
print("\n" + "=" * 72)
print("TARGET RESOLUTION")
print("=" * 72)
all_cols = list(train_df.columns)
overall_cols = [c for c in all_cols if "OVERALL" in c.upper()]
print("  Columns containing 'OVERALL':", overall_cols if overall_cols else "NONE")
for requested, actual in REQUESTED_TARGET_ALIAS.items():
    exists_req = requested in all_cols
    exists_act = actual in all_cols
    if not exists_act:
        raise KeyError(f"Configured target column '{actual}' not found in the dataset.")
    if requested == actual:
        print(f"  '{requested}' -> found directly.")
    else:
        print(f"  '{requested}' NOT FOUND -> mapped to '{actual}' "
              f"(exists={exists_act}). Documented alias, not a silent substitution.")

TARGETS = list(TARGET_COLUMNS)
N_TARGETS = len(TARGETS)
print("\n  TARGET INDEX MAPPING (enforced everywhere):")
for i, t in enumerate(TARGETS):
    print(f"     output[:, {i}] = {t}")
''')

code(r'''# ---- feature columns: the supplied *_scaled features, targets excluded ----
EXCLUDE = {"DATE", "IS_ANOMALY", "ANOMALY_TYPE", *TARGETS}
RAW_FEATURES = [c for c in train_df.columns
                if not c.endswith("_scaled") and c not in EXCLUDE]
FEATURE_COLS = [f"{c}_scaled" for c in RAW_FEATURES]
TARGET_COLS_SCALED = [f"{t}_scaled" for t in TARGETS]

missing = [c for c in FEATURE_COLS + TARGET_COLS_SCALED if c not in train_df.columns]
if missing:
    raise KeyError(f"Missing expected scaled columns: {missing[:20]}")

print(f"Feature columns used: {len(FEATURE_COLS)} (pre-scaled, supplied by your preprocessing)")
print("IS_ANOMALY / ANOMALY_TYPE are EXCLUDED from the feature matrix "
      "(labels, not forecasting inputs).")
print("\nFirst 10 feature columns:", FEATURE_COLS[:10])
''')

md(r"""### 3.1 Leakage audit of the supplied scaling

Claim to verify: every `_scaled` column is a z-score fitted on **train only**,
and the targets use `log1p` then a train-only z-score. If true, the supplied
preprocessing carries no val/test leakage and can be used as-is.""")

code(r'''print("=" * 72)
print("SCALING PROVENANCE CHECK")
print("=" * 72)

worst_feat = 0.0
for c in RAW_FEATURES:
    mu, sd = train_df[c].mean(), train_df[c].std(ddof=0)
    if sd == 0:
        continue
    for d in (val_df, test_df):
        err = float(np.max(np.abs((d[c] - mu) / sd - d[f"{c}_scaled"])))
        worst_feat = max(worst_feat, err)
print(f"  Features: max |recomputed(train mu,sd) - supplied _scaled| on val+test "
      f"= {worst_feat:.3e}")

TARGET_LOG_MU, TARGET_LOG_SD = {}, {}
worst_targ = 0.0
for t in TARGETS:
    lg = np.log1p(train_df[t])
    mu, sd = float(lg.mean()), float(lg.std(ddof=0))
    TARGET_LOG_MU[t], TARGET_LOG_SD[t] = mu, sd
    for d in (train_df, val_df, test_df):
        err = float(np.max(np.abs((np.log1p(d[t]) - mu) / sd - d[f"{t}_scaled"])))
        worst_targ = max(worst_targ, err)
    print(f"  {t}: log1p+zscore train-fit  mu={mu:.6f} sd={sd:.6f}")
print(f"  Targets: max reconstruction error across all splits = {worst_targ:.3e}")

if worst_feat > 1e-8 or worst_targ > 1e-6:
    raise AssertionError(
        "Supplied _scaled columns are NOT reproducible from train-only statistics. "
        "The scaling provenance cannot be trusted; stopping rather than proceeding."
    )
print("\n  VERDICT: all scaling is train-fit only. No preprocessing leakage. "
      "Supplied columns used as-is.")
''')

# ---------------- sequence construction ----------------
md(r"""## 4. Sequence construction on one continuous series

The three splits are **contiguous, gap-free daily** records (verified below), so
they are concatenated into a single chronological series. A window for a target
at time `t` is `[t-L, t-1]` — strictly historical.

**Why concatenate rather than window each split independently?** Windowing each
split alone would discard the first `L` days of validation and test (no history
available), and would waste the `*_with_context.csv` files' purpose. Your
supplied `val_with_context.csv` / `test_with_context.csv` already encode exactly
this idea: they prepend 30 rows copied verbatim from the *preceding* split so the
earliest val/test windows have history. Concatenation generalises that to any
`L`, so **every** validation and test target remains predictable at
`SEQUENCE_LENGTH = 56`.

**Why this is leakage-free.** A window for a validation target may include train
rows, and a window for a test target may include validation rows — these are
observations strictly *before* the prediction time, i.e. legitimately known at
inference. The target row `t` itself is never inside its own window, so
same-timestamp derived features (e.g. `OPERATING_MARGIN_PCT`, which is an exact
function of both targets at time `t`) cannot leak the answer. Split membership is
assigned by the **target date**, so no target ever changes partition.""")

code(r'''full_df = (pd.concat([train_df, val_df, test_df], ignore_index=True)
             .sort_values("DATE").reset_index(drop=True))

n_expected = (full_df.DATE.max() - full_df.DATE.min()).days + 1
step_counts = full_df.DATE.diff().dropna().dt.days.value_counts().to_dict()

print("=" * 72)
print("CONTINUOUS SERIES CHECK")
print("=" * 72)
print(f"  rows={len(full_df)}  {full_df.DATE.min().date()} .. {full_df.DATE.max().date()}")
print(f"  duplicate dates : {full_df.DATE.duplicated().sum()}")
print(f"  day-step counts : {step_counts}   (must be exactly {{1: n-1}})")
print(f"  expected daily rows={n_expected}  actual={len(full_df)}")

if full_df.DATE.duplicated().any():
    raise AssertionError("Duplicate dates in the concatenated series.")
if set(step_counts.keys()) != {1}:
    raise AssertionError(f"Series is not gap-free daily; day steps found: {step_counts}")
if n_expected != len(full_df):
    raise AssertionError("Row count does not match the calendar span.")
print("  VERDICT: perfectly contiguous daily series. Concatenation is valid.")

TRAIN_END = train_df.DATE.max()
VAL_END   = val_df.DATE.max()

F_ALL = full_df[FEATURE_COLS].to_numpy(np.float32)
Y_ALL = full_df[TARGET_COLS_SCALED].to_numpy(np.float32)
DATES = full_df["DATE"].to_numpy()

L = int(SEQUENCE_LENGTH)
if L >= len(full_df):
    raise ValueError(f"SEQUENCE_LENGTH={L} too large for {len(full_df)} rows.")

idx_by_split = {"train": [], "val": [], "test": []}
for i in range(L, len(full_df)):
    d = full_df.DATE.iloc[i]
    if d <= TRAIN_END:
        idx_by_split["train"].append(i)
    elif d <= VAL_END:
        idx_by_split["val"].append(i)
    else:
        idx_by_split["test"].append(i)


def build(indices):
    X = np.stack([F_ALL[i - L:i] for i in indices]).astype(np.float32)
    y = np.stack([Y_ALL[i] for i in indices]).astype(np.float32)
    return X, y, np.array([DATES[i] for i in indices])


X_train, y_train, dt_train = build(idx_by_split["train"])
X_val,   y_val,   dt_val   = build(idx_by_split["val"])
X_test,  y_test,  dt_test  = build(idx_by_split["test"])

INPUT_DIM = X_train.shape[-1]

print("\n" + "=" * 72)
print("SEQUENCE SHAPES")
print("=" * 72)
print("Train X shape:          ", X_train.shape)
print("Train y shape:          ", y_train.shape)
print("Validation X shape:     ", X_val.shape)
print("Validation y shape:     ", y_val.shape)
print("Test X shape:           ", X_test.shape)
print("Test y shape:           ", y_test.shape)
print("Number of input features:", INPUT_DIM)
print("Sequence length:        ", L)
print()
print(f"Train targets: {dt_train.min()} .. {dt_train.max()}  (n={len(dt_train)})")
print(f"Val   targets: {dt_val.min()} .. {dt_val.max()}  (n={len(dt_val)})")
print(f"Test  targets: {dt_test.min()} .. {dt_test.max()}  (n={len(dt_test)})")

# every official val/test row must survive as a predictable target
assert len(dt_val) == len(val_df), (len(dt_val), len(val_df))
assert len(dt_test) == len(test_df), (len(dt_test), len(test_df))
print(f"\nAll {len(val_df)} validation and {len(test_df)} test targets retained "
      f"(none lost to the lookback window).")
print(f"Train targets = {len(train_df)} - {L} = {len(train_df) - L} "
      f"(first {L} days have no full history).")
''')

md(r"""## 5. Data validation gate""")

code(r'''print("=" * 72)
print("DATA VALIDATION")
print("=" * 72)
problems = []

def chk(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        problems.append(label)

chk("Dataset found", True, "train.csv / val.csv / test.csv")
chk("Train split found", len(X_train) > 0, f"{len(X_train)} sequences")
chk("Validation split found", len(X_val) > 0, f"{len(X_val)} sequences")
chk("Test split found", len(X_test) > 0, f"{len(X_test)} sequences")
for t in TARGETS:
    chk(f"Target {t} found", t in full_df.columns)
chk("Feature dimensions consistent",
    X_train.shape[-1] == X_val.shape[-1] == X_test.shape[-1], f"{INPUT_DIM}")
chk("Sequence dimensions valid",
    X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == L, f"L={L}")
chk("No unexpected target mismatch",
    y_train.shape[1] == y_val.shape[1] == y_test.shape[1] == N_TARGETS, f"{N_TARGETS}")
chk("Chronological order: train < val < test",
    dt_train.max() < dt_val.min() < dt_val.max() < dt_test.min())
chk("No target-date overlap between splits",
    len(set(dt_train) & set(dt_val)) == 0 and len(set(dt_val) & set(dt_test)) == 0)

fin = True
for nm, a in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
              ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
    nn_, ni = int(np.isnan(a).sum()), int(np.isinf(a).sum())
    if nn_ or ni:
        fin = False
        print(f"        !! {nm}: {nn_} NaN, {ni} Inf")
chk("No NaN/Inf in model input", fin)
chk("Input magnitudes bounded (pre-scaled)",
    float(np.abs(X_train).max()) < 100.0,
    f"max|X_train| = {float(np.abs(X_train).max()):.3f}")

if problems:
    raise ValueError("Validation failed: " + ", ".join(problems))
print("\nAll validation checks passed.")
''')

# ---------------- output dirs ----------------
md(r"""## 6. Output folder structure""")

code(r'''MODEL_KEYS = ["cnn_lstm", "tcn_lstm", "lstm_mtrans_mlp"]
MODEL_DISPLAY = {"cnn_lstm": "CNN-LSTM", "tcn_lstm": "TCN-LSTM",
                 "lstm_mtrans_mlp": "LSTM-mTrans-MLP"}
SUBDIRS = ["checkpoints/latest", "checkpoints/best", "metrics",
           "predictions", "plots", "logs", "config"]

for mk in MODEL_KEYS:
    for sd in SUBDIRS:
        (OUT_P / mk / sd).mkdir(parents=True, exist_ok=True)
(OUT_P / "comparison" / "plots").mkdir(parents=True, exist_ok=True)

def mdir(mk, *parts):
    return OUT_P / mk / Path(*parts)

print("Output tree under:", OUT_P.resolve())
for mk in MODEL_KEYS:
    print("  " + mk + "/  " + " ".join(SUBDIRS))
print("  comparison/plots/")
''')

# ---------------- dataloaders ----------------
md(r"""## 7. DataLoaders

Only the training loader is shuffled. Shuffling training *windows* does not
break temporal integrity — each window is a self-contained
`(history → next-day target)` example and the split boundaries are untouched.
Validation and test order is preserved so predictions stay chronological.""")

code(r'''def make_loader(X, y, shuffle):
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))

train_loader = make_loader(X_train, y_train, True)
val_loader   = make_loader(X_val,   y_val,   False)
test_loader  = make_loader(X_test,  y_test,  False)

print(f"train batches={len(train_loader)} (shuffled)")
print(f"val   batches={len(val_loader)} (order preserved)")
print(f"test  batches={len(test_loader)} (order preserved)")
xb, yb = next(iter(train_loader))
print("sample batch X:", tuple(xb.shape), " y:", tuple(yb.shape))
''')

# ---------------- metrics ----------------
md(r"""## 8. Metrics — exactly the ten required, per target

`MAE, RMSE, MAPE, sMAPE, R2, ExplainedVar, MedianAE, MaxError, MAE_scaled, RMSE_scaled`

- The 8 primary metrics are computed in **original financial units** (dollars),
  by inverting the supplied `log1p` + train-z-score transform.
- `MAPE` / `sMAPE` use a floored denominator (`MAPE_EPSILON`). Both targets are
  well above zero here (min ≈ \$77k), so MAPE is well-behaved for this dataset.
- `MAE_scaled = MAE / train_target_std`, `RMSE_scaled = RMSE / train_target_std`,
  using the **train** std in original units for every split, so the values are
  comparable across splits and models.
- Reported **separately per target**, never averaged into one score.""")

code(r'''METRIC_NAMES = ["MAE", "RMSE", "MAPE", "sMAPE", "R2",
                "ExplainedVar", "MedianAE", "MaxError",
                "MAE_scaled", "RMSE_scaled"]


def to_original_units(y_scaled):
    """Invert log1p + train z-score, per target."""
    y_scaled = np.asarray(y_scaled, dtype=np.float64)
    out = np.empty_like(y_scaled)
    for i, t in enumerate(TARGETS):
        out[:, i] = np.expm1(y_scaled[:, i] * TARGET_LOG_SD[t] + TARGET_LOG_MU[t])
    return out


y_train_orig = to_original_units(y_train)
TRAIN_TARGET_STD = y_train_orig.std(axis=0, ddof=0)
if np.any(TRAIN_TARGET_STD <= 0) or not np.all(np.isfinite(TRAIN_TARGET_STD)):
    raise ValueError(f"Invalid train target std: {TRAIN_TARGET_STD}")

print("Train-target std (original units), used for *_scaled metrics:")
for i, t in enumerate(TARGETS):
    print(f"   {t}: {TRAIN_TARGET_STD[i]:,.4f}")


def metrics_1d(y_true, y_pred, train_std):
    y_true = np.asarray(y_true, np.float64).ravel()
    y_pred = np.asarray(y_pred, np.float64).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() != y_true.size:
        warnings.warn(f"Dropped {y_true.size - int(m.sum())} non-finite pair(s).")
    y_true, y_pred = y_true[m], y_pred[m]
    if y_true.size == 0:
        return {k: float("nan") for k in METRIC_NAMES}

    err = y_true - y_pred
    ae = np.abs(err)
    mae = float(ae.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float((ae / np.maximum(np.abs(y_true), MAPE_EPSILON)).mean() * 100)
    smape = float((2 * ae / np.maximum(np.abs(y_true) + np.abs(y_pred),
                                       MAPE_EPSILON)).mean() * 100)
    if np.var(y_true) <= 0 or y_true.size < 2:
        r2 = evs = float("nan")
    else:
        r2 = float(r2_score(y_true, y_pred))
        evs = float(explained_variance_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape,
            "R2": r2, "ExplainedVar": evs,
            "MedianAE": float(median_absolute_error(y_true, y_pred)),
            "MaxError": float(max_error(y_true, y_pred)),
            "MAE_scaled": float(mae / train_std),
            "RMSE_scaled": float(rmse / train_std)}


def metrics_all(y_true_s, y_pred_s):
    yt, yp = to_original_units(y_true_s), to_original_units(y_pred_s)
    return {t: metrics_1d(yt[:, i], yp[:, i], TRAIN_TARGET_STD[i])
            for i, t in enumerate(TARGETS)}


def flatten(prefix, per_target):
    return {f"{prefix}_{t}_{k}": v
            for t, d in per_target.items() for k, v in d.items()}


# self-test of the maths on synthetic values
_a = np.array([100.0, 200.0, 300.0, 400.0])
_b = _a + np.array([10.0, -10.0, 10.0, -10.0])
_m = metrics_1d(_a, _b, 100.0)
assert abs(_m["MAE"] - 10) < 1e-9 and abs(_m["RMSE"] - 10) < 1e-9
assert abs(_m["MAE_scaled"] - 0.1) < 1e-9 and abs(_m["MaxError"] - 10) < 1e-9
print("\nMetric self-test passed (MAE / RMSE / MAE_scaled / MaxError).")
print("Metrics tracked:", METRIC_NAMES)
''')

# ---------------- models ----------------
md(r"""## 9. CNN-LSTM — Paper Fidelity

```
Paper:   "Enhanced Multivariate Time Series Forecasting"
Authors: A. Mahmoud and A. Mohammed
Venue:   Neural Processing Letters, 56(5), p.223
Model:   CNN-LSTM  (one of four compared: CNN-LSTM, CNN-BiLSTM,
                    TCN-LSTM, TCN-BiLSTM)
```

**Architecture described by the paper (high level):** multivariate time-series
input → convolutional feature extraction for local temporal patterns → LSTM for
sequential dependencies → forecasting output.

**Implementation here:**
`[B,T,F] → Conv1d → ReLU → Conv1d → ReLU → Dropout → LSTM → last step → Linear → [B,2]`

**Implementation assumptions — not explicitly specified in the paper** (and not
verifiable from the sources I could reach):

- 2 conv layers, 64 channels, kernel 3, `padding=1`
- LSTM: 2 layers, hidden 128, **unidirectional**
- Dropout 0.2; head `Linear(128 → 2)`
- Final LSTM time step used as the sequence representation

**Deliberately NOT added:** no bidirectional LSTM, no Transformer, no attention,
no dilation, no residual connections.

**On padding and leakage:** the convolution mixes neighbouring steps *inside the
window*. Every step in that window strictly precedes the target, so this is not
future leakage — the target row is never in its own window.""")

code(r'''class CNNLSTM(nn.Module):
    """CNN-LSTM benchmark: conv feature extraction -> LSTM -> linear head."""

    def __init__(self, input_dim, n_targets, conv_channels=64, kernel_size=3,
                 lstm_hidden=128, lstm_layers=2, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=pad)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0,
                            bidirectional=False)
        self.head = nn.Linear(lstm_hidden, n_targets)

    def forward(self, x):                    # [B, T, F]
        h = x.transpose(1, 2)                # [B, F, T]
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        h = self.drop(h).transpose(1, 2)     # [B, T, C]
        out, _ = self.lstm(h)
        return self.head(out[:, -1, :])      # [B, n_targets]


print("CNNLSTM defined.")
''')

md(r"""## 10. TCN-LSTM — Paper Fidelity

```
Paper:   "Enhanced Multivariate Time Series Forecasting"
Authors: A. Mahmoud and A. Mohammed
Venue:   Neural Processing Letters, 56(5), p.223
Model:   TCN-LSTM
```

**Architecture described by the paper (high level):** a Temporal Convolutional
Network extracts temporal/local patterns → LSTM models sequential dependencies →
forecasting head.

**Implementation here:** a standard TCN residual stack — dilated **causal**
convolutions with `Chomp1d` cropping, weight normalisation, ReLU, dropout, and a
`1×1` residual projection when channel counts differ — then an LSTM and a linear
head.

`[B,T,F] → [TCN residual block × 4, dilations 1,2,4,8] → LSTM → last step → Linear → [B,2]`

**Implementation assumptions — not explicitly specified in the paper:**

- 4 residual blocks, 64 channels, kernel 3, dilations `1, 2, 4, 8`
  (receptive field `1 + 2·(3-1)·(1+2+4+8) = 61` ≥ `L = 56`, so the stack can see
  the whole window)
- Each block: 2 × (weight-normed causal Conv1d → Chomp → ReLU → Dropout) + residual
- LSTM: 2 layers, hidden 128, unidirectional; dropout 0.2; head `Linear(128 → 2)`

**Deliberately NOT added:** no Transformer, no attention, no bidirectional LSTM.
The convolutions are genuinely dilated and genuinely causal — this is not a
generic CNN.

**Causality:** `Chomp1d` strips the right-hand padding so output step `i` never
depends on input steps `> i`.""")

code(r'''class Chomp1d(nn.Module):
    """Trim trailing padding to keep the convolution causal."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN residual block: 2x (causal dilated conv -> chomp -> ReLU -> dropout)."""

    def __init__(self, n_in, n_out, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_in, n_out, kernel_size,
                                           padding=padding, dilation=dilation))
        self.chomp1, self.relu1, self.drop1 = Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_out, n_out, kernel_size,
                                           padding=padding, dilation=dilation))
        self.chomp2, self.relu2, self.drop2 = Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu_out = nn.ReLU()
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TCNLSTM(nn.Module):
    """TCN-LSTM benchmark: dilated causal TCN -> LSTM -> linear head."""

    def __init__(self, input_dim, n_targets, tcn_channels=(64, 64, 64, 64),
                 kernel_size=3, lstm_hidden=128, lstm_layers=2, dropout=0.2):
        super().__init__()
        blocks, prev = [], input_dim
        for i, ch in enumerate(tcn_channels):
            blocks.append(TemporalBlock(prev, ch, kernel_size,
                                        dilation=2 ** i, dropout=dropout))
            prev = ch
        self.tcn = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(prev, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0,
                            bidirectional=False)
        self.head = nn.Linear(lstm_hidden, n_targets)

    def forward(self, x):
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(h)
        return self.head(out[:, -1, :])


_rf = 1 + 2 * (3 - 1) * (1 + 2 + 4 + 8)
print(f"TCNLSTM defined. Receptive field = {_rf} (sequence length L = {L}) "
      f"-> covers whole window: {_rf >= L}")
''')

md(r"""## 11. LSTM-mTrans-MLP — Paper Fidelity

```
Paper:   "LSTM-Transformer-Based Robust Hybrid Deep Learning Model for
          Financial Time Series Forecasting"
Authors: Kabir et al.
Year:    2025
Venue:   Sci, 7(1), 7
Model:   LSTM-mTrans-MLP
```

**Architecture described by the paper (verified from the abstract):** a hybrid
integrating an **LSTM** network, a **modified Transformer** network, and a
**multilayered perceptron (MLP)** — LSTM for sequential dependencies, the
Transformer component for long-range temporal relationships, the MLP for
nonlinear feature relationships and final prediction.

**Implementation here:**
`[B,T,F] → LSTM → +positional encoding → mTrans blocks → mean-pool → MLP → Linear → [B,2]`

> **Implementation assumption — not explicitly specified in the paper.**
> The paper describes the "modified Transformer" conceptually, but I could not
> access implementation-level detail for the specific modification (the
> publisher page returned HTTP 403). This notebook implements it as a
> **pre-normalisation Transformer encoder** — LayerNorm before the attention
> sub-layer and before the feed-forward sub-layer, residual connections around
> both, GELU in the FFN — applied to the LSTM output sequence with sinusoidal
> positional encoding.
>
> This is a documented, reasonable reconstruction. It is **not** presented as the
> paper's exact mechanism. Given the paper's architecture subsection, this block
> can be revised to match precisely.

**Further implementation assumptions:** LSTM 2 layers / hidden 128 /
unidirectional; mTrans 2 blocks, `d_model=128`, 4 heads, FFN 256, dropout 0.2;
fixed sinusoidal positional encoding; mean pooling over time; MLP
`128 → 128 → 64 → 2` with GELU.

**Deliberately NOT added:** no CNN or TCN front end. The defining
`LSTM → mTrans → MLP` order is preserved.

**Attention masking:** self-attention is unmasked *within the window*. Every step
in the window precedes the target, so attending across it exposes no future
information relative to the prediction.""")

code(r'''class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, :pe[:, 1::2].shape[1]]
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MTransBlock(nn.Module):
    """Pre-norm Transformer encoder block.

    IMPLEMENTATION ASSUMPTION: pre-norm ordering + GELU FFN is this notebook's
    reconstruction of the paper's 'modified Transformer'; the paper's own
    implementation detail was not accessible.
    """

    def __init__(self, d_model, n_heads=4, ffn_dim=256, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(a)
        return x + self.drop2(self.ffn(self.norm2(x)))


class LSTMmTransMLP(nn.Module):
    """LSTM -> modified Transformer -> MLP -> linear head."""

    def __init__(self, input_dim, n_targets, lstm_hidden=128, lstm_layers=2,
                 n_blocks=2, n_heads=4, ffn_dim=256,
                 mlp_hidden=(128, 64), dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0,
                            bidirectional=False)
        self.posenc = SinusoidalPositionalEncoding(lstm_hidden)
        self.blocks = nn.ModuleList([MTransBlock(lstm_hidden, n_heads, ffn_dim, dropout)
                                     for _ in range(n_blocks)])
        self.norm_out = nn.LayerNorm(lstm_hidden)
        layers, prev = [], lstm_hidden
        for hd in mlp_hidden:
            layers += [nn.Linear(prev, hd), nn.GELU(), nn.Dropout(dropout)]
            prev = hd
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_targets)

    def forward(self, x):
        h, _ = self.lstm(x)
        h = self.posenc(h)
        for b in self.blocks:
            h = b(h)
        h = self.norm_out(h).mean(dim=1)
        return self.head(self.mlp(h))


print("LSTMmTransMLP defined.")
''')

# ---------------- shape tests ----------------
md(r"""## 12. Shape validation, parameter counts, model summaries""")

code(r'''def build_model(mk):
    if mk == "cnn_lstm":        return CNNLSTM(INPUT_DIM, N_TARGETS)
    if mk == "tcn_lstm":        return TCNLSTM(INPUT_DIM, N_TARGETS)
    if mk == "lstm_mtrans_mlp": return LSTMmTransMLP(INPUT_DIM, N_TARGETS)
    raise KeyError(mk)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


xb, yb = next(iter(train_loader))
B = xb.shape[0]
assert tuple(xb.shape) == (B, L, INPUT_DIM), xb.shape
assert tuple(yb.shape) == (B, N_TARGETS), yb.shape

MODEL_SUMMARY = {}
print("=" * 72)
print("SHAPE VALIDATION AND PARAMETER COUNTS")
print("=" * 72)
for mk in MODEL_KEYS:
    set_seed(RANDOM_SEED)
    m = build_model(mk).to(DEVICE).eval()
    with torch.no_grad():
        out = m(xb.to(DEVICE))
    if tuple(out.shape) != (B, N_TARGETS):
        raise ValueError(f"{MODEL_DISPLAY[mk]} -> {tuple(out.shape)}, expected {(B, N_TARGETS)}")
    if not torch.isfinite(out).all():
        raise ValueError(f"{MODEL_DISPLAY[mk]} produced non-finite output on the shape test.")
    n = count_params(m)
    MODEL_SUMMARY[mk] = {
        "model_name": MODEL_DISPLAY[mk],
        "input_shape": [L, INPUT_DIM], "output_shape": [N_TARGETS],
        "sequence_length": L, "n_features": INPUT_DIM,
        "trainable_parameters": int(n),
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS, "early_stopping_patience": PATIENCE,
        "scheduler": f"ReduceLROnPlateau(min, factor={SCHED_FACTOR}, "
                     f"patience={SCHED_PATIENCE}, min_lr={SCHED_MIN_LR})",
        "loss": "SmoothL1Loss", "device": str(DEVICE),
        "targets": TARGETS, "seed": RANDOM_SEED,
    }
    print(f"  {MODEL_DISPLAY[mk]:18s} Input [B,{L},{INPUT_DIM}] -> Output [B,{N_TARGETS}]  "
          f"PASSED | params={n:,}")
    del m

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

with open(OUT_P / "comparison" / "model_summary.json", "w") as f:
    json.dump(MODEL_SUMMARY, f, indent=2)
for mk in MODEL_KEYS:
    with open(mdir(mk, "config", "model_summary.json"), "w") as f:
        json.dump(MODEL_SUMMARY[mk], f, indent=2)

print("\n" + "=" * 72)
for mk in MODEL_KEYS:
    s = MODEL_SUMMARY[mk]
    print(s["model_name"])
    print(f"   Input {s['input_shape']} -> Output {s['output_shape']} | "
          f"seq={s['sequence_length']} feats={s['n_features']}")
    print(f"   params={s['trainable_parameters']:,} | {s['optimizer']} "
          f"lr={s['learning_rate']} wd={s['weight_decay']} bs={s['batch_size']}")
    print(f"   max_epochs={s['max_epochs']} patience={s['early_stopping_patience']} "
          f"loss={s['loss']} device={s['device']}")
    print(f"   scheduler={s['scheduler']}")
    print(f"   targets={s['targets']}")
''')

# ---------------- loss + checkpoints ----------------
md(r"""## 13. Loss

`nn.SmoothL1Loss` (Huber) for **all three** models. Rationale: hospital financial
series carry occasional extreme values, and a Huber-type objective is less
dominated by a few large residuals than pure MSE while remaining a standard
regression loss.

> **Implementation decision, not paper-specified.** Neither paper's loss
> specification was accessible. Using the *same* loss across all three is
> required by the fair-comparison contract — the intended difference is the
> architecture, not the objective.""")

code(r'''CRITERION = nn.SmoothL1Loss()
print("Loss:", CRITERION.__class__.__name__, "- identical for all three models")
''')

md(r"""## 14. Checkpoint utilities

`latest.pt` (full resumable state, every epoch) and `best.pt` (lowest
**validation** loss). Compatibility is verified on load; a mismatch raises rather
than loading wrong weights. Writes are atomic (temp file + replace) so an
interrupt cannot corrupt a good checkpoint. **The test set never influences
checkpoint selection.**""")

code(r'''def _sig(mk):
    return {"model_name": MODEL_DISPLAY[mk], "input_dim": int(INPUT_DIM),
            "sequence_length": int(L), "n_targets": int(N_TARGETS)}


def save_latest(mk, epoch, model, opt, sched, scaler, best_vl, best_ep, hist, cfg):
    p = mdir(mk, "checkpoints", "latest", "latest.pt")
    payload = {"epoch": int(epoch), "model_state_dict": model.state_dict(),
               "optimizer_state_dict": opt.state_dict(),
               "scheduler_state_dict": sched.state_dict(),
               "best_val_loss": float(best_vl), "best_epoch": int(best_ep),
               "history": hist, "config": cfg, "compat": _sig(mk)}
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    tmp = p.with_suffix(".pt.tmp"); torch.save(payload, tmp); tmp.replace(p)


def save_best(mk, epoch, model, vl, hist, cfg):
    p = mdir(mk, "checkpoints", "best", "best.pt")
    tmp = p.with_suffix(".pt.tmp")
    torch.save({"epoch": int(epoch), "model_state_dict": model.state_dict(),
                "val_loss": float(vl), "history": hist, "config": cfg,
                "compat": _sig(mk)}, tmp)
    tmp.replace(p)
    with open(mdir(mk, "checkpoints", "best", "best_model_info.json"), "w") as f:
        json.dump({"model_name": MODEL_DISPLAY[mk], "best_epoch": int(epoch),
                   "best_validation_loss": float(vl)}, f, indent=2)


def _check_compat(mk, ck, path):
    want, got = _sig(mk), ck.get("compat")
    if got is None:
        raise ValueError(f"Checkpoint {path} has no compatibility signature. "
                         "Delete it or set RESUME=False.")
    if got != want:
        raise ValueError("Checkpoint is incompatible with the current dataset/model "
                         f"configuration.\n  checkpoint: {got}\n  current   : {want}\n"
                         f"  file: {path}\nRefusing to load mismatched weights.")


def load_latest(mk, model, opt, sched, scaler):
    p = mdir(mk, "checkpoints", "latest", "latest.pt")
    if not p.exists():
        return None
    try:
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Could not read checkpoint {p}: {e}\n"
                           "It may be corrupted. Delete it or set RESUME=False.") from e
    _check_compat(mk, ck, p)
    model.load_state_dict(ck["model_state_dict"])
    opt.load_state_dict(ck["optimizer_state_dict"])
    sched.load_state_dict(ck["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ck:
        scaler.load_state_dict(ck["scaler_state_dict"])
    return (int(ck["epoch"]) + 1, float(ck["best_val_loss"]),
            int(ck.get("best_epoch", ck["epoch"])), list(ck.get("history", [])))


def load_best_for_eval(mk):
    p = mdir(mk, "checkpoints", "best", "best.pt")
    if not p.exists():
        raise FileNotFoundError(
            f"No best checkpoint for {MODEL_DISPLAY[mk]} at {p}. Train first. "
            "The latest checkpoint is deliberately NOT substituted for final reporting.")
    ck = torch.load(p, map_location=DEVICE, weights_only=False)
    _check_compat(mk, ck, p)
    m = build_model(mk).to(DEVICE)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m, int(ck["epoch"]), float(ck["val_loss"])


print("Checkpoint utilities ready (atomic writes, compat-verified, resumable).")
''')

# ---------------- training engine ----------------
md(r"""## 15. Training engine

Per epoch: all **ten** metrics for **both** targets on **train** and
**validation** (40 metric columns), plus `train_loss`, `val_loss`,
`learning_rate`, `epoch_time`. Includes a non-finite-loss guard with
diagnostics, optional gradient clipping, and duplicate-epoch protection when
appending history on resume.""")

code(r'''@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses, ys, ps = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
            out = model(xb); loss = CRITERION(out, yb)
        losses.append(float(loss.detach().float().cpu()))
        ys.append(yb.detach().float().cpu().numpy())
        ps.append(out.detach().float().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(ys), np.concatenate(ps)


def report_epoch(name, ep, tot, tl, vl, lr, trm, vam, secs):
    print("=" * 60)
    print(f"{name} | Epoch {ep}/{tot}")
    print("=" * 60)
    print(f"\nTrain Loss: {tl:.6f}\nVal Loss:   {vl:.6f}\nLR:         {lr:.8f}"
          f"\nEpoch time: {secs:.1f}s")
    for t in TARGETS:
        for lbl, d in (("TRAIN", trm), ("VALIDATION", vam)):
            print(f"\n{t} - {lbl}")
            for k in METRIC_NAMES:
                print(f"{k}: {d[t][k]:.6f}")
    print()


def train_model(mk, resume=None):
    name = MODEL_DISPLAY[mk]
    resume = RESUME if resume is None else resume
    set_seed(RANDOM_SEED)

    model = build_model(mk).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=SCHED_FACTOR,
        patience=SCHED_PATIENCE, min_lr=SCHED_MIN_LR)
    scaler = torch.amp.GradScaler(DEVICE.type) if AMP_ENABLED else None

    cfg = dict(MODEL_SUMMARY[mk])
    cfg.update({
        "resume_requested": bool(resume), "amp_enabled": bool(AMP_ENABLED),
        "grad_clip_norm": GRAD_CLIP_NORM, "mape_epsilon": MAPE_EPSILON,
        "train_metrics_eval_pass": bool(TRAIN_METRICS_EVAL_PASS),
        "target_transform": "log1p + train-fit z-score (supplied by preprocessing)",
        "train_target_std_original_units": TRAIN_TARGET_STD.tolist(),
        "feature_columns": FEATURE_COLS,
        "n_train_windows": int(len(X_train)), "n_val_windows": int(len(X_val)),
        "n_test_windows": int(len(X_test)),
        "implementation_notes": [
            "Layer sizes are implementation assumptions, not paper-specified.",
            "SmoothL1Loss shared across all three models (implementation decision).",
            "AdamW + ReduceLROnPlateau are implementation configuration.",
            "Gradient clipping is a numerical-stability measure, not paper-specified.",
        ],
    })
    with open(mdir(mk, "config", "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    start_ep, best_vl, best_ep, hist = 1, float("inf"), -1, []
    if resume:
        got = load_latest(mk, model, opt, sched, scaler)
        if got is None:
            print(f"[{name}] RESUME=True but no checkpoint found - starting at epoch 1.")
        else:
            start_ep, best_vl, best_ep, hist = got
            print(f"[{name}] Resumed. Continuing at epoch {start_ep} "
                  f"(best val {best_vl:.6f} @ epoch {best_ep}).")
    if start_ep > EPOCHS:
        print(f"[{name}] Already at epoch {start_ep-1} >= EPOCHS={EPOCHS}. Nothing to do.")
        return hist

    no_improve = len([h for h in hist if h["epoch"] > best_ep]) if hist else 0

    for ep in range(start_ep, EPOCHS + 1):
        t0 = time.time()
        model.train()
        blosses, trt, trp = [], [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                out = model(xb); loss = CRITERION(out, yb)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"[{name}] Non-finite loss at epoch {ep} (loss={loss.item()}).\n"
                    f"  finite inputs : {bool(torch.isfinite(xb).all())}\n"
                    f"  finite targets: {bool(torch.isfinite(yb).all())}\n"
                    f"  max|input|    : {float(xb.abs().max()):.4f}\n"
                    f"  AMP enabled   : {AMP_ENABLED}\n"
                    "  Likely causes: unscaled inputs, learning rate too high, or fp16 "
                    "overflow under AMP. Values are not silently replaced.")

            if scaler is not None:
                scaler.scale(loss).backward()
                if GRAD_CLIP_NORM is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                opt.step()

            blosses.append(float(loss.detach().float().cpu()))
            if not TRAIN_METRICS_EVAL_PASS:
                trt.append(yb.detach().float().cpu().numpy())
                trp.append(out.detach().float().cpu().numpy())

        tl = float(np.mean(blosses))
        if TRAIN_METRICS_EVAL_PASS:
            _, tra, trb = evaluate(model, train_loader)
        else:
            tra, trb = np.concatenate(trt), np.concatenate(trp)
        trm = metrics_all(tra, trb)

        vl, vat, vap = evaluate(model, val_loader)
        vam = metrics_all(vat, vap)

        sched.step(vl)
        lr = float(opt.param_groups[0]["lr"])
        secs = time.time() - t0

        row = {"epoch": int(ep), "learning_rate": lr, "train_loss": tl,
               "val_loss": vl, "epoch_time": secs}
        row.update(flatten("train", trm)); row.update(flatten("val", vam))
        hist = [h for h in hist if int(h["epoch"]) != int(ep)]
        hist.append(row); hist.sort(key=lambda h: int(h["epoch"]))
        pd.DataFrame(hist).to_csv(mdir(mk, "metrics", "history.csv"), index=False)

        report_epoch(name, ep, EPOCHS, tl, vl, lr, trm, vam, secs)
        with open(mdir(mk, "logs", "train_log.txt"), "a") as lf:
            lf.write(f"epoch={ep} train_loss={tl:.6f} val_loss={vl:.6f} "
                     f"lr={lr:.8f} time={secs:.1f}s\n")

        if vl < best_vl - 1e-12:
            best_vl, best_ep, no_improve = vl, ep, 0
            save_best(mk, ep, model, vl, hist, cfg)
            print(f"  -> new best validation loss ({best_vl:.6f}); best.pt updated")
        else:
            no_improve += 1
            print(f"  -> no improvement {no_improve}/{PATIENCE} "
                  f"(best {best_vl:.6f} @ epoch {best_ep})")

        save_latest(mk, ep, model, opt, sched, scaler, best_vl, best_ep, hist, cfg)

        if no_improve >= PATIENCE:
            print("\nEarly stopping triggered.")
            print(f"Best epoch: {best_ep}")
            print(f"Best validation loss: {best_vl:.6f}")
            break

    print(f"\n[{name}] Done. Best epoch {best_ep}, best val loss {best_vl:.6f}")
    return hist


print("Training engine ready.")
''')

md(r"""## 16. CNN-LSTM — training""")
code(r'''HISTORIES = {}
if "cnn_lstm" in MODELS_TO_RUN:
    HISTORIES["cnn_lstm"] = train_model("cnn_lstm")
else:
    print("cnn_lstm skipped.")
''')

md(r"""## 17. TCN-LSTM — training""")
code(r'''if "tcn_lstm" in MODELS_TO_RUN:
    HISTORIES["tcn_lstm"] = train_model("tcn_lstm")
else:
    print("tcn_lstm skipped.")
''')

md(r"""## 18. LSTM-mTrans-MLP — training""")
code(r'''if "lstm_mtrans_mlp" in MODELS_TO_RUN:
    HISTORIES["lstm_mtrans_mlp"] = train_model("lstm_mtrans_mlp")
else:
    print("lstm_mtrans_mlp skipped.")
''')

# ---------------- test eval ----------------
md(r"""## 19. Final test evaluation — BEST checkpoint, evaluated once

The best (lowest-validation-loss) checkpoint is loaded for each model and the
test set is scored **once**. The latest checkpoint is never substituted.
Predictions are written in chronological test order, with dates.""")

code(r'''TEST_RESULTS, TEST_PRED = {}, {}

for mk in MODELS_TO_RUN:
    name = MODEL_DISPLAY[mk]
    model, bep, bvl = load_best_for_eval(mk)
    print(f"[{name}] loaded BEST checkpoint from epoch {bep} (val loss {bvl:.6f})")

    tloss, yts, yps = evaluate(model, test_loader)
    yto, ypo = to_original_units(yts), to_original_units(yps)
    per_t = metrics_all(yts, yps)
    TEST_RESULTS[mk] = per_t
    TEST_PRED[mk] = (yto, ypo)

    print("\n" + "=" * 60)
    print(f"FINAL TEST RESULTS - {name}")
    print("=" * 60)
    print(f"(test loss in scaled target space: {tloss:.6f})")
    for t in TARGETS:
        print(f"\n{t}")
        for k in METRIC_NAMES:
            print(f"{k}: {per_t[t][k]:.6f}")
    print()

    with open(mdir(mk, "metrics", "test_metrics.json"), "w") as f:
        json.dump({"best_epoch": bep, "best_val_loss": bvl,
                   "test_loss_scaled_space": tloss, "metrics": per_t}, f, indent=2)

    pdf = pd.DataFrame({"index": np.arange(len(yto)),
                        "DATE": pd.to_datetime(dt_test)})
    for i, t in enumerate(TARGETS):
        pdf[f"actual_{t}"] = yto[:, i]
        pdf[f"predicted_{t}"] = ypo[:, i]
    pdf.to_csv(mdir(mk, "predictions", "test_predictions.csv"), index=False)
    print("Saved:", mdir(mk, "predictions", "test_predictions.csv"))

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
''')

md(r"""## 20. Test metrics CSV and model comparison""")

code(r'''rows = []
for mk in MODELS_TO_RUN:
    for t in TARGETS:
        r = {"model": MODEL_DISPLAY[mk], "target": t}
        r.update({k: TEST_RESULTS[mk][t][k] for k in METRIC_NAMES})
        rows.append(r)

test_metrics_df = pd.DataFrame(rows, columns=["model", "target"] + METRIC_NAMES)
test_metrics_df.to_csv(OUT_P / "comparison" / "test_metrics.csv", index=False)
test_metrics_df.to_csv(OUT_P / "comparison" / "model_comparison.csv", index=False)

print("Saved:", OUT_P / "comparison" / "test_metrics.csv")
print("Saved:", OUT_P / "comparison" / "model_comparison.csv")
print("\nTarget-wise results kept separate (never averaged into one score).\n")
with pd.option_context("display.max_columns", None, "display.width", 250):
    print(test_metrics_df.to_string(index=False))
''')

# ---------------- plots ----------------
md(r"""## 21. Plots — training curves, actual vs predicted, residuals, comparison

All figures: font size 20, DPI 300, saved to each model's `plots/` folder
(rendered to file rather than inline to keep the notebook small).""")

code(r'''def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def training_curves(mk):
    hp = mdir(mk, "metrics", "history.csv")
    if not hp.exists():
        print(f"  no history for {MODEL_DISPLAY[mk]}"); return
    h = pd.read_csv(hp); name = MODEL_DISPLAY[mk]; od = mdir(mk, "plots")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(h.epoch, h.train_loss, lw=2.5, label="Train")
    ax.plot(h.epoch, h.val_loss, lw=2.5, label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (SmoothL1)")
    ax.set_title(f"{name} - Loss"); ax.legend(); ax.grid(alpha=.3)
    _save(fig, od / "loss_curve.png")

    for metric in ("R2", "MAE", "RMSE"):
        for t in TARGETS:
            tc, vc = f"train_{t}_{metric}", f"val_{t}_{metric}"
            if tc not in h.columns:
                continue
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(h.epoch, h[tc], lw=2.5, label="Train")
            ax.plot(h.epoch, h[vc], lw=2.5, label="Validation")
            ax.set_xlabel("Epoch"); ax.set_ylabel(metric)
            ax.set_title(f"{name}\n{t} - {metric}", fontsize=20)
            ax.legend(); ax.grid(alpha=.3)
            _save(fig, od / f"{metric.lower()}_curve_{t}.png")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(h.epoch, h.learning_rate, lw=2.5, color="green")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate"); ax.set_yscale("log")
    ax.set_title(f"{name} - Learning Rate"); ax.grid(alpha=.3)
    _save(fig, od / "learning_rate.png")
    print(f"  {name}: training curves saved")


def actual_vs_pred(mk):
    yto, ypo = TEST_PRED[mk]; name = MODEL_DISPLAY[mk]; od = mdir(mk, "plots")
    dts = pd.to_datetime(dt_test)
    for i, t in enumerate(TARGETS):
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.plot(dts, yto[:, i], lw=2.0, label="Actual")
        ax.plot(dts, ypo[:, i], lw=2.0, label="Predicted", alpha=.85)
        ax.set_xlabel("Test date (chronological)"); ax.set_ylabel(t)
        ax.set_title(f"{name}\n{t} - Actual vs Predicted", fontsize=20)
        ax.legend(); ax.grid(alpha=.3)
        fig.autofmt_xdate()
        _save(fig, od / f"test_actual_vs_predicted_{t}.png")
    print(f"  {name}: actual-vs-predicted saved")


def residuals(mk):
    yto, ypo = TEST_PRED[mk]; name = MODEL_DISPLAY[mk]; od = mdir(mk, "plots")
    res = yto - ypo; dts = pd.to_datetime(dt_test)
    for i, t in enumerate(TARGETS):
        r = res[:, i]
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.plot(dts, r, lw=1.8); ax.axhline(0, ls="--", lw=2, color="black")
        ax.set_xlabel("Test date (chronological)")
        ax.set_ylabel("Residual (actual - predicted)")
        ax.set_title(f"{name}\n{t} - Residual over test period", fontsize=20)
        ax.grid(alpha=.3); fig.autofmt_xdate()
        _save(fig, od / f"residual_over_time_{t}.png")

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(r, bins=40, edgecolor="black")
        ax.axvline(0, ls="--", lw=2, color="black")
        ax.set_xlabel("Residual (actual - predicted)"); ax.set_ylabel("Frequency")
        ax.set_title(f"{name}\n{t} - Residual distribution", fontsize=20)
        ax.grid(alpha=.3)
        _save(fig, od / f"residual_distribution_{t}.png")
    print(f"  {name}: residual plots saved")


print("Training curves:")
for mk in MODELS_TO_RUN: training_curves(mk)
print("Actual vs predicted:")
for mk in MODELS_TO_RUN: actual_vs_pred(mk)
print("Residual analysis (descriptive diagnostics only - no distributional claim):")
for mk in MODELS_TO_RUN: residuals(mk)
''')

code(r'''cmp_dir = OUT_P / "comparison" / "plots"

def comparison_plot(metric):
    for t in TARGETS:
        names = [MODEL_DISPLAY[mk] for mk in MODELS_TO_RUN]
        vals = [TEST_RESULTS[mk][t][metric] for mk in MODELS_TO_RUN]
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(names, vals, edgecolor="black")
        ax.set_ylabel(metric); ax.set_title(f"{t}\nTest {metric} by model", fontsize=20)
        ax.grid(alpha=.3, axis="y"); ax.tick_params(axis="x", rotation=15)
        fin = [v for v in vals if np.isfinite(v)]
        if fin:
            span = (max(fin) - min(min(fin), 0)) or 1.0
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width()/2, v + .02*span,
                            f"{v:.4g}", ha="center", va="bottom", fontsize=16)
        _save(fig, cmp_dir / f"comparison_{metric}_{t}.png")

for m in ("R2", "MAE", "RMSE"):
    comparison_plot(m); print(f"  {m} comparison saved")
print("Saved to:", cmp_dir)
''')

# ---------------- summary + manifest ----------------
md(r"""## 22. Final summary table

Every value below comes from actual test predictions produced by each model's
best checkpoint on the untouched test period (2024-07-01 … 2025-12-31).""")

code(r'''print("=" * 72)
print("FINAL TEST SUMMARY - all models, both targets")
print("=" * 72 + "\n")
with pd.option_context("display.max_columns", None, "display.width", 300,
                       "display.float_format", lambda v: f"{v:,.4f}"):
    print(test_metrics_df.to_string(index=False))

print("\nInterpretation notes:")
print("  - 8 primary metrics are in ORIGINAL dollar units.")
print("  - MAE_scaled / RMSE_scaled divide by the TRAIN target std:")
for i, t in enumerate(TARGETS):
    print(f"      {t}: train_std = {TRAIN_TARGET_STD[i]:,.4f}")
print(f"  - MAPE denominator floor = {MAPE_EPSILON}; both targets are >> 0 here, "
      "so MAPE is well behaved.")
print(f"  - Test period: {pd.to_datetime(dt_test).min().date()} .. "
      f"{pd.to_datetime(dt_test).max().date()} ({len(dt_test)} days)")

print("\nBest-epoch summary:")
for mk in MODELS_TO_RUN:
    info = json.loads((mdir(mk, "checkpoints", "best", "best_model_info.json")).read_text())
    h = pd.read_csv(mdir(mk, "metrics", "history.csv"))
    print(f"  {MODEL_DISPLAY[mk]:18s} best_epoch={info['best_epoch']:3d} "
          f"best_val_loss={info['best_validation_loss']:.6f} "
          f"epochs_run={len(h)}")
''')

md(r"""## 23. Output manifest""")

code(r'''EXPECTED = {
    "latest checkpoint": ("checkpoints", "latest", "latest.pt"),
    "best checkpoint": ("checkpoints", "best", "best.pt"),
    "best_model_info.json": ("checkpoints", "best", "best_model_info.json"),
    "history.csv": ("metrics", "history.csv"),
    "test_metrics.json": ("metrics", "test_metrics.json"),
    "test_predictions.csv": ("predictions", "test_predictions.csv"),
    "config.json": ("config", "config.json"),
    "model_summary.json": ("config", "model_summary.json"),
}

print("=" * 60)
print("EXPERIMENT OUTPUT MANIFEST")
print("=" * 60)
for mk in MODEL_KEYS:
    print(f"\n{MODEL_DISPLAY[mk]}")
    if mk not in MODELS_TO_RUN:
        print("  (not run)"); continue
    for lbl, parts in EXPECTED.items():
        print(f"  [{'OK' if mdir(mk, *parts).exists() else 'MISSING'}] {lbl}")
    pngs = sorted(mdir(mk, "plots").glob("*.png"))
    print(f"  [{'OK' if pngs else 'MISSING'}] plots ({len(pngs)} png)")
    for p in pngs:
        print(f"        {p.name}")

print("\nCOMPARISON")
for lbl, p in [("test_metrics.csv", OUT_P/"comparison"/"test_metrics.csv"),
               ("model_comparison.csv", OUT_P/"comparison"/"model_comparison.csv"),
               ("model_summary.json", OUT_P/"comparison"/"model_summary.json")]:
    print(f"  [{'OK' if p.exists() else 'MISSING'}] {lbl}")
cp = sorted((OUT_P/"comparison"/"plots").glob("*.png"))
print(f"  [{'OK' if cp else 'MISSING'}] comparison plots ({len(cp)} png)")
for p in cp:
    print(f"        {p.name}")
print("\nAll outputs under:", OUT_P.resolve())
''')

md(r"""## 24. Interpretation boundaries

Three categories that must not be conflated in the write-up:

**1. Paper facts.** Verified: both works exist, their venues, and their
high-level composition (CNN/TCN+LSTM hybrids compared in the first; LSTM +
modified Transformer + MLP in the second). Their *reported* results belong to
*their* datasets (traffic, air quality, financial series) and are not comparable
to results here.

**2. Implementation decisions.** Every layer size, channel/kernel/dilation
choice, head count, dropout rate, the shared `SmoothL1Loss`, `AdamW`,
`ReduceLROnPlateau` settings, gradient clipping, and the pre-norm reconstruction
of the "modified Transformer". None is claimed to be paper-specified.

**3. Experimental results.** Whatever the tables and plots above show. There is
no target R² and no threshold any model is expected to clear. Regression
forecasting has no generic "accuracy" metric, so none is reported.

A benchmark beating the proposed model is as legitimate a result as the reverse,
and should be reported as such.

### Known caveat worth stating in the paper

Both targets shift upward across the splits (train mean revenue ≈ \$143k → val
≈ \$173k → test ≈ \$183k; expenses ≈ \$101k → \$110k → \$114k). The target
transform is fitted on train only — correct for leakage avoidance, but it means
every model extrapolates into a higher regime on test. This affects all models
equally and should be disclosed rather than presented as model weakness.
""")

nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "colab": {"provenance": [], "toc_visible": True},
    "accelerator": "GPU",
}
nb["cells"] = cells
nbf.write(nb, "../DeepBudgetVis_Benchmarks_RepoData.ipynb")
print("FULL notebook written. cells:", len(cells),
      "| code:", sum(1 for c in cells if c.cell_type == "code"),
      "| md:", sum(1 for c in cells if c.cell_type == "markdown"))
