"""
Builds DeepBudgetVis_Benchmarks_CNNLSTM_TCNLSTM_LSTMmTransMLP.ipynb

IMPORTANT: this builder writes the notebook with EMPTY outputs on every code
cell. No results are embedded, because the target dataset (the user's
Google-Drive preprocessed hospital dataset) is not accessible from the build
environment. Any embedded number would be fabricated, which is forbidden.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src):
    c = nbf.v4.new_code_cell(src)
    c['outputs'] = []
    c['execution_count'] = None
    cells.append(c)


# ============================================================
# 1. Title / research context
# ============================================================
md(r"""# DeepBudget-Vis — Literature-Based Benchmark Models

**Project:** *DeepBudget-Vis: A Hybrid Deep Learning Framework with Interactive
Visual Analytics for Hospital Revenue Cycle Forecasting and Budget Optimization*

This notebook implements **three literature-based benchmark forecasting models**
to compare against the proposed DeepBudget-Vis / HRC-DRGNet model:

1. **CNN-LSTM**
2. **TCN-LSTM**
3. **LSTM-mTrans-MLP**

## Terminology

These are referred to throughout as **literature-based benchmark models**
(equivalently, *literature-based SOTA comparison models*). They are **not**
claimed to be a universal state-of-the-art ranking — the literature survey does
not establish such a ranking across all forecasting research. They were selected
because they are strong deep-learning / hybrid forecasting architectures
represented in the project's literature survey and are reasonably adaptable to a
multivariate hospital financial time-series dataset.

## Targets

Exactly two targets, in this fixed output order:

| Output index | Target |
|---|---|
| `output[:, 0]` | `NET_PATIENT_REVENUE` |
| `output[:, 1]` | `TOT_OVERALL_EXP` |

## Fair-comparison contract

All three models share: the same chronological train/validation/test split, the
same sequence length, the same input features, the same target definitions and
scaling, the same loss, the same optimizer policy, the same early-stopping and
`ReduceLROnPlateau` policy, the same seed, and the same metric definitions.
**Only the architecture differs.**

## What this notebook does NOT contain

No pre-filled results. Every metric, plot and table is produced by actually
running the cells on your dataset. There are no example/placeholder numbers
anywhere in this notebook.
""")

md(r"""## Source verification note (read before citing)

I was able to verify that the two cited works exist and to confirm their
**high-level model composition**, but I could **not** retrieve their
implementation-level details (layer counts, filter widths, hidden sizes,
learning rates) — both publisher pages returned access errors during
preparation of this notebook.

**Verified:**

- Kabir et al., *"LSTM–Transformer-Based Robust Hybrid Deep Learning Model for
  Financial Time Series Forecasting"*, **Sci** 7(1), 7 (2025). The abstract
  states the work proposes the ensemble model **LSTM-mTrans-MLP**, integrating
  an LSTM network, a **modified Transformer** network, and a multilayered
  perceptron. Source:
  [mdpi.com/2413-4155/7/1/7](https://www.mdpi.com/2413-4155/7/1/7)
  (also indexed as a UALR thesis:
  [research.ualr.edu/etd/1235](https://research.ualr.edu/etd/1235)).
- A closely-matching work on hybrid multivariate forecasting compares exactly
  **CNN-LSTM, CNN-BiLSTM, TCN-LSTM and TCN-BiLSTM**, reporting TCN-BiLSTM as
  best overall on its two datasets (Traffic Volume R² ≈ 0.976, Air Quality
  R² ≈ 0.94). Source:
  [d-nb.info/1353813266/34](https://d-nb.info/1353813266/34). Those reported R²
  values belong to **that paper's own datasets** (traffic / air quality) and are
  **not** comparable to, or predictive of, results on this hospital dataset.

**Consequence:** every numeric architecture choice in this notebook (channel
counts, kernel sizes, dilation schedule, hidden sizes, number of heads, MLP
widths, dropout) is an **implementation assumption**, explicitly labelled as
such in the per-model fidelity sections. None of them is presented as a value
taken from the papers.

*Content from the sources above was paraphrased/summarised for licensing
compliance.*
""")

# ============================================================
# 2. Configuration
# ============================================================
md(r"""## 2. USER CONFIGURATION — the only section you should need to edit

Set your paths here. Everything after this section runs automatically.

The loader **discovers** your files rather than assuming filenames, and **fails
loudly** if anything required is missing. It never invents or substitutes data.
""")

code(r'''# ============================================================
# USER CONFIGURATION — EDIT THIS SECTION ONLY
# ============================================================

# ---- Paths -------------------------------------------------
BASE_DIR   = "/content/drive/MyDrive/GPT forecast"
DATA_DIR   = f"{BASE_DIR}/preprocessed_data"
OUTPUT_DIR = f"{BASE_DIR}/baseline_outputs_v1"

# ---- Targets (output order is fixed by this list) ----------
TARGETS = [
    "NET_PATIENT_REVENUE",   # -> output[:, 0]
    "TOT_OVERALL_EXP",       # -> output[:, 1]
]

# Accepted alternative column names, in case your preprocessed files use a
# variant spelling. The resolver prints exactly which column it matched.
# Leave as-is unless your dataset genuinely uses a different name.
TARGET_ALIASES = {
    "NET_PATIENT_REVENUE": [
        "NET_PATIENT_REVENUE",
    ],
    "TOT_OVERALL_EXP": [
        "TOT_OVERALL_EXP",
        "TOTAL_OVERALL_EXP",
        "TOT_OVERALL_EXPENSE",
        "TOTAL_OPERATING_EXP",   # present in some earlier project versions
    ],
}

# ---- Sequence / training ----------------------------------
SEQUENCE_LENGTH = 56
RANDOM_SEED     = 42
BATCH_SIZE      = 128
EPOCHS          = 100
PATIENCE        = 15
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4

# ---- Resume -----------------------------------------------
RESUME = True          # True -> continue from latest checkpoint if one exists

# ---- Target transform -------------------------------------
# "auto"          : use a scaler file if one is discovered, else assume raw units
# "none"          : y is already in original financial units
# "log1p_zscore"  : y_scaled = (log1p(y) - mu) / sigma, mu/sigma from a JSON file
# "sklearn"       : a pickled/joblib sklearn scaler with .inverse_transform
TARGET_TRANSFORM = "auto"

# ---- Metric options ---------------------------------------
# If True, run an extra no-dropout eval pass over the training set each epoch to
# compute train metrics (slower, but train metrics are then directly comparable
# to validation metrics). If False, train metrics are accumulated from the
# training forward pass, i.e. WITH dropout active.
TRAIN_METRICS_EVAL_PASS = False

MAPE_EPSILON = 1.0     # safe denominator floor for MAPE / sMAPE (target units)

# ---- Performance ------------------------------------------
USE_AMP         = True    # mixed precision on CUDA; auto-disabled on CPU
GRAD_CLIP_NORM  = 1.0     # None to disable
NUM_WORKERS     = 2

# ---- Scheduler (implementation configuration, not from the papers) ----
SCHED_FACTOR   = 0.5
SCHED_PATIENCE = 5
SCHED_MIN_LR   = 1e-7

# ---- Which models to run ----------------------------------
MODELS_TO_RUN = ["cnn_lstm", "tcn_lstm", "lstm_mtrans_mlp"]

# ============================================================
# END OF USER CONFIGURATION
# ============================================================
print("Configuration loaded.")
print("BASE_DIR  :", BASE_DIR)
print("DATA_DIR  :", DATA_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("TARGETS   :", TARGETS)
''')

# ============================================================
# 3. Drive mount
# ============================================================
md(r"""## 3. Google Drive mount

Skipped automatically if not running in Colab.""")

code(r'''IN_COLAB = False
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted.")
else:
    print("Not running in Colab - skipping Drive mount.")
    print("Ensure BASE_DIR points at a locally reachable path.")
''')

# ============================================================
# 4. Dependencies
# ============================================================
md(r"""## 4. Dependencies

Colab already ships everything needed. This cell only verifies versions and
installs nothing unless a package is genuinely missing.""")

code(r'''import importlib, subprocess, sys

REQUIRED = ["numpy", "pandas", "torch", "sklearn", "matplotlib", "joblib"]
missing = []
for pkg in REQUIRED:
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing.append("scikit-learn" if pkg == "sklearn" else pkg)

if missing:
    print("Installing missing packages:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("All required packages already available.")
''')

# ============================================================
# 5. Imports + seeding + device
# ============================================================
md(r"""## 5. Imports, reproducibility, device""")

code(r'''import os
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils import weight_norm

from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    median_absolute_error,
    max_error,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Publication-quality plot defaults (fontsize 20, dpi 300) ----
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def set_seed(seed: int):
    """Seed python, numpy and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic where practical. cudnn.deterministic can slow training;
    # documented as an implementation choice.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print("Device: CUDA")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Device: CPU")
    print("NOTE: CPU training will be slow; fine for debugging.")

AMP_ENABLED = bool(USE_AMP and DEVICE.type == "cuda")
print("AMP enabled:", AMP_ENABLED)
print("Seed:", RANDOM_SEED)
print("torch:", torch.__version__)
''')

# ============================================================
# 6. Dataset discovery
# ============================================================
md(r"""## 6. Dataset discovery

This notebook does **not** assume your filenames. It inspects `DATA_DIR` and
reports everything it finds, then picks a supported layout:

- **Layout A** — a single `.npz` containing train/val/test arrays
- **Layout B** — separate `.npy` files (e.g. `X_train.npy`, `y_train.npy`, …)
- **Layout C** — flat `.csv` files (`train`/`val`/`test`), from which causal
  sequences are constructed

If none of these can be resolved, it raises a clear error. It never fabricates
or downloads data.""")

code(r'''DATA_DIR_P = Path(DATA_DIR)
OUTPUT_DIR_P = Path(OUTPUT_DIR)

if not DATA_DIR_P.exists():
    raise FileNotFoundError(
        f"DATA_DIR does not exist: {DATA_DIR_P}\n"
        "Fix DATA_DIR in the USER CONFIGURATION cell. "
        "No data will be created or downloaded automatically."
    )

print("=" * 70)
print("DATASET DISCOVERY")
print("=" * 70)
print("Scanning:", DATA_DIR_P, "\n")

found_files = sorted(
    [p for p in DATA_DIR_P.rglob("*") if p.is_file()],
    key=lambda p: str(p).lower(),
)
if not found_files:
    raise FileNotFoundError(f"DATA_DIR is empty: {DATA_DIR_P}")

by_ext = {}
for p in found_files:
    by_ext.setdefault(p.suffix.lower(), []).append(p)

for ext, paths in sorted(by_ext.items()):
    print(f"  {ext or '<no ext>'}  ({len(paths)} file(s))")
    for p in paths[:25]:
        size_mb = p.stat().st_size / 1e6
        print(f"      {p.relative_to(DATA_DIR_P)}   [{size_mb:.2f} MB]")
    if len(paths) > 25:
        print(f"      ... and {len(paths) - 25} more")
print()


def _match(paths, *must_contain):
    """Return paths whose lowercase name contains all given substrings."""
    out = []
    for p in paths:
        name = p.name.lower()
        if all(tok in name for tok in must_contain):
            out.append(p)
    return out


NPZ_FILES = by_ext.get(".npz", [])
NPY_FILES = by_ext.get(".npy", [])
CSV_FILES = by_ext.get(".csv", [])

SPLIT_TOKENS = {
    "train": ["train"],
    "val":   ["val", "valid", "validation", "dev"],
    "test":  ["test", "holdout"],
}


def _find_split_file(paths, split, extra=None):
    """Find one file for a split, optionally also requiring `extra` token."""
    for tok in SPLIT_TOKENS[split]:
        toks = [tok] + ([extra] if extra else [])
        hits = _match(paths, *toks)
        if hits:
            hits.sort(key=lambda p: len(p.name))
            return hits[0]
    return None


LAYOUT = None
DATA_SOURCE_DESC = ""

# ---------- Layout A: single npz with all splits ----------
for npz_path in NPZ_FILES:
    with np.load(npz_path, allow_pickle=True) as z:
        keys = set(z.files)
    lower = {k.lower(): k for k in keys}
    has_train_x = any(("train" in k and k.startswith(("x", "feat"))) for k in lower)
    has_train_y = any(("train" in k and k.startswith(("y", "targ", "label"))) for k in lower)
    if has_train_x and has_train_y:
        LAYOUT = "A_npz"
        NPZ_PATH = npz_path
        DATA_SOURCE_DESC = f"single .npz archive: {npz_path.name}"
        print(f"[DETECTED] Layout A - single npz with splits: {npz_path.name}")
        print("           keys:", sorted(keys))
        break

# ---------- Layout B: separate npy files ----------
if LAYOUT is None and NPY_FILES:
    cand = {}
    ok = True
    for split in ("train", "val", "test"):
        xp = _find_split_file(NPY_FILES, split, "x") or _find_split_file(NPY_FILES, split, "feat")
        yp = _find_split_file(NPY_FILES, split, "y") or _find_split_file(NPY_FILES, split, "targ")
        if xp is None or yp is None:
            ok = False
            break
        cand[split] = (xp, yp)
    if ok:
        LAYOUT = "B_npy"
        NPY_MAP = cand
        DATA_SOURCE_DESC = "separate .npy arrays per split"
        print("[DETECTED] Layout B - separate .npy arrays:")
        for s, (xp, yp) in cand.items():
            print(f"           {s}: X={xp.name}  y={yp.name}")

# ---------- Layout C: flat csv per split ----------
if LAYOUT is None and CSV_FILES:
    cand = {}
    ok = True
    for split in ("train", "val", "test"):
        p = _find_split_file(CSV_FILES, split)
        # avoid picking helper files like *_with_context.csv when a plain one exists
        if p is None:
            ok = False
            break
        cand[split] = p
    if ok:
        LAYOUT = "C_csv"
        CSV_MAP = cand
        DATA_SOURCE_DESC = "flat .csv per split (sequences constructed in-notebook)"
        print("[DETECTED] Layout C - flat CSV per split:")
        for s, p in cand.items():
            print(f"           {s}: {p.name}")

if LAYOUT is None:
    raise FileNotFoundError(
        "Could not resolve a supported dataset layout in DATA_DIR.\n\n"
        "Supported layouts:\n"
        "  A) one .npz containing X_train/y_train/X_val/y_val/X_test/y_test\n"
        "  B) separate .npy files, e.g. X_train.npy, y_train.npy, ...\n"
        "  C) flat CSVs named like train.csv / val.csv / test.csv\n\n"
        f"Files actually found in {DATA_DIR_P}:\n  "
        + "\n  ".join(str(p.relative_to(DATA_DIR_P)) for p in found_files[:60])
    )

print("\nResolved layout:", LAYOUT, "|", DATA_SOURCE_DESC)
''')

# ============================================================
# 7. Load arrays
# ============================================================
md(r"""## 7. Load / construct sequences

For array layouts (A, B) the supplied sequences are used **as provided** — they
are not rebuilt. For the CSV layout (C), sequences are constructed **causally**:
the window `[t-SEQUENCE_LENGTH, t-1]` predicts the target at `t`, so no future
observation ever enters an input window.""")

code(r'''def _resolve_target_columns(available_cols):
    """Map each configured TARGET to an actual column name, via TARGET_ALIASES."""
    resolved = {}
    avail_lower = {c.lower(): c for c in available_cols}
    for canonical in TARGETS:
        hit = None
        for alias in TARGET_ALIASES.get(canonical, [canonical]):
            if alias in available_cols:
                hit = alias
                break
            if alias.lower() in avail_lower:
                hit = avail_lower[alias.lower()]
                break
        if hit is None:
            raise KeyError(
                f"Target '{canonical}' not found in the dataset.\n"
                f"Aliases tried: {TARGET_ALIASES.get(canonical, [canonical])}\n"
                f"Available columns ({len(available_cols)}): "
                f"{list(available_cols)[:80]}\n\n"
                "Add the correct name to TARGET_ALIASES in the configuration cell. "
                "Targets will NOT be silently substituted."
            )
        resolved[canonical] = hit
    return resolved


def _build_sequences(feat_arr, targ_arr, seq_len):
    """Causal windowing: X[i] = feats[i : i+seq_len], y[i] = targs[i+seq_len]."""
    n = len(feat_arr)
    if n <= seq_len:
        raise ValueError(
            f"Split has {n} rows, which is <= SEQUENCE_LENGTH ({seq_len}). "
            "Cannot construct a single window."
        )
    xs = np.stack([feat_arr[i:i + seq_len] for i in range(n - seq_len)])
    ys = np.stack([targ_arr[i + seq_len] for i in range(n - seq_len)])
    return xs.astype(np.float32), ys.astype(np.float32)


TARGET_COL_MAP = None
splits = {}

if LAYOUT == "A_npz":
    with np.load(NPZ_PATH, allow_pickle=True) as z:
        keys = list(z.files)
        lower = {k.lower(): k for k in keys}

        def pick(split, kind):
            prefixes = ("x", "feat") if kind == "x" else ("y", "targ", "label")
            for lk, orig in lower.items():
                if split in lk and lk.startswith(prefixes):
                    return orig
            for tok in SPLIT_TOKENS[split]:
                for lk, orig in lower.items():
                    if tok in lk and lk.startswith(prefixes):
                        return orig
            raise KeyError(f"No '{kind}' array for split '{split}' in {NPZ_PATH.name}. Keys: {keys}")

        for split in ("train", "val", "test"):
            splits[split] = (
                np.asarray(z[pick(split, "x")], dtype=np.float32),
                np.asarray(z[pick(split, "y")], dtype=np.float32),
            )

elif LAYOUT == "B_npy":
    for split, (xp, yp) in NPY_MAP.items():
        splits[split] = (
            np.load(xp).astype(np.float32),
            np.load(yp).astype(np.float32),
        )

elif LAYOUT == "C_csv":
    frames = {s: pd.read_csv(p) for s, p in CSV_MAP.items()}
    TARGET_COL_MAP = _resolve_target_columns(frames["train"].columns)
    print("Resolved target columns:")
    for k, v in TARGET_COL_MAP.items():
        note = "" if k == v else "   <-- matched via alias"
        print(f"   {k}  ->  '{v}'{note}")

    target_cols = [TARGET_COL_MAP[t] for t in TARGETS]

    # Feature columns: numeric, excluding the targets themselves.
    drop_like = set(target_cols)
    feat_cols = [
        c for c in frames["train"].columns
        if c not in drop_like and pd.api.types.is_numeric_dtype(frames["train"][c])
    ]
    if not feat_cols:
        raise ValueError("No numeric feature columns found after excluding targets.")

    print(f"\nUsing {len(feat_cols)} numeric feature columns from the CSVs.")
    print("NOTE: targets are excluded from the feature matrix to avoid trivially "
          "leaking the label at time t. Lagged target columns already present in "
          "your preprocessed CSVs (if any) are retained as legitimate history.")

    for split, df in frames.items():
        missing = [c for c in feat_cols + target_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Split '{split}' is missing columns: {missing[:20]}")
        splits[split] = _build_sequences(
            df[feat_cols].to_numpy(np.float32),
            df[target_cols].to_numpy(np.float32),
            SEQUENCE_LENGTH,
        )

X_train, y_train = splits["train"]
X_val,   y_val   = splits["val"]
X_test,  y_test  = splits["test"]

print("\n" + "=" * 70)
print("LOADED ARRAY SHAPES")
print("=" * 70)
print("Train X shape:          ", X_train.shape)
print("Train y shape:          ", y_train.shape)
print("Validation X shape:     ", X_val.shape)
print("Validation y shape:     ", y_val.shape)
print("Test X shape:           ", X_test.shape)
print("Test y shape:           ", y_test.shape)
''')

code(r'''# ---- Normalise dimensionality and derive shape constants ----
def _as_3d(X, name):
    if X.ndim == 3:
        return X
    raise ValueError(
        f"{name} must be 3-D [samples, sequence_length, features], got shape {X.shape}. "
        "If your arrays are 2-D, they are not sequenced; use the CSV layout so this "
        "notebook can build causal windows, or pre-sequence them yourself."
    )


X_train = _as_3d(X_train, "X_train")
X_val   = _as_3d(X_val,   "X_val")
X_test  = _as_3d(X_test,  "X_test")


def _as_2d_targets(y, name):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError(f"{name} must be 2-D [samples, n_targets], got {y.shape}")
    return y


y_train = _as_2d_targets(y_train, "y_train")
y_val   = _as_2d_targets(y_val,   "y_val")
y_test  = _as_2d_targets(y_test,  "y_test")

SEQ_LEN_ACTUAL = X_train.shape[1]
INPUT_DIM      = X_train.shape[-1]     # never hard-coded
N_TARGETS      = len(TARGETS)

if SEQ_LEN_ACTUAL != SEQUENCE_LENGTH:
    warnings.warn(
        f"Configured SEQUENCE_LENGTH={SEQUENCE_LENGTH} but the loaded arrays have "
        f"sequence length {SEQ_LEN_ACTUAL}. Using the DATA's value ({SEQ_LEN_ACTUAL}) "
        "so the supplied preprocessing is not silently overridden."
    )
    SEQUENCE_LENGTH = SEQ_LEN_ACTUAL

# If y has more columns than we need, select the target columns explicitly.
if y_train.shape[1] != N_TARGETS:
    if TARGET_COL_MAP is None and y_train.shape[1] > N_TARGETS:
        raise ValueError(
            f"y arrays have {y_train.shape[1]} columns but {N_TARGETS} targets are "
            f"configured ({TARGETS}), and the array layout carries no column names, "
            "so the correct columns cannot be identified safely.\n"
            "Either supply 2-column y arrays in the order "
            f"{TARGETS}, or use the CSV layout so columns can be resolved by name."
        )
    raise ValueError(
        f"y arrays have {y_train.shape[1]} columns, expected {N_TARGETS}: {TARGETS}"
    )

print("Number of input features:", INPUT_DIM)
print("Sequence length:         ", SEQUENCE_LENGTH)
print("Number of targets:       ", N_TARGETS)
print()
print("TARGET INDEX MAPPING (enforced everywhere in this notebook):")
for i, t in enumerate(TARGETS):
    print(f"   output[:, {i}] = {t}")
''')

# ============================================================
# 8. Target transform discovery
# ============================================================
md(r"""## 8. Target transform / scaler resolution

Metrics must be reported in a well-defined target space. This cell determines
whether the supplied `y` arrays are already in original financial units, or
whether a scaler must be inverted first.

**No scaler is ever fitted on validation or test data.** If a transform has to
be derived here, it is derived from **train only**.""")

code(r'''import joblib

SCALER_OBJ = None
TARGET_MU = None
TARGET_SIGMA = None
RESOLVED_TRANSFORM = "none"

scaler_candidates = [
    p for p in found_files
    if p.suffix.lower() in {".joblib", ".pkl", ".pickle", ".json"}
    and any(tok in p.name.lower() for tok in
            ("scaler", "target", "transform", "meta", "preprocess", "norm"))
]

print("Candidate scaler / metadata files:")
if scaler_candidates:
    for p in scaler_candidates:
        print("   ", p.relative_to(DATA_DIR_P))
else:
    print("    (none found)")
print()

def _try_load_sklearn_scaler():
    for p in scaler_candidates:
        if p.suffix.lower() in {".joblib", ".pkl", ".pickle"}:
            try:
                obj = joblib.load(p)
            except Exception as e:
                print(f"    could not load {p.name}: {e}")
                continue
            if hasattr(obj, "inverse_transform"):
                n_feat = getattr(obj, "n_features_in_", None)
                if n_feat is not None and int(n_feat) != N_TARGETS:
                    print(f"    skipping {p.name}: expects {n_feat} cols, need {N_TARGETS}")
                    continue
                print(f"    using sklearn-style target scaler: {p.name}")
                return obj
    return None


def _try_load_mu_sigma():
    for p in scaler_candidates:
        if p.suffix.lower() != ".json":
            continue
        try:
            meta = json.loads(p.read_text())
        except Exception:
            continue
        for mk, sk in (("mean", "std"), ("mu", "sigma"),
                       ("target_mean", "target_std"), ("log_mean", "log_std")):
            if mk in meta and sk in meta:
                mu = np.asarray(meta[mk], dtype=np.float64).ravel()
                sg = np.asarray(meta[sk], dtype=np.float64).ravel()
                if mu.size == N_TARGETS and sg.size == N_TARGETS:
                    print(f"    using mean/std from {p.name}")
                    return mu, sg
    return None


if TARGET_TRANSFORM in ("auto", "sklearn"):
    SCALER_OBJ = _try_load_sklearn_scaler()
    if SCALER_OBJ is not None:
        RESOLVED_TRANSFORM = "sklearn"

if RESOLVED_TRANSFORM == "none" and TARGET_TRANSFORM in ("auto", "log1p_zscore"):
    ms = _try_load_mu_sigma()
    if ms is not None:
        TARGET_MU, TARGET_SIGMA = ms
        RESOLVED_TRANSFORM = "log1p_zscore"

if TARGET_TRANSFORM == "none":
    RESOLVED_TRANSFORM = "none"

if TARGET_TRANSFORM == "sklearn" and RESOLVED_TRANSFORM != "sklearn":
    raise FileNotFoundError(
        "TARGET_TRANSFORM='sklearn' but no loadable sklearn scaler with "
        f"inverse_transform and {N_TARGETS} columns was found in DATA_DIR."
    )
if TARGET_TRANSFORM == "log1p_zscore" and RESOLVED_TRANSFORM != "log1p_zscore":
    raise FileNotFoundError(
        "TARGET_TRANSFORM='log1p_zscore' but no JSON with matching mean/std "
        "for the targets was found in DATA_DIR."
    )


def to_original_units(y_arr):
    """Map model/target space -> original financial units."""
    y_arr = np.asarray(y_arr, dtype=np.float64)
    if RESOLVED_TRANSFORM == "sklearn":
        return SCALER_OBJ.inverse_transform(y_arr)
    if RESOLVED_TRANSFORM == "log1p_zscore":
        return np.expm1(y_arr * TARGET_SIGMA + TARGET_MU)
    return y_arr


print("Resolved target transform:", RESOLVED_TRANSFORM)
if RESOLVED_TRANSFORM == "none":
    print("  -> y arrays are treated as ALREADY being in original financial units.")
    print("     All 8 primary metrics are therefore computed directly on y.")
else:
    print("  -> y arrays are in transformed space; metrics are computed AFTER "
          "inverting to original units.")

# ---- Train-target std, used ONLY for MAE_scaled / RMSE_scaled ----
y_train_original = to_original_units(y_train)
TRAIN_TARGET_STD = y_train_original.std(axis=0, ddof=0)

if not np.all(np.isfinite(TRAIN_TARGET_STD)) or np.any(TRAIN_TARGET_STD <= 0):
    raise ValueError(
        f"Train target std is invalid: {TRAIN_TARGET_STD}. "
        "MAE_scaled / RMSE_scaled would be undefined."
    )

print("\nTrain-target std (original units), used for *_scaled metrics:")
for i, t in enumerate(TARGETS):
    print(f"   {t}: {TRAIN_TARGET_STD[i]:,.4f}")
print("\nDefinition: MAE_scaled = MAE / train_target_std ; "
      "RMSE_scaled = RMSE / train_target_std  (per target).")
print("The SAME train-derived std is used for train, validation and test, so the "
      "scaled metrics are comparable across splits and models.")
''')

# ============================================================
# 9. Validation checks
# ============================================================
md(r"""## 9. Automatic data validation

Structural problems raise an error. Nothing is silently repaired.""")

code(r'''print("=" * 70)
print("DATA VALIDATION")
print("=" * 70)

checks = []

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    checks.append((label, bool(condition), detail))
    print(f"[{status}] {label}" + (f"  -- {detail}" if detail else ""))
    return bool(condition)


check("Dataset found", len(found_files) > 0, f"{len(found_files)} files")
check("Train split found", X_train.shape[0] > 0, f"{X_train.shape[0]} sequences")
check("Validation split found", X_val.shape[0] > 0, f"{X_val.shape[0]} sequences")
check("Test split found", X_test.shape[0] > 0, f"{X_test.shape[0]} sequences")

for i, t in enumerate(TARGETS):
    src = TARGET_COL_MAP[t] if TARGET_COL_MAP else f"y column index {i}"
    check(f"Target {t} found", True, f"source: {src}")

check("Feature dimensions consistent",
      X_train.shape[-1] == X_val.shape[-1] == X_test.shape[-1],
      f"{X_train.shape[-1]} / {X_val.shape[-1]} / {X_test.shape[-1]}")
check("Sequence dimensions valid",
      X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == SEQUENCE_LENGTH,
      f"seq_len={SEQUENCE_LENGTH}")
check("No unexpected target mismatch",
      y_train.shape[1] == y_val.shape[1] == y_test.shape[1] == N_TARGETS,
      f"n_targets={N_TARGETS}")
check("Sample counts align X/y",
      X_train.shape[0] == y_train.shape[0]
      and X_val.shape[0] == y_val.shape[0]
      and X_test.shape[0] == y_test.shape[0])

finite_ok = True
for nm, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
                ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_nan or n_inf:
        finite_ok = False
        print(f"        !! {nm}: {n_nan} NaN, {n_inf} Inf")
check("No NaN/Inf in model input", finite_ok)

failed = [c for c in checks if not c[1]]
if failed:
    raise ValueError(
        "Data validation failed:\n  "
        + "\n  ".join(f"{lbl} ({dt})" for lbl, _, dt in failed)
        + "\n\nStructural issues are not auto-repaired. Fix the preprocessing "
          "or the configuration and re-run."
    )

print("\nAll validation checks passed.")
''')

# ============================================================
# 10. Output folder structure
# ============================================================
md(r"""## 10. Output folder structure""")

code(r'''MODEL_KEYS = ["cnn_lstm", "tcn_lstm", "lstm_mtrans_mlp"]
MODEL_DISPLAY = {
    "cnn_lstm": "CNN-LSTM",
    "tcn_lstm": "TCN-LSTM",
    "lstm_mtrans_mlp": "LSTM-mTrans-MLP",
}

SUBDIRS = ["checkpoints/latest", "checkpoints/best", "metrics",
           "predictions", "plots", "logs", "config"]

for mk in MODEL_KEYS:
    for sd in SUBDIRS:
        (OUTPUT_DIR_P / mk / sd).mkdir(parents=True, exist_ok=True)

(OUTPUT_DIR_P / "comparison" / "plots").mkdir(parents=True, exist_ok=True)


def mdir(model_key, *parts):
    """Path helper inside a model's output directory."""
    return OUTPUT_DIR_P / model_key / Path(*parts)


print("Output tree created under:", OUTPUT_DIR_P)
for mk in MODEL_KEYS:
    print(f"  {mk}/")
    for sd in SUBDIRS:
        print(f"      {sd}/")
print("  comparison/")
print("      plots/")
''')

# ============================================================
# 11. Dataloaders
# ============================================================
md(r"""## 11. Dataset / DataLoader construction

Only the **training** loader is shuffled. Shuffling training *windows* does not
break temporal integrity: each window is a self-contained
`(past → next-step target)` example, and the chronological split boundaries are
untouched. Validation and test loaders preserve original order so predictions
can be written out in chronological sequence.""")

code(r'''def make_loader(X, y, shuffle, batch_size=None):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    return DataLoader(
        ds,
        batch_size=batch_size or BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )


train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val,   shuffle=False)
test_loader  = make_loader(X_test,  y_test,  shuffle=False)

print(f"train batches: {len(train_loader)}  (shuffle=True)")
print(f"val   batches: {len(val_loader)}  (shuffle=False, order preserved)")
print(f"test  batches: {len(test_loader)}  (shuffle=False, order preserved)")

_xb, _yb = next(iter(train_loader))
print("\nSample batch  X:", tuple(_xb.shape), " y:", tuple(_yb.shape))
''')

# ============================================================
# 12. Metrics
# ============================================================
md(r"""## 12. Metric functions — exactly the ten required metrics

`MAE, RMSE, MAPE, sMAPE, R2, ExplainedVar, MedianAE, MaxError, MAE_scaled, RMSE_scaled`

Computed **separately for each target**, never combined into a single score.

- The 8 primary metrics are computed in **original financial units**.
- `MAPE` / `sMAPE` use a floored denominator (`MAPE_EPSILON`, in target units) so
  a zero actual can never cause a division by zero. Because hospital financial
  series can legitimately contain values near zero, MAPE should be read with
  that caveat — sMAPE and the scaled metrics are more robust here.
- `MAE_scaled = MAE / train_target_std`, `RMSE_scaled = RMSE / train_target_std`,
  using the **train-derived** std for every split.""")

code(r'''METRIC_NAMES = [
    "MAE", "RMSE", "MAPE", "sMAPE", "R2",
    "ExplainedVar", "MedianAE", "MaxError",
    "MAE_scaled", "RMSE_scaled",
]


def compute_metrics_single_target(y_true, y_pred, train_std):
    """All ten metrics for ONE target. Inputs are 1-D, in original units."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n_dropped = int((~mask).sum())
    if n_dropped:
        warnings.warn(f"Dropped {n_dropped} non-finite pair(s) before metrics.")
    y_true, y_pred = y_true[mask], y_pred[mask]

    if y_true.size == 0:
        return {m: float("nan") for m in METRIC_NAMES}

    err = y_true - y_pred
    abs_err = np.abs(err)

    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))

    denom = np.maximum(np.abs(y_true), MAPE_EPSILON)
    mape = float((abs_err / denom).mean() * 100.0)

    sdenom = np.maximum(np.abs(y_true) + np.abs(y_pred), MAPE_EPSILON)
    smape = float((2.0 * abs_err / sdenom).mean() * 100.0)

    # R2 / ExplainedVar are undefined for a constant target; report NaN.
    if np.var(y_true) <= 0 or y_true.size < 2:
        r2 = float("nan")
        evs = float("nan")
    else:
        r2 = float(r2_score(y_true, y_pred))
        evs = float(explained_variance_score(y_true, y_pred))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "R2": r2,
        "ExplainedVar": evs,
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "MaxError": float(max_error(y_true, y_pred)),
        "MAE_scaled": float(mae / train_std),
        "RMSE_scaled": float(rmse / train_std),
    }


def compute_metrics_all_targets(y_true_2d, y_pred_2d):
    """Returns {target_name: {metric: value}} in original units."""
    y_true_o = to_original_units(y_true_2d)
    y_pred_o = to_original_units(y_pred_2d)
    out = {}
    for i, t in enumerate(TARGETS):
        out[t] = compute_metrics_single_target(
            y_true_o[:, i], y_pred_o[:, i], TRAIN_TARGET_STD[i]
        )
    return out


def flatten_metrics(prefix, per_target):
    """{'train_NET_PATIENT_REVENUE_MAE': v, ...}"""
    flat = {}
    for t, md_ in per_target.items():
        for m, v in md_.items():
            flat[f"{prefix}_{t}_{m}"] = v
    return flat


# ---- self-test on synthetic values (verifies the maths, not the model) ----
_yt = np.array([[100.0, 50.0], [200.0, 60.0], [300.0, 70.0], [400.0, 80.0]])
_yp = _yt + np.array([[10.0, -5.0], [-10.0, 5.0], [10.0, -5.0], [-10.0, 5.0]])
_m = compute_metrics_single_target(_yt[:, 0], _yp[:, 0], 100.0)
assert abs(_m["MAE"] - 10.0) < 1e-9, _m
assert abs(_m["RMSE"] - 10.0) < 1e-9, _m
assert abs(_m["MAE_scaled"] - 0.1) < 1e-9, _m
assert abs(_m["MaxError"] - 10.0) < 1e-9, _m
print("Metric self-test passed (MAE/RMSE/MAE_scaled/MaxError verified on synthetic input).")
print("Metrics tracked:", METRIC_NAMES)
''')

# ============================================================
# 13-15 Models with fidelity docs
# ============================================================
md(r"""## 13. CNN-LSTM — Paper Fidelity

```
Paper:      "Enhanced Multivariate Time Series Forecasting"
Authors:    A. Mahmoud and A. Mohammed
Venue:      Neural Processing Letters, 56(5), p.223
Model:      CNN-LSTM (one of four compared: CNN-LSTM, CNN-BiLSTM,
            TCN-LSTM, TCN-BiLSTM)
```

**Architecture described by the paper (high level):** a multivariate time-series
input is passed through convolutional feature extraction to capture local
temporal patterns, then through an LSTM to model sequential/temporal
dependencies, then to a forecasting output.

**Implementation here:**
`Input [B,T,F] → Conv1d → ReLU → Conv1d → ReLU → Dropout → LSTM → last hidden
state → Linear → [B,2]`

**Implementation assumptions — not explicitly specified in the paper** (and not
verifiable from the sources I could access; see the Source verification note at
the top):

- 2 convolutional layers, 64 channels each, kernel size 3, `padding=1`
- LSTM: 2 layers, hidden size 128, unidirectional
- Dropout 0.2; final `Linear(128 → 2)`
- The final LSTM time step is used as the sequence representation

**Components intentionally NOT added** (to keep the benchmark faithful and
distinguishable): no bidirectional LSTM, no Transformer, no attention, no TCN
dilation, no residual connections.

**Note on convolution padding and leakage:** the convolution mixes neighbouring
time steps *within the input window*. Every time step in that window strictly
precedes the target time step, so this is **not** future leakage — the target at
time `t` is never visible to the encoder of window `[t-T, t-1]`.
""")

code(r'''class CNNLSTM(nn.Module):
    """CNN-LSTM benchmark. Conv feature extraction -> LSTM -> linear head."""

    def __init__(self, input_dim, n_targets,
                 conv_channels=64, kernel_size=3,
                 lstm_hidden=128, lstm_layers=2, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=pad)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            conv_channels, lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,           # explicitly unidirectional
        )
        self.head = nn.Linear(lstm_hidden, n_targets)

    def forward(self, x):                  # x: [B, T, F]
        h = x.transpose(1, 2)              # -> [B, F, T] for Conv1d
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        h = self.drop(h)
        h = h.transpose(1, 2)              # -> [B, T, C]
        out, _ = self.lstm(h)
        return self.head(out[:, -1, :])    # [B, n_targets]


print("CNNLSTM defined.")
''')

md(r"""## 14. TCN-LSTM — Paper Fidelity

```
Paper:      "Enhanced Multivariate Time Series Forecasting"
Authors:    A. Mahmoud and A. Mohammed
Venue:      Neural Processing Letters, 56(5), p.223
Model:      TCN-LSTM
```

**Architecture described by the paper (high level):** a Temporal Convolutional
Network extracts temporal/local patterns from the multivariate input; an LSTM
then models sequential dependencies; a forecasting head produces the output.

**Implementation here:** a standard TCN residual stack (dilated **causal**
convolutions with `Chomp1d` cropping, weight normalisation, ReLU, dropout, and a
`1×1` residual projection when channel counts differ), followed by an LSTM and a
linear head.

`Input [B,T,F] → [TCN residual block × 4, dilations 1,2,4,8] → LSTM → last
hidden state → Linear → [B,2]`

**Implementation assumptions — not explicitly specified in the paper:**

- 4 residual blocks, 64 channels each, kernel size 3, dilations `1, 2, 4, 8`
- Each block: 2 × (weight-normed causal Conv1d → Chomp → ReLU → Dropout) + residual
- LSTM: 2 layers, hidden size 128, unidirectional; dropout 0.2
- Final `Linear(128 → 2)`

**Components intentionally NOT added:** no Transformer, no attention, no
bidirectional LSTM. The convolutions are genuinely dilated and genuinely causal —
this is not a generic CNN.

**Causality:** `Chomp1d` removes the right-hand padding so output step `i` never
depends on input steps `> i`. Combined with the windowing scheme, no future
information can reach the prediction.
""")

code(r'''class Chomp1d(nn.Module):
    """Remove trailing padding so the convolution stays causal."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Standard TCN residual block: 2x (causal dilated conv -> chomp -> ReLU -> dropout)."""

    def __init__(self, n_in, n_out, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_in, n_out, kernel_size,
                                           padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_out, n_out, kernel_size,
                                           padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu_out = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, ch_out in enumerate(channels):
            ch_in = input_dim if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(ch_in, ch_out, kernel_size,
                                        dilation=2 ** i, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):     # x: [B, F, T]
        return self.network(x)


class TCNLSTM(nn.Module):
    """TCN-LSTM benchmark. Dilated causal TCN -> LSTM -> linear head."""

    def __init__(self, input_dim, n_targets,
                 tcn_channels=(64, 64, 64, 64), kernel_size=3,
                 lstm_hidden=128, lstm_layers=2, dropout=0.2):
        super().__init__()
        self.tcn = TemporalConvNet(input_dim, list(tcn_channels),
                                   kernel_size=kernel_size, dropout=dropout)
        self.lstm = nn.LSTM(
            tcn_channels[-1], lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(lstm_hidden, n_targets)

    def forward(self, x):                 # [B, T, F]
        h = self.tcn(x.transpose(1, 2))   # [B, C, T]
        h = h.transpose(1, 2)             # [B, T, C]
        out, _ = self.lstm(h)
        return self.head(out[:, -1, :])


print("TCNLSTM defined (Chomp1d + dilated causal convs + residuals).")
''')

md(r"""## 15. LSTM-mTrans-MLP — Paper Fidelity

```
Paper:      "LSTM-Transformer-Based Robust Hybrid Deep Learning Model for
             Financial Time Series Forecasting"
Authors:    Kabir et al.
Year:       2025
Venue:      Sci, 7(1), 7   (also indexed as a UALR thesis)
Model:      LSTM-mTrans-MLP
```

**Architecture described by the paper (high level, verified from the abstract):**
an ensemble/hybrid model integrating an **LSTM** network, a **modified
Transformer** network, and a **multilayered perceptron (MLP)**. The LSTM captures
sequential dependencies, the Transformer component captures long-range temporal
relationships, and the MLP handles nonlinear feature relationships and final
prediction.

**Implementation here:**
`Input [B,T,F] → LSTM → +positional encoding → mTrans encoder blocks (pre-norm
multi-head self-attention + residual, GELU feed-forward + residual) → masked mean
pooling over time → MLP → Linear → [B,2]`

> **Implementation assumption — not explicitly specified in the paper.**
> The paper describes the "modified Transformer" component conceptually, but I
> could not access implementation-level detail for the specific modification
> (the publisher page returned HTTP 403 during preparation of this notebook).
> This notebook therefore implements the component as a **pre-normalisation
> Transformer encoder** (LayerNorm before attention and before the feed-forward
> sub-layer, with residual connections around both, and a GELU activation in the
> feed-forward network), operating on the LSTM output sequence with sinusoidal
> positional encoding added.
>
> This is a documented, reasonable reconstruction — it is **not** presented as
> the paper's exact mechanism. If you can supply the paper's architecture
> subsection, this block can be revised to match it precisely.

**Further implementation assumptions:**

- LSTM: 2 layers, hidden size 128, unidirectional
- mTrans: 2 encoder blocks, `d_model = 128`, 4 heads, FFN dim 256, dropout 0.2
- Sinusoidal (non-learned) positional encoding
- Sequence aggregation: mean over time steps
- MLP head: `128 → 128 → 64 → 2` with GELU and dropout

**Components intentionally NOT added:** no CNN/TCN front end. The defining
`LSTM → mTrans → MLP` order is preserved.

**Attention masking:** self-attention here is *unmasked within the input window*.
Every step in that window precedes the target step, so attending across the
window does not expose future information relative to the prediction target.
""")

code(r'''class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div)[:, :pe[:, 1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):                             # [B, T, d_model]
        return x + self.pe[:, :x.size(1), :]


class MTransBlock(nn.Module):
    """Pre-norm Transformer encoder block.

    IMPLEMENTATION ASSUMPTION: pre-normalisation ordering and GELU feed-forward
    are this notebook's reconstruction of the paper's 'modified Transformer'
    component, since implementation-level detail was not accessible.
    """

    def __init__(self, d_model, n_heads=4, ffn_dim=256, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(a)                 # residual around attention
        h = self.norm2(x)
        x = x + self.drop2(self.ffn(h))       # residual around FFN
        return x


class LSTMmTransMLP(nn.Module):
    """LSTM-mTrans-MLP benchmark: LSTM -> modified Transformer -> MLP -> head."""

    def __init__(self, input_dim, n_targets,
                 lstm_hidden=128, lstm_layers=2,
                 n_blocks=2, n_heads=4, ffn_dim=256,
                 mlp_hidden=(128, 64), dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.posenc = SinusoidalPositionalEncoding(lstm_hidden)
        self.blocks = nn.ModuleList([
            MTransBlock(lstm_hidden, n_heads, ffn_dim, dropout)
            for _ in range(n_blocks)
        ])
        self.norm_out = nn.LayerNorm(lstm_hidden)

        mlp_layers, prev = [], lstm_hidden
        for hdim in mlp_hidden:
            mlp_layers += [nn.Linear(prev, hdim), nn.GELU(), nn.Dropout(dropout)]
            prev = hdim
        self.mlp = nn.Sequential(*mlp_layers)
        self.head = nn.Linear(prev, n_targets)

    def forward(self, x):                 # [B, T, F]
        h, _ = self.lstm(x)               # [B, T, H]
        h = self.posenc(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_out(h)
        h = h.mean(dim=1)                 # temporal aggregation
        return self.head(self.mlp(h))


print("LSTMmTransMLP defined (LSTM -> mTrans -> MLP).")
''')

# ============================================================
# 16-17 factory, shape tests, param counts
# ============================================================
md(r"""## 16. Model factory, shape validation, parameter counts

Each model is instantiated and given a real batch. Shapes must be exactly
`[B, SEQUENCE_LENGTH, INPUT_DIM] → [B, 2]`, otherwise the notebook stops.""")

code(r'''def build_model(model_key):
    """Instantiate a benchmark model by key. Identical I/O contract for all three."""
    if model_key == "cnn_lstm":
        return CNNLSTM(INPUT_DIM, N_TARGETS)
    if model_key == "tcn_lstm":
        return TCNLSTM(INPUT_DIM, N_TARGETS)
    if model_key == "lstm_mtrans_mlp":
        return LSTMmTransMLP(INPUT_DIM, N_TARGETS)
    raise KeyError(f"Unknown model key: {model_key}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("=" * 70)
print("SHAPE VALIDATION AND PARAMETER COUNTS")
print("=" * 70)

xb, yb = next(iter(train_loader))
B = xb.shape[0]
expected_in = (B, SEQUENCE_LENGTH, INPUT_DIM)
expected_out = (B, N_TARGETS)

if tuple(xb.shape) != expected_in:
    raise ValueError(f"Batch X shape {tuple(xb.shape)} != expected {expected_in}")
if tuple(yb.shape) != expected_out:
    raise ValueError(f"Batch y shape {tuple(yb.shape)} != expected {expected_out}")

MODEL_SUMMARY = {}
for mk in MODEL_KEYS:
    set_seed(RANDOM_SEED)
    m = build_model(mk).to(DEVICE)
    m.eval()
    with torch.no_grad():
        out = m(xb.to(DEVICE))
    if tuple(out.shape) != expected_out:
        raise ValueError(
            f"{MODEL_DISPLAY[mk]} produced {tuple(out.shape)}, expected {expected_out}. "
            "Stopping rather than reshaping silently."
        )
    if not torch.isfinite(out).all():
        raise ValueError(f"{MODEL_DISPLAY[mk]} produced non-finite values on the shape test.")

    n_par = count_parameters(m)
    MODEL_SUMMARY[mk] = {
        "model_name": MODEL_DISPLAY[mk],
        "input_shape": [SEQUENCE_LENGTH, INPUT_DIM],
        "output_shape": [N_TARGETS],
        "sequence_length": int(SEQUENCE_LENGTH),
        "n_features": int(INPUT_DIM),
        "trainable_parameters": int(n_par),
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "scheduler": (f"ReduceLROnPlateau(mode=min, factor={SCHED_FACTOR}, "
                      f"patience={SCHED_PATIENCE}, min_lr={SCHED_MIN_LR})"),
        "loss": "SmoothL1Loss",
        "device": str(DEVICE),
        "targets": TARGETS,
        "seed": RANDOM_SEED,
    }
    print(f"\n{MODEL_DISPLAY[mk]}")
    print(f"   Input  : [B, {SEQUENCE_LENGTH}, {INPUT_DIM}]")
    print(f"   Output : [B, {N_TARGETS}]   (shape test PASSED)")
    print(f"   {MODEL_DISPLAY[mk]} trainable parameters: {n_par:,}")
    del m

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

with open(OUTPUT_DIR_P / "comparison" / "model_summary.json", "w") as f:
    json.dump(MODEL_SUMMARY, f, indent=2)
for mk in MODEL_KEYS:
    with open(mdir(mk, "config", "model_summary.json"), "w") as f:
        json.dump(MODEL_SUMMARY[mk], f, indent=2)

print("\nmodel_summary.json written for every model.")
''')

md(r"""## 17. Model summary printout""")

code(r'''for mk in MODEL_KEYS:
    s = MODEL_SUMMARY[mk]
    print("=" * 70)
    print(s["model_name"])
    print("=" * 70)
    print(f"  Input dimensions        : [B, {s['sequence_length']}, {s['n_features']}]")
    print(f"  Output dimensions       : [B, {s['output_shape'][0]}]")
    print(f"  Sequence length         : {s['sequence_length']}")
    print(f"  Number of features      : {s['n_features']}")
    print(f"  Trainable parameters    : {s['trainable_parameters']:,}")
    print(f"  Optimizer               : {s['optimizer']}")
    print(f"  Learning rate           : {s['learning_rate']}")
    print(f"  Weight decay            : {s['weight_decay']}")
    print(f"  Batch size              : {s['batch_size']}")
    print(f"  Maximum epochs          : {s['max_epochs']}")
    print(f"  Early stopping patience : {s['early_stopping_patience']}")
    print(f"  Scheduler               : {s['scheduler']}")
    print(f"  Loss                    : {s['loss']}")
    print(f"  Device                  : {s['device']}")
    print(f"  Targets                 : {s['targets']}")
    print()
''')

# ============================================================
# 18. Loss + checkpoint utilities
# ============================================================
md(r"""## 18. Loss function

`nn.SmoothL1Loss` (Huber) is used for **all three** models. Rationale: hospital
financial series contain large-magnitude and occasionally extreme values, and a
Huber-type loss is less dominated by a handful of extreme residuals than pure
MSE while remaining a standard regression objective.

> **Implementation decision, not a paper-specified component.** Neither cited
> paper's loss specification was accessible to me. Using the *same* loss across
> all three models is required by the fair-comparison contract — the intended
> architectural difference is the model, not the objective.""")

code(r'''CRITERION = nn.SmoothL1Loss()
print("Loss:", CRITERION.__class__.__name__, "(identical for all three models)")
''')

md(r"""## 19. Checkpoint utilities

- `checkpoints/latest/latest.pt` — full resumable state, written every epoch
- `checkpoints/best/best.pt` — best **validation-loss** state only
- `checkpoints/best/best_model_info.json` — best epoch / loss / model name

Compatibility is verified on load (model name, feature dim, sequence length,
target count). An incompatible checkpoint raises rather than loading wrong
weights. **The test set is never used for checkpoint selection.**""")

code(r'''def _compat_signature(model_key):
    return {
        "model_name": MODEL_DISPLAY[model_key],
        "input_dim": int(INPUT_DIM),
        "sequence_length": int(SEQUENCE_LENGTH),
        "n_targets": int(N_TARGETS),
    }


def save_latest_checkpoint(model_key, epoch, model, optimizer, scheduler,
                           amp_scaler, best_val_loss, best_epoch, history, config):
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "history": history,
        "config": config,
        "compat": _compat_signature(model_key),
    }
    if amp_scaler is not None:
        payload["scaler_state_dict"] = amp_scaler.state_dict()

    path = mdir(model_key, "checkpoints", "latest", "latest.pt")
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)          # atomic-ish write: never corrupt a good file
    tmp.replace(path)


def save_best_checkpoint(model_key, epoch, model, val_loss, history, config):
    path = mdir(model_key, "checkpoints", "best", "best.pt")
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "val_loss": float(val_loss),
        "history": history,
        "config": config,
        "compat": _compat_signature(model_key),
    }, tmp)
    tmp.replace(path)

    with open(mdir(model_key, "checkpoints", "best", "best_model_info.json"), "w") as f:
        json.dump({
            "model_name": MODEL_DISPLAY[model_key],
            "best_epoch": int(epoch),
            "best_validation_loss": float(val_loss),
        }, f, indent=2)


def _verify_compat(model_key, ckpt, path):
    want = _compat_signature(model_key)
    got = ckpt.get("compat")
    if got is None:
        raise ValueError(
            f"Checkpoint {path} has no compatibility signature (written by an older "
            "version?). Delete it or set RESUME=False to start cleanly."
        )
    if got != want:
        raise ValueError(
            "Checkpoint is incompatible with the current dataset/model configuration.\n"
            f"  checkpoint: {got}\n  current   : {want}\n"
            f"  file      : {path}\n"
            "Refusing to load mismatched weights. Delete the checkpoint or set RESUME=False."
        )


def load_latest_checkpoint(model_key, model, optimizer, scheduler, amp_scaler):
    """Returns (start_epoch, best_val_loss, best_epoch, history) or None."""
    path = mdir(model_key, "checkpoints", "latest", "latest.pt")
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Could not read checkpoint {path}: {e}\n"
            "It may be corrupted. Delete it or set RESUME=False."
        ) from e

    _verify_compat(model_key, ckpt, path)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if amp_scaler is not None and "scaler_state_dict" in ckpt:
        amp_scaler.load_state_dict(ckpt["scaler_state_dict"])

    return (
        int(ckpt["epoch"]) + 1,
        float(ckpt["best_val_loss"]),
        int(ckpt.get("best_epoch", ckpt["epoch"])),
        list(ckpt.get("history", [])),
    )


def load_best_for_eval(model_key):
    """Load the BEST checkpoint for final test evaluation."""
    path = mdir(model_key, "checkpoints", "best", "best.pt")
    if not path.exists():
        raise FileNotFoundError(
            f"No best checkpoint for {MODEL_DISPLAY[model_key]} at {path}. "
            "Train the model before evaluating. The latest checkpoint is deliberately "
            "NOT substituted for final reporting."
        )
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    _verify_compat(model_key, ckpt, path)
    model = build_model(model_key).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, int(ckpt["epoch"]), float(ckpt["val_loss"])


print("Checkpoint utilities ready (latest / best / compat-verified / resumable).")
''')

# ============================================================
# 20. Training engine
# ============================================================
md(r"""## 20. Training utilities

Per epoch, all **ten** metrics are computed for **both** targets, on **train**
and **validation** — 40 metric values per epoch, plus `train_loss`, `val_loss`,
`learning_rate`, `epoch_time`.

Guards included: non-finite loss detection, optional gradient clipping, AMP with
resumable scaler state, and duplicate-epoch protection when appending to
`history.csv`.""")

code(r'''@torch.no_grad()
def evaluate_loader(model, loader):
    """Eval-mode pass. Returns (mean_loss, y_true_2d, y_pred_2d) in TARGET space."""
    model.eval()
    losses, ys, ps = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
            out = model(xb)
            loss = CRITERION(out, yb)
        losses.append(float(loss.detach().float().cpu()))
        ys.append(yb.detach().float().cpu().numpy())
        ps.append(out.detach().float().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(ys), np.concatenate(ps)


def print_epoch_report(display_name, epoch, total_epochs, train_loss, val_loss,
                       lr, train_metrics, val_metrics, epoch_time):
    print("=" * 60)
    print(f"{display_name} | Epoch {epoch}/{total_epochs}")
    print("=" * 60)
    print()
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Val Loss:   {val_loss:.6f}")
    print(f"LR:         {lr:.8f}")
    print(f"Epoch time: {epoch_time:.1f}s")
    for t in TARGETS:
        for split_label, md_ in (("TRAIN", train_metrics), ("VALIDATION", val_metrics)):
            print()
            print(f"{t} - {split_label}")
            for m in METRIC_NAMES:
                print(f"{m}: {md_[t][m]:.6f}")
    print()


def train_model(model_key, resume=None):
    """Full training loop for one benchmark model."""
    display = MODEL_DISPLAY[model_key]
    resume = RESUME if resume is None else resume

    set_seed(RANDOM_SEED)          # identical init policy for every model

    model = build_model(model_key).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHED_FACTOR,
        patience=SCHED_PATIENCE, min_lr=SCHED_MIN_LR,
    )
    amp_scaler = torch.amp.GradScaler(DEVICE.type) if AMP_ENABLED else None

    config = dict(MODEL_SUMMARY[model_key])
    config.update({
        "resume_requested": bool(resume),
        "amp_enabled": bool(AMP_ENABLED),
        "grad_clip_norm": GRAD_CLIP_NORM,
        "target_transform": RESOLVED_TRANSFORM,
        "train_metrics_eval_pass": bool(TRAIN_METRICS_EVAL_PASS),
        "mape_epsilon": MAPE_EPSILON,
        "data_source": DATA_SOURCE_DESC,
        "train_target_std": TRAIN_TARGET_STD.tolist(),
        "implementation_notes": [
            "Layer sizes are implementation assumptions, not paper-specified.",
            "SmoothL1Loss shared across all three models (implementation decision).",
            "AdamW + ReduceLROnPlateau are implementation configuration.",
            "Gradient clipping is a numerical-stability measure, not paper-specified.",
        ],
    })
    with open(mdir(model_key, "config", "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    start_epoch, best_val_loss, best_epoch, history = 1, float("inf"), -1, []

    if resume:
        loaded = load_latest_checkpoint(model_key, model, optimizer, scheduler, amp_scaler)
        if loaded is None:
            print(f"[{display}] RESUME=True but no checkpoint found - starting from epoch 1.")
        else:
            start_epoch, best_val_loss, best_epoch, history = loaded
            print(f"[{display}] Resumed from checkpoint. "
                  f"Continuing at epoch {start_epoch} "
                  f"(best val loss {best_val_loss:.6f} @ epoch {best_epoch}).")
    else:
        latest_path = mdir(model_key, "checkpoints", "latest", "latest.pt")
        if latest_path.exists():
            print(f"[{display}] RESUME=False - existing checkpoint will be OVERWRITTEN "
                  f"as training progresses: {latest_path}")

    if start_epoch > EPOCHS:
        print(f"[{display}] Already trained to epoch {start_epoch - 1} >= EPOCHS={EPOCHS}. "
              "Nothing to do.")
        return model, history

    epochs_no_improve = 0
    if history:
        # reconstruct patience counter from loaded history
        after_best = [h for h in history if h["epoch"] > best_epoch]
        epochs_no_improve = len(after_best)

    log_path = mdir(model_key, "logs", "train_log.txt")

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        model.train()
        batch_losses, tr_true, tr_pred = [], [], []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                out = model(xb)
                loss = CRITERION(out, yb)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"[{display}] Non-finite loss at epoch {epoch} "
                    f"(loss={loss.item()}). Training halted.\n"
                    "Diagnostics: check input scaling, learning rate, and whether the "
                    "target transform is appropriate. Values are not silently replaced."
                )

            if amp_scaler is not None:
                amp_scaler.scale(loss).backward()
                if GRAD_CLIP_NORM is not None:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                if GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            batch_losses.append(float(loss.detach().float().cpu()))
            if not TRAIN_METRICS_EVAL_PASS:
                tr_true.append(yb.detach().float().cpu().numpy())
                tr_pred.append(out.detach().float().cpu().numpy())

        train_loss = float(np.mean(batch_losses))

        if TRAIN_METRICS_EVAL_PASS:
            _, tr_true_a, tr_pred_a = evaluate_loader(model, train_loader)
        else:
            tr_true_a, tr_pred_a = np.concatenate(tr_true), np.concatenate(tr_pred)

        train_metrics = compute_metrics_all_targets(tr_true_a, tr_pred_a)

        val_loss, va_true, va_pred = evaluate_loader(model, val_loader)
        val_metrics = compute_metrics_all_targets(va_true, va_pred)

        scheduler.step(val_loss)                     # monitors VALIDATION loss only
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.time() - t0

        row = {
            "epoch": int(epoch),
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time": epoch_time,
        }
        row.update(flatten_metrics("train", train_metrics))
        row.update(flatten_metrics("val", val_metrics))

        # duplicate-epoch protection when resuming
        history = [h for h in history if int(h["epoch"]) != int(epoch)]
        history.append(row)
        history.sort(key=lambda h: int(h["epoch"]))

        pd.DataFrame(history).to_csv(mdir(model_key, "metrics", "history.csv"), index=False)

        print_epoch_report(display, epoch, EPOCHS, train_loss, val_loss,
                           current_lr, train_metrics, val_metrics, epoch_time)

        with open(log_path, "a") as lf:
            lf.write(f"epoch={epoch} train_loss={train_loss:.6f} "
                     f"val_loss={val_loss:.6f} lr={current_lr:.8f} "
                     f"time={epoch_time:.1f}s\n")

        improved = val_loss < best_val_loss - 1e-12
        if improved:
            best_val_loss, best_epoch, epochs_no_improve = val_loss, epoch, 0
            save_best_checkpoint(model_key, epoch, model, val_loss, history, config)
            print(f"  -> new best validation loss ({best_val_loss:.6f}); best.pt updated")
        else:
            epochs_no_improve += 1
            print(f"  -> no improvement for {epochs_no_improve}/{PATIENCE} epoch(s) "
                  f"(best {best_val_loss:.6f} @ epoch {best_epoch})")

        save_latest_checkpoint(model_key, epoch, model, optimizer, scheduler,
                               amp_scaler, best_val_loss, best_epoch, history, config)

        if epochs_no_improve >= PATIENCE:
            print()
            print("Early stopping triggered.")
            print(f"Best epoch: {best_epoch}")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break

    print(f"\n[{display}] Training finished. "
          f"Best epoch {best_epoch}, best val loss {best_val_loss:.6f}")
    return model, history


print("Training engine ready.")
''')

# ============================================================
# 21. Per-model training
# ============================================================
md(r"""## 21. CNN-LSTM — training (with resume logic)

Set `RESUME = True` in the configuration cell to continue an interrupted run.
Early stopping monitors validation loss only; the test set is untouched here.""")

code(r'''TRAIN_HISTORIES = {}

if "cnn_lstm" in MODELS_TO_RUN:
    _, TRAIN_HISTORIES["cnn_lstm"] = train_model("cnn_lstm")
else:
    print("cnn_lstm not in MODELS_TO_RUN - skipped.")
''')

md(r"""## 22. TCN-LSTM — training (with resume logic)""")

code(r'''if "tcn_lstm" in MODELS_TO_RUN:
    _, TRAIN_HISTORIES["tcn_lstm"] = train_model("tcn_lstm")
else:
    print("tcn_lstm not in MODELS_TO_RUN - skipped.")
''')

md(r"""## 23. LSTM-mTrans-MLP — training (with resume logic)""")

code(r'''if "lstm_mtrans_mlp" in MODELS_TO_RUN:
    _, TRAIN_HISTORIES["lstm_mtrans_mlp"] = train_model("lstm_mtrans_mlp")
else:
    print("lstm_mtrans_mlp not in MODELS_TO_RUN - skipped.")
''')

# ============================================================
# 24. Final test evaluation
# ============================================================
md(r"""## 24. Final test evaluation — BEST checkpoint only

For each model the **best** checkpoint (lowest validation loss) is loaded and the
test set is evaluated **once**. The latest checkpoint is never substituted for
final reporting. Predictions are written in original chronological test order.""")

code(r'''TEST_RESULTS = {}     # model_key -> {target -> {metric: value}}
TEST_PREDICTIONS = {}  # model_key -> (y_true_original, y_pred_original)

for mk in MODELS_TO_RUN:
    display = MODEL_DISPLAY[mk]
    model, best_ep, best_vl = load_best_for_eval(mk)
    print(f"[{display}] loaded BEST checkpoint from epoch {best_ep} "
          f"(val loss {best_vl:.6f})")

    test_loss, y_true_s, y_pred_s = evaluate_loader(model, test_loader)
    y_true_o = to_original_units(y_true_s)
    y_pred_o = to_original_units(y_pred_s)

    per_target = compute_metrics_all_targets(y_true_s, y_pred_s)
    TEST_RESULTS[mk] = per_target
    TEST_PREDICTIONS[mk] = (y_true_o, y_pred_o)

    print()
    print("=" * 60)
    print(f"FINAL TEST RESULTS - {display}")
    print("=" * 60)
    print(f"(test loss in target space: {test_loss:.6f})")
    for t in TARGETS:
        print()
        print(t)
        for m in METRIC_NAMES:
            print(f"{m}: {per_target[t][m]:.6f}")
    print()

    with open(mdir(mk, "metrics", "test_metrics.json"), "w") as f:
        json.dump({"best_epoch": best_ep, "best_val_loss": best_vl,
                   "test_loss_target_space": test_loss,
                   "metrics": per_target}, f, indent=2)

    # ---- predictions CSV, chronological order preserved ----
    pred_df = pd.DataFrame({"index": np.arange(len(y_true_o))})
    for i, t in enumerate(TARGETS):
        pred_df[f"actual_{t}"] = y_true_o[:, i]
        pred_df[f"predicted_{t}"] = y_pred_o[:, i]
    pred_df.to_csv(mdir(mk, "predictions", "test_predictions.csv"), index=False)
    print(f"Saved: {mdir(mk, 'predictions', 'test_predictions.csv')}")

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
''')

md(r"""## 25. Test metrics CSV and model comparison""")

code(r'''rows = []
for mk in MODELS_TO_RUN:
    for t in TARGETS:
        row = {"model": MODEL_DISPLAY[mk], "target": t}
        row.update({m: TEST_RESULTS[mk][t][m] for m in METRIC_NAMES})
        rows.append(row)

test_metrics_df = pd.DataFrame(rows, columns=["model", "target"] + METRIC_NAMES)
test_metrics_path = OUTPUT_DIR_P / "comparison" / "test_metrics.csv"
test_metrics_df.to_csv(test_metrics_path, index=False)

comparison_path = OUTPUT_DIR_P / "comparison" / "model_comparison.csv"
test_metrics_df.to_csv(comparison_path, index=False)

print("Saved:", test_metrics_path)
print("Saved:", comparison_path)
print()
print("Target-wise results are kept separate (never averaged into one score).")
print()
with pd.option_context("display.max_columns", None, "display.width", 250):
    print(test_metrics_df.to_string(index=False))
''')

# ============================================================
# 26. Plots
# ============================================================
md(r"""## 26. Training curves

Loss, plus per-target R², MAE and RMSE across epochs. All figures: font size 20,
DPI 300.""")

code(r'''def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(model_key):
    hist_path = mdir(model_key, "metrics", "history.csv")
    if not hist_path.exists():
        print(f"  no history.csv for {MODEL_DISPLAY[model_key]} - skipped")
        return
    h = pd.read_csv(hist_path)
    display = MODEL_DISPLAY[model_key]
    outdir = mdir(model_key, "plots")

    # ---- loss ----
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(h["epoch"], h["train_loss"], linewidth=2.5, label="Train")
    ax.plot(h["epoch"], h["val_loss"], linewidth=2.5, label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (SmoothL1)")
    ax.set_title(f"{display} - Loss")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, outdir / "loss_curve.png")

    # ---- per-target metric curves ----
    for metric in ("R2", "MAE", "RMSE"):
        for t in TARGETS:
            tr_col, va_col = f"train_{t}_{metric}", f"val_{t}_{metric}"
            if tr_col not in h.columns or va_col not in h.columns:
                continue
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(h["epoch"], h[tr_col], linewidth=2.5, label="Train")
            ax.plot(h["epoch"], h[va_col], linewidth=2.5, label="Validation")
            ax.set_xlabel("Epoch"); ax.set_ylabel(metric)
            ax.set_title(f"{display}\n{t} - {metric}", fontsize=20)
            ax.legend(); ax.grid(alpha=0.3)
            _save(fig, outdir / f"{metric.lower()}_curve_{t}.png")

    # ---- learning rate ----
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(h["epoch"], h["learning_rate"], linewidth=2.5, color="green")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
    ax.set_title(f"{display} - Learning Rate")
    ax.set_yscale("log"); ax.grid(alpha=0.3)
    _save(fig, outdir / "learning_rate.png")

    print(f"  {display}: training curves saved to {outdir}")


print("Generating training curves...")
for mk in MODELS_TO_RUN:
    plot_training_curves(mk)
''')

md(r"""## 27. Test actual vs predicted (chronological test order)""")

code(r'''def plot_actual_vs_predicted(model_key):
    y_true_o, y_pred_o = TEST_PREDICTIONS[model_key]
    display = MODEL_DISPLAY[model_key]
    outdir = mdir(model_key, "plots")
    idx = np.arange(len(y_true_o))

    for i, t in enumerate(TARGETS):
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.plot(idx, y_true_o[:, i], linewidth=2.0, label="Actual")
        ax.plot(idx, y_pred_o[:, i], linewidth=2.0, label="Predicted", alpha=0.85)
        ax.set_xlabel("Test sample (chronological order)")
        ax.set_ylabel(t)
        ax.set_title(f"{display}\n{t} - Actual vs Predicted", fontsize=20)
        ax.legend(); ax.grid(alpha=0.3)
        _save(fig, outdir / f"test_actual_vs_predicted_{t}.png")

    print(f"  {display}: actual-vs-predicted plots saved")


print("Generating actual vs predicted plots...")
for mk in MODELS_TO_RUN:
    plot_actual_vs_predicted(mk)
''')

md(r"""## 28. Residual analysis

Residual is defined as `actual - prediction`. Two views per target: residual over
test order, and residual distribution.

These plots are **descriptive diagnostics only**. No claim of unbiasedness,
normality, or homoscedasticity is made or implied — establishing that would
require formal statistical testing, which is not performed here.""")

code(r'''def plot_residuals(model_key):
    y_true_o, y_pred_o = TEST_PREDICTIONS[model_key]
    display = MODEL_DISPLAY[model_key]
    outdir = mdir(model_key, "plots")
    resid = y_true_o - y_pred_o
    idx = np.arange(len(resid))

    for i, t in enumerate(TARGETS):
        r = resid[:, i]

        fig, ax = plt.subplots(figsize=(16, 7))
        ax.plot(idx, r, linewidth=1.8)
        ax.axhline(0.0, linestyle="--", linewidth=2.0, color="black")
        ax.set_xlabel("Test sample (chronological order)")
        ax.set_ylabel("Residual (actual - predicted)")
        ax.set_title(f"{display}\n{t} - Residual over test order", fontsize=20)
        ax.grid(alpha=0.3)
        _save(fig, outdir / f"residual_over_time_{t}.png")

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(r, bins=40, edgecolor="black")
        ax.axvline(0.0, linestyle="--", linewidth=2.0, color="black")
        ax.set_xlabel("Residual (actual - predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{display}\n{t} - Residual distribution", fontsize=20)
        ax.grid(alpha=0.3)
        _save(fig, outdir / f"residual_distribution_{t}.png")

    print(f"  {display}: residual plots saved")


print("Generating residual analysis...")
for mk in MODELS_TO_RUN:
    plot_residuals(mk)
''')

md(r"""## 29. Final comparison plots

R², MAE and RMSE across the three models, plotted separately per target.""")

code(r'''cmp_dir = OUTPUT_DIR_P / "comparison" / "plots"

def plot_comparison(metric):
    for t in TARGETS:
        names = [MODEL_DISPLAY[mk] for mk in MODELS_TO_RUN]
        vals = [TEST_RESULTS[mk][t][metric] for mk in MODELS_TO_RUN]

        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(names, vals, edgecolor="black")
        ax.set_ylabel(metric)
        ax.set_title(f"{t}\nTest {metric} by model", fontsize=20)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=15)

        finite = [v for v in vals if np.isfinite(v)]
        if finite:
            span = (max(finite) - min(min(finite), 0)) or 1.0
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2,
                            v + 0.02 * span, f"{v:.4g}",
                            ha="center", va="bottom", fontsize=16)
        _save(fig, cmp_dir / f"comparison_{metric}_{t}.png")


print("Generating comparison plots...")
for metric in ("R2", "MAE", "RMSE"):
    plot_comparison(metric)
    print(f"  {metric} comparison saved")
print("Saved to:", cmp_dir)
''')

# ============================================================
# 30. Final summary + manifest
# ============================================================
md(r"""## 30. Final summary table

Every value below comes from actual test predictions produced by the best
checkpoint of each model. Nothing is pre-filled.""")

code(r'''print("=" * 70)
print("FINAL TEST SUMMARY - all models, both targets")
print("=" * 70)
print()
with pd.option_context("display.max_columns", None,
                       "display.width", 300,
                       "display.float_format", lambda v: f"{v:,.4f}"):
    print(test_metrics_df.to_string(index=False))

print()
print("Reminder on interpretation:")
print("  - The 8 primary metrics are in ORIGINAL target units.")
print(f"  - Target transform applied for inversion: {RESOLVED_TRANSFORM}")
print("  - MAE_scaled / RMSE_scaled divide by the TRAIN target std:")
for i, t in enumerate(TARGETS):
    print(f"      {t}: train_std = {TRAIN_TARGET_STD[i]:,.4f}")
print("  - MAPE uses a floored denominator "
      f"(epsilon={MAPE_EPSILON}); read with care if actuals approach zero.")

try:
    from IPython.display import display as _disp
    _disp(test_metrics_df)
except Exception:
    pass
''')

md(r"""## 31. Output manifest""")

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
    print()
    print(MODEL_DISPLAY[mk])
    if mk not in MODELS_TO_RUN:
        print("  (not run in this session)")
        continue
    for label, parts in EXPECTED.items():
        mark = "OK" if mdir(mk, *parts).exists() else "MISSING"
        print(f"  [{mark}] {label}")
    plots = sorted(mdir(mk, "plots").glob("*.png"))
    print(f"  [{'OK' if plots else 'MISSING'}] plots ({len(plots)} png)")
    for p in plots:
        print(f"        {p.name}")

print()
print("COMPARISON")
for label, p in [
    ("test_metrics.csv", OUTPUT_DIR_P / "comparison" / "test_metrics.csv"),
    ("model_comparison.csv", OUTPUT_DIR_P / "comparison" / "model_comparison.csv"),
    ("model_summary.json", OUTPUT_DIR_P / "comparison" / "model_summary.json"),
]:
    print(f"  [{'OK' if p.exists() else 'MISSING'}] {label}")
cmp_plots = sorted((OUTPUT_DIR_P / "comparison" / "plots").glob("*.png"))
print(f"  [{'OK' if cmp_plots else 'MISSING'}] comparison plots ({len(cmp_plots)} png)")
for p in cmp_plots:
    print(f"        {p.name}")

print()
print("All outputs written under:", OUTPUT_DIR_P)
''')

md(r"""## 32. Interpretation boundaries

Three categories must not be conflated when writing this up:

**1. Paper facts** — what the cited works actually state. Verified here: the
existence of both works, their venues, and their high-level model composition
(CNN/TCN + LSTM hybrids compared in the first; LSTM + modified Transformer + MLP
in the second). Their *reported results* belong to *their* datasets (traffic, air
quality, financial series) and are not comparable to results on this hospital
dataset.

**2. Implementation decisions** — every layer size, channel count, kernel size,
dilation schedule, head count, dropout rate, the shared `SmoothL1Loss`, `AdamW`,
`ReduceLROnPlateau` settings, gradient clipping, AMP, and the pre-norm
reconstruction of the "modified Transformer". These were chosen to make the
architectures runnable and comparable on this dataset. **None** is claimed to be
paper-specified.

**3. Experimental results** — whatever the tables and plots above actually show
after you run this notebook. Report those numbers as they come out. There is no
target R², and no expectation that any model should exceed any particular
threshold. Regression forecasting has no generic "accuracy" metric, so none is
reported.

A benchmark model losing to the proposed model is a result; a benchmark model
beating the proposed model is equally a result, and should be reported as such.
""")

nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "colab": {"provenance": [], "toc_visible": True},
    "accelerator": "GPU",
}

nb['cells'] = cells
nbf.write(nb, '../DeepBudgetVis_Benchmarks_CNNLSTM_TCNLSTM_LSTMmTransMLP.ipynb')
print("FINAL notebook written. total cells:", len(cells))
print("  code cells    :", sum(1 for c in cells if c.cell_type == "code"))
print("  markdown cells:", sum(1 for c in cells if c.cell_type == "markdown"))
