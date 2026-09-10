# Literature-based benchmark models for DeepBudget-Vis

Deliverable: **`../DeepBudgetVis_Benchmarks_CNNLSTM_TCNLSTM_LSTMmTransMLP.ipynb`**
(repository root) — one complete, Colab-ready notebook implementing three
literature-based benchmark forecasting models:

1. **CNN-LSTM**
2. **TCN-LSTM**
3. **LSTM-mTrans-MLP**

## Read this first — two things you must know

### 1. The notebook contains no results

Every code cell ships with **empty output**. The target dataset lives in your
Google Drive (`/content/drive/MyDrive/GPT forecast/preprocessed_data`) and is not
reachable from the environment where this notebook was authored. Embedding any
metric would have meant inventing it. Results appear only when you run it.

### 2. The dataset described in the request is NOT the dataset in this repository

Verified by direct inspection of `train.csv` in this repo:

| Item | Your request | This repository |
|---|---|---|
| Target 2 | `TOT_OVERALL_EXP` | `TOTAL_OPERATING_EXP` — `TOT_OVERALL_EXP` **does not exist** |
| Features | ~728 | 39 scaled features (85 columns total) |
| Sequence length | 56 | 21 / 30 used in prior work here |
| Layout | `preprocessed_data/` arrays | flat CSVs only |

So the notebook was written for **your Drive dataset**, not this repo's CSVs. It
therefore:

- **discovers** your files instead of assuming filenames (three supported
  layouts: single `.npz`, separate `.npy`, or flat `.csv`);
- **never hard-codes** `728` — `input_dim` is always `X_train.shape[-1]`;
- resolves `TOT_OVERALL_EXP` as the primary target name, with a configurable
  `TARGET_ALIASES` fallback list (which includes `TOTAL_OPERATING_EXP`), and
  **prints exactly which column it matched**;
- **fails loudly** if a target or split cannot be found — it never substitutes a
  different column and never creates placeholder data.

If your Drive dataset genuinely uses a different second-target name, add it to
`TARGET_ALIASES` in the configuration cell.

## Paper fidelity — what I could and could not verify

I could confirm both works exist and their **high-level composition**. I could
**not** retrieve implementation-level detail: both publisher pages returned
access errors (MDPI `HTTP 403`, the other source a TLS chain failure).

**Verified:**

- Kabir et al., *"LSTM–Transformer-Based Robust Hybrid Deep Learning Model for
  Financial Time Series Forecasting"*, **Sci** 7(1), 7 (2025) — abstract states
  the proposed **LSTM-mTrans-MLP** integrates an LSTM network, a **modified
  Transformer** network, and a multilayered perceptron.
  <https://www.mdpi.com/2413-4155/7/1/7>
- A closely-matching hybrid-forecasting work compares exactly **CNN-LSTM,
  CNN-BiLSTM, TCN-LSTM, TCN-BiLSTM**, reporting TCN-BiLSTM best on its own two
  datasets (Traffic Volume R² ≈ 0.976, Air Quality R² ≈ 0.94).
  <https://d-nb.info/1353813266/34>

**Consequence:** every numeric architecture choice (channels, kernel size,
dilations, hidden sizes, head count, MLP widths, dropout), the shared
`SmoothL1Loss`, `AdamW`, and the `ReduceLROnPlateau` settings are labelled in the
notebook as **implementation assumptions**, not paper values. The "modified
Transformer" is implemented as a **pre-norm encoder block** (LayerNorm →
multi-head attention → residual; LayerNorm → GELU FFN → residual) and explicitly
documented as a reconstruction, not the paper's exact mechanism.

Those R² figures above belong to **traffic/air-quality data** and say nothing
about what to expect on hospital financial data.

*Source content was paraphrased for licensing compliance.*

## Files here

- `build_benchmark_notebook.py` — generates the notebook (single source of truth;
  edit this, not the `.ipynb`, then re-run it)
- `smoke_test.py` — end-to-end verification against **synthetic** data

## Smoke-test status

`python3 smoke_test.py` → **all 3 scenarios pass**:

| Scenario | Result |
|---|---|
| Layout C (CSV), fresh, 2 epochs | pass — all artifacts written |
| `RESUME=True`, extend to 4 epochs | pass — epochs 1–4, continued numbering, **no duplicate rows** |
| Layout A (NPZ), fresh, 2 epochs | pass — all artifacts written |

Also asserted: `history.csv` carries **40 metric columns** (10 metrics × 2
targets × train/val) plus `train_loss`, `val_loss`, `learning_rate`, `epoch_time`;
`test_predictions.csv` has the exact required column names; every model emits
14 PNGs and the comparison folder emits 6.

Parameter counts observed across the two synthetic feature dimensionalities
(12 vs 9 features) differed accordingly — direct evidence `input_dim` is derived,
not hard-coded.

**These smoke-test numbers are meaningless as research results** — the data is
random. The test proves only that the code executes and emits every expected
artifact.

## How to run

1. Open the notebook in Colab (GPU runtime recommended).
2. Edit **only** the `USER CONFIGURATION` cell — set `BASE_DIR` / `DATA_DIR` /
   `OUTPUT_DIR`, and confirm `TARGETS`.
3. `Runtime → Run all`.

Set `RESUME = True` to continue an interrupted run. Early stopping and best-model
selection use **validation loss only**; the test set is evaluated exactly once,
from the **best** checkpoint, after training completes.
