# Benchmark models built + executed on the repo dataset

**Notebook:** `../DeepBudgetVis_Benchmarks_RepoData.ipynb` — already **executed**,
so every output in it is real (26/26 code cells, 0 errors).

Models: **CNN-LSTM**, **TCN-LSTM**, **LSTM-mTrans-MLP** (literature-based
benchmark models — no universal SOTA ranking is claimed).

**Nothing here has been committed or pushed to GitHub.**

## Target-name discrepancy

The task asks for `TOT_OVERALL_EXP`. **That column does not exist** in this
dataset — verified: no column contains the substring `OVERALL`. The second
financial target here is **`TOTAL_OPERATING_EXP`**, and that is what the notebook
uses and reports, via an explicit printed alias mapping in Section 3. If
`TOT_OVERALL_EXP` is a genuinely different quantity from a newer preprocessing
run, this notebook is **not** using it.

## Data handling (verified in-notebook, not assumed)

- The three splits form a **perfectly contiguous gap-free daily series**: 3653
  rows, 2016-01-01 … 2025-12-31, zero duplicate dates, all day-steps == 1.
- Features: the **39 supplied `_scaled` columns**. Verified reproducible from
  train-only mean/std to 1.8e-15 → **no preprocessing leakage**; used as-is.
- Targets: supplied `log1p` + train-only z-score. Verified to 2.3e-14. Metrics
  are computed after inverting to **original dollars**.
- `IS_ANOMALY` / `ANOMALY_TYPE` excluded from features (labels, not inputs).
- Windows are built on the concatenated series so **all 547 validation and all
  549 test targets survive** at `SEQUENCE_LENGTH = 56`. A window is `[t-56, t-1]`
  — the target row is never inside its own window, so same-timestamp derived
  features (e.g. `OPERATING_MARGIN_PCT`, an exact function of both targets) cannot
  leak the answer. Split membership is assigned by **target date**.

## Actual test results (2024-07-01 … 2025-12-31, 549 days)

Best checkpoint per model, evaluated once. Original dollar units.

### NET_PATIENT_REVENUE

| Model | MAE | RMSE | MAPE | sMAPE | R² | ExplVar | MedianAE | MaxError | MAE_scaled | RMSE_scaled |
|---|---|---|---|---|---|---|---|---|---|---|
| CNN-LSTM | 8,008.69 | 10,444.68 | 4.339 | 4.379 | **0.6357** | 0.6717 | 6,424.26 | 40,182.71 | 0.3917 | 0.5109 |
| TCN-LSTM | 8,026.69 | 10,490.92 | 4.330 | 4.371 | 0.6324 | 0.6721 | 6,630.28 | 42,930.34 | 0.3926 | 0.5132 |
| LSTM-mTrans-MLP | 9,717.80 | 12,231.68 | 5.204 | 5.315 | 0.5003 | 0.6211 | 8,184.30 | 40,002.35 | 0.4754 | 0.5983 |

### TOTAL_OPERATING_EXP

| Model | MAE | RMSE | MAPE | sMAPE | R² | ExplVar | MedianAE | MaxError | MAE_scaled | RMSE_scaled |
|---|---|---|---|---|---|---|---|---|---|---|
| CNN-LSTM | 2,278.79 | 4,305.72 | 1.952 | 1.981 | **0.5045** | 0.5291 | 1,621.36 | 47,451.47 | 0.2997 | 0.5662 |
| TCN-LSTM | 2,374.38 | 4,359.33 | 2.026 | 2.061 | 0.4921 | 0.5343 | 1,668.14 | 47,579.04 | 0.3122 | 0.5733 |
| LSTM-mTrans-MLP | 2,336.32 | 4,352.05 | 2.014 | 2.036 | 0.4938 | 0.5010 | 1,678.55 | 46,909.49 | 0.3072 | 0.5723 |

Training:

| Model | params | best epoch | best val loss | epochs run |
|---|---|---|---|---|
| CNN-LSTM | 246,402 | 16 | 0.053460 | 31 |
| TCN-LSTM | 321,858 | 29 | 0.052840 | 44 |
| LSTM-mTrans-MLP | 494,914 | 5 | 0.062664 | 20 |

All three early-stopped on validation loss (`PATIENCE = 15`); none hit the
100-epoch cap.

## Validation → test degradation (disclose this in the paper)

| Model | Target | val R² | test R² | Δ |
|---|---|---|---|---|
| CNN-LSTM | NET_PATIENT_REVENUE | +0.7450 | +0.6357 | −0.1093 |
| CNN-LSTM | TOTAL_OPERATING_EXP | +0.7446 | +0.5045 | −0.2401 |
| TCN-LSTM | NET_PATIENT_REVENUE | +0.7569 | +0.6324 | −0.1245 |
| TCN-LSTM | TOTAL_OPERATING_EXP | +0.7472 | +0.4921 | −0.2551 |
| LSTM-mTrans-MLP | NET_PATIENT_REVENUE | +0.7054 | +0.5003 | −0.2051 |
| LSTM-mTrans-MLP | TOTAL_OPERATING_EXP | +0.7106 | +0.4938 | −0.2168 |

**All three models** lose R² from validation to test by a similar margin. Both
targets shift upward across splits (revenue mean ≈ \$143k train → \$173k val →
\$183k test; expenses ≈ \$101k → \$110k → \$114k). The transform is train-fitted
(correct for leakage avoidance), so every model extrapolates into a higher regime
on test. This is a property of the split, affects all models equally, and should
be stated rather than read as a weakness of any one architecture.

## Paper fidelity

I verified both cited works exist and their **high-level composition only**.
Implementation-level detail was **not retrievable** — MDPI returned `HTTP 403`
and the other source failed TLS verification.

So **all** layer sizes, channel/kernel/dilation choices, head counts, dropout,
the shared `SmoothL1Loss`, `AdamW` and `ReduceLROnPlateau` settings are labelled
in-notebook as **implementation assumptions**, never as paper values. The
"modified Transformer" is a pre-norm encoder **reconstruction**, explicitly
documented as such.

Verified sources:
[Kabir et al., Sci 7(1), 7 (2025)](https://www.mdpi.com/2413-4155/7/1/7) ·
[hybrid multivariate forecasting comparison](https://d-nb.info/1353813266/34).
The R² ≈ 0.976 / 0.94 figures in the latter are from **traffic / air-quality**
data — do not cite them as expectations for hospital financials.
*Source content paraphrased for licensing compliance.*

Architectures kept distinguishable: no BiLSTM/Transformer in CNN-LSTM or
TCN-LSTM; no CNN/TCN front end in LSTM-mTrans-MLP; the TCN uses genuine dilated
**causal** convolutions (`Chomp1d` + residuals, receptive field 61 ≥ L = 56).

## Files

```
baseline_outputs_v1/
├── cnn_lstm/ tcn_lstm/ lstm_mtrans_mlp/
│   ├── checkpoints/{latest,best}/     latest.pt, best.pt, best_model_info.json
│   ├── metrics/                       history.csv (45 cols), test_metrics.json
│   ├── predictions/                   test_predictions.csv (with DATE)
│   ├── plots/                         14 png each @ 300 dpi
│   ├── logs/                          train_log.txt
│   └── config/                        config.json, model_summary.json
└── comparison/
    ├── test_metrics.csv  model_comparison.csv  model_summary.json
    └── plots/                         6 png (R2/MAE/RMSE x 2 targets)
```

`history.csv` = 40 metric columns (10 metrics × 2 targets × train/val) +
`epoch`, `learning_rate`, `train_loss`, `val_loss`, `epoch_time`.

## Re-running

`build_nb.py` regenerates the notebook (edit this, not the `.ipynb`). To re-run
training, either open the notebook and Run All, or delete
`baseline_outputs_v1/*/checkpoints/` first — with `RESUME = True` the notebook
will otherwise correctly resume and report there is nothing left to do.
