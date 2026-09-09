# DeepBudget-Vis v2 — supporting files

This directory contains all the code, experiment logs, and saved outputs
behind `DeepBudgetVis_Proposed_Final_v2.ipynb` (copied to the repository
root). The notebook is self-contained and runnable top-to-bottom; the files
here are provided for reproducibility and audit purposes.

## What is in this directory

- `pipeline.py` — data loading, leakage-safe scaling verification, the
  `MultiStreamDataset` class (including the target-index contamination fix).
- `models.py` — `BaselineForecaster` (faithful port of v1's architecture)
  and `UpgradedForecaster` (v2, with Upgrades 1/2/3/6 applied).
- `train_utils.py` — shared training/prediction loop.
- `build_notebook.py` — the script that generated
  `DeepBudgetVis_Proposed_Final_v2.ipynb` programmatically (every cell's code
  and captured output trace back to an actual run in this directory).
- `run_window_selection.py` — Upgrade 8 experiment (train→val only).
- `run_loss_selection.py` — Upgrade 4 experiment (train→val only).
- `run_hp_search.py` — Upgrade 10 experiment (inner expanding folds inside
  train.csv only).
- `run_final_training_selected.py` — final training of the upgraded model
  with the frozen configuration (produces `outputs/upgraded_final/`).
- `run_baseline_v1_retrain.py` — retrains the ORIGINAL v1 architecture on the
  same 84/18/18 split for a fair comparison (produces
  `outputs/baseline_v1_retrained/`).
- `run_final_test_evaluation.py` — the one-shot final test evaluation for
  both models (Section 16 of the notebook).
- `run_ablation_diag.py` — post-hoc, val-only diagnostic ablation (history
  stream on/off) used only for interpretation (Section 17), never for model
  selection.

## experiments/

Raw logs and JSON results from every selection experiment, plus two written
audit documents:
- `capex_accounting_audit.md` — full CAPEX/derived-column integrity audit.
- `distribution_shift_note.md` — measured train/val/test target distribution
  shift, used in the research interpretation section.
- `hp_search_outcome.log` — honest record of the hyperparameter search
  result NOT transferring to the real training budget.

## outputs/

- `upgraded_final/` — final upgraded-model checkpoint, config, history,
  validation metrics, test metrics, test predictions.
- `baseline_v1_retrained/` — retrained-v1 checkpoint, config, history,
  validation metrics, test metrics, test predictions.
- `final_test_comparison.csv` — side-by-side test metrics for both models,
  both targets.

## How to reproduce

From this directory (`Advertisement_effect/deepbudgetvis_v2/`), with
`pandas`, `numpy`, `torch`, `scikit-learn`, `scipy` installed:

```bash
python3 run_window_selection.py       # ~8 min on CPU
python3 run_loss_selection.py         # ~2 min
python3 run_hp_search.py              # ~20 min
python3 run_final_training_selected.py
python3 run_baseline_v1_retrain.py
python3 run_final_test_evaluation.py  # loads test.csv, run this LAST
```

Or simply open and run `DeepBudgetVis_Proposed_Final_v2.ipynb` top-to-bottom
— it reproduces the exact same numbers embedded as outputs in that notebook
(all runs use `SEED=42`, deterministic on CPU).

## Headline result (see the notebook's "RESEARCH INTERPRETATION" section)

The upgraded model beats the retrained-v1 baseline on **validation**, but
the retrained-v1 baseline beats the upgraded model on the held-out **test**
set, for both targets. This is reported honestly, not hidden.
