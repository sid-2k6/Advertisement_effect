"""
FINAL, ONE-SHOT test evaluation. Run only after ALL architecture, feature,
hyperparameter, window, loss, and checkpoint decisions have been frozen
(they were: seq_len=21, loss=mse, hyperparameters=defaults for the upgraded
model; seq_len=30, v1 original hyperparameters for the baseline retrain).

This script is executed EXACTLY ONCE. Its output is not used to make any
further decisions.
"""
import sys, os, json
sys.path.insert(0, '.')
import pipeline as P
import models as M
import train_utils as TU
import torch
import pandas as pd
import numpy as np

device = P.get_device()
train_df, val_df, test_df = P.load_splits()
target_stats = P.compute_target_scaler_stats(train_df)

results = {}

# ---------------- Upgraded model (final, default hyperparameters) ----------------
UPG_DIR = 'outputs/upgraded_final'
with open(os.path.join(UPG_DIR, 'config.json')) as f:
    upg_cfg = json.load(f)
print('Upgraded model config:', upg_cfg, flush=True)

upg_model = M.UpgradedForecaster(len(P.DEMAND_COLS), len(P.REVCYCLE_COLS), len(P.EXPENSE_COLS),
                                    hidden=upg_cfg['hidden'], dropout=upg_cfg['dropout'],
                                    lag_window=upg_cfg['lag_window'])
upg_model.load_state_dict(torch.load(os.path.join(UPG_DIR, 'best_model.pt'), map_location=device))
upg_model.to(device)

test_ds_upg = P.MultiStreamDataset(test_df, upg_cfg['seq_len'], include_target_history=True, n_context=P.CONTEXT_ROWS)
print(f"Upgraded model test windows: {len(test_ds_upg)} (must equal official test.csv rows = {len(pd.read_csv(P.TEST_CSV))})", flush=True)
assert len(test_ds_upg) == len(pd.read_csv(P.TEST_CSV))

y_true_s, y_pred_s, anom_flags = TU.predict(upg_model, test_ds_upg, device, use_history_stream=True)
y_true = P.inverse_targets(y_true_s, target_stats)
y_pred = P.inverse_targets(y_pred_s, target_stats)
upg_test_metrics = P.compute_metrics_per_target(y_true, y_pred, y_true_s, y_pred_s)
print("\n=== UPGRADED MODEL - FINAL TEST METRICS ===", flush=True)
for t, m in upg_test_metrics.items():
    print(f"  {t}: {m}", flush=True)

np.savez(os.path.join(UPG_DIR, 'test_predictions.npz'),
         y_true=y_true, y_pred=y_pred, y_true_scaled=y_true_s, y_pred_scaled=y_pred_s,
         is_anomaly=anom_flags, dates=test_df['DATE'].values[-len(test_ds_upg):])
with open(os.path.join(UPG_DIR, 'test_metrics.json'), 'w') as f:
    json.dump(upg_test_metrics, f, indent=2)

results['upgraded'] = upg_test_metrics

# ---------------- Baseline v1 (retrained on new split) ----------------
BASE_DIR = 'outputs/baseline_v1_retrained'
with open(os.path.join(BASE_DIR, 'config.json')) as f:
    base_cfg = json.load(f)
print('\nBaseline v1 (retrained) config:', base_cfg, flush=True)

base_model = M.BaselineForecaster(len(P.DEMAND_COLS), len(P.REVCYCLE_COLS), len(P.EXPENSE_COLS), len(P.TARGET_COLS),
                                     hidden=base_cfg['hidden'], dropout=base_cfg['dropout'],
                                     lag_window=base_cfg['lag_window'])
base_model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'best_model.pt'), map_location=device))
base_model.to(device)

test_ds_base = P.MultiStreamDataset(test_df, base_cfg['seq_len'], include_target_history=False, n_context=P.CONTEXT_ROWS)
print(f"Baseline model test windows: {len(test_ds_base)} (must equal official test.csv rows = {len(pd.read_csv(P.TEST_CSV))})", flush=True)
assert len(test_ds_base) == len(pd.read_csv(P.TEST_CSV))

y_true_s_b, y_pred_s_b, anom_flags_b = TU.predict(base_model, test_ds_base, device, use_history_stream=False)
y_true_b = P.inverse_targets(y_true_s_b, target_stats)
y_pred_b = P.inverse_targets(y_pred_s_b, target_stats)
base_test_metrics = P.compute_metrics_per_target(y_true_b, y_pred_b, y_true_s_b, y_pred_s_b)
print("\n=== BASELINE V1 (RETRAINED) - FINAL TEST METRICS ===", flush=True)
for t, m in base_test_metrics.items():
    print(f"  {t}: {m}", flush=True)

np.savez(os.path.join(BASE_DIR, 'test_predictions.npz'),
         y_true=y_true_b, y_pred=y_pred_b, y_true_scaled=y_true_s_b, y_pred_scaled=y_pred_s_b,
         is_anomaly=anom_flags_b, dates=test_df['DATE'].values[-len(test_ds_base):])
with open(os.path.join(BASE_DIR, 'test_metrics.json'), 'w') as f:
    json.dump(base_test_metrics, f, indent=2)

results['baseline_v1_retrained'] = base_test_metrics

with open('experiments/final_test_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n\n=== SIDE-BY-SIDE COMPARISON (TEST SET, same 84/18/18 split) ===", flush=True)
for target in P.TARGET_COLS:
    print(f"\n{target}:")
    print(f"  {'metric':15s} {'baseline_v1':>15s} {'upgraded_v2':>15s} {'delta':>12s}")
    for metric in ['MAE', 'RMSE', 'MAPE', 'SMAPE', 'R2', 'ExplainedVar', 'MedianAE', 'MaxError']:
        bv = base_test_metrics[target][metric]
        uv = upg_test_metrics[target][metric]
        print(f"  {metric:15s} {bv:15.4f} {uv:15.4f} {uv-bv:12.4f}")
