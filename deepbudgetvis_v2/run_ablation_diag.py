"""
Post-hoc diagnostic ablation (VALIDATION ONLY - test.csv not touched here).

Purpose: understand why the fully-upgraded model did not outperform the
retrained baseline on test, by isolating the single largest structural
change (the target-history stream) with everything else held at the same
final default hyperparameters and seq_len=21. This is diagnostic/interpretive
only - it does not change which checkpoint was already evaluated on test.
"""
import sys, os, json, time
sys.path.insert(0, '.')
import pipeline as P
import models as M
import train_utils as TU
import torch
import pandas as pd

P.set_seed()
device = P.get_device()
SEQ_LEN = 21
HP = {'hidden': 48, 'dropout': 0.25, 'lr': 1e-3, 'weight_decay': 1e-4, 'lag_window': 21, 'batch_size': 32}

train_df, val_df, test_df = P.load_splits()
train_ds_h = P.MultiStreamDataset(train_df, SEQ_LEN, include_target_history=True, n_context=0)
val_ds_h = P.MultiStreamDataset(val_df, SEQ_LEN, include_target_history=True, n_context=P.CONTEXT_ROWS)

print("=== Ablation: WITHOUT history stream (use_history_stream=False) ===", flush=True)
P.set_seed()
model_noh = M.UpgradedForecaster(len(P.DEMAND_COLS), len(P.REVCYCLE_COLS), len(P.EXPENSE_COLS),
                                    hidden=HP['hidden'], dropout=HP['dropout'], lag_window=HP['lag_window'],
                                    use_history_stream=False)
t0 = time.time()
best_val_loss_noh, best_state_noh, best_epoch_noh, _ = TU.run_training(
    model_noh, train_ds_h, val_ds_h, device, loss_name='mse', lr=HP['lr'], weight_decay=HP['weight_decay'],
    batch_size=HP['batch_size'], max_epochs=120, patience=15, use_history_stream=False, verbose=False)
print(f"best_val_loss={best_val_loss_noh:.5f} at epoch {best_epoch_noh} ({time.time()-t0:.1f}s)", flush=True)

model_noh.load_state_dict(best_state_noh)
model_noh.to(device)
y_true_s, y_pred_s, _ = TU.predict(model_noh, val_ds_h, device, use_history_stream=False)
target_stats = P.compute_target_scaler_stats(train_df)
y_true = P.inverse_targets(y_true_s, target_stats)
y_pred = P.inverse_targets(y_pred_s, target_stats)
metrics_noh = P.compute_metrics_per_target(y_true, y_pred, y_true_s, y_pred_s)
for t, m in metrics_noh.items():
    print(f"  {t}: MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} R2={m['R2']:.4f}", flush=True)

print("\n=== Reference: WITH history stream (already trained, from outputs/upgraded_final_defaulthp) ===", flush=True)
with open('outputs/upgraded_final_defaulthp/val_metrics.json') as f:
    metrics_with_h = json.load(f)
for t, m in metrics_with_h.items():
    print(f"  {t}: MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} R2={m['R2']:.4f}", flush=True)

out = {
    'without_history_stream': {'best_val_loss': best_val_loss_noh, 'best_epoch': best_epoch_noh, 'metrics': metrics_noh},
    'with_history_stream': {'best_val_loss': 0.12732616956863138, 'metrics': metrics_with_h},
}
with open('experiments/ablation_history_stream.json', 'w') as f:
    json.dump(out, f, indent=2)
