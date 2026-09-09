"""
Retrain the ORIGINAL v1 architecture (BaselineForecaster, faithful port of
Prop_model_v1 (1).ipynb's DeepBudgetVisForecaster) on the NEW 84/18/18
chronological split, using v1's ORIGINAL notebook hyperparameters
(hidden=48, dropout=0.25, lr=1e-3, weight_decay=1e-4, lag_window=21,
batch_size=16, seq_len=30 - all taken directly from Cells 2 and 6 of
Prop_model_v1 (1).ipynb) so the comparison against the upgraded model is
fair: same split, same targets, same metrics, same test period, only the
architecture differs.

This is necessary because the original notebook's printed results (R2=0.68
combined, see Cell 13/14 outputs) were produced under a DIFFERENT (110/5/5
month, non-chronologically-verified) split that is not directly comparable
to the 84/18/18 split used for the upgraded model in this project.
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
print('device', device, flush=True)

SEQ_LEN = 30  # v1's original notebook window - unchanged for this baseline retrain
V1_HP = {'hidden': 48, 'dropout': 0.25, 'lr': 1e-3, 'weight_decay': 1e-4, 'lag_window': 21, 'batch_size': 16}
print('v1 original hyperparameters:', V1_HP, flush=True)

OUTPUT_DIR = 'outputs/baseline_v1_retrained'
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df, val_df, test_df = P.load_splits()

train_ds = P.MultiStreamDataset(train_df, SEQ_LEN, include_target_history=False, n_context=0)
val_ds = P.MultiStreamDataset(val_df, SEQ_LEN, include_target_history=False, n_context=P.CONTEXT_ROWS)
print(f"train windows={len(train_ds)} val windows={len(val_ds)}", flush=True)

model = M.BaselineForecaster(len(P.DEMAND_COLS), len(P.REVCYCLE_COLS), len(P.EXPENSE_COLS), len(P.TARGET_COLS),
                               hidden=V1_HP['hidden'], dropout=V1_HP['dropout'], lag_window=V1_HP['lag_window'])
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,} (should match original notebook's printed 99,235)", flush=True)

t0 = time.time()
best_val_loss, best_state, best_epoch, history = TU.run_training(
    model, train_ds, val_ds, device,
    loss_name='mse', lr=V1_HP['lr'], weight_decay=V1_HP['weight_decay'],
    batch_size=V1_HP['batch_size'], max_epochs=120, patience=15,
    use_history_stream=False, verbose=True, return_history=True
)
dt = time.time() - t0
print(f"\nTraining complete in {dt:.1f}s. Best val_loss={best_val_loss:.5f} at epoch {best_epoch}", flush=True)

torch.save(best_state, os.path.join(OUTPUT_DIR, 'best_model.pt'))
pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, 'history.csv'), index=False)

config = {
    'seq_len': SEQ_LEN, 'loss': 'mse', **V1_HP,
    'n_params': n_params, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
    'train_windows': len(train_ds), 'val_windows': len(val_ds),
    'seed': P.SEED, 'training_time_s': dt,
}
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

model.load_state_dict(best_state)
model.to(device)
y_true_s, y_pred_s, anom_flags = TU.predict(model, val_ds, device, use_history_stream=False)
target_stats = P.compute_target_scaler_stats(train_df)
with open(os.path.join(OUTPUT_DIR, 'target_scaler_stats.json'), 'w') as f:
    json.dump(target_stats, f, indent=2)

y_true = P.inverse_targets(y_true_s, target_stats)
y_pred = P.inverse_targets(y_pred_s, target_stats)
val_metrics = P.compute_metrics_per_target(y_true, y_pred, y_true_s, y_pred_s)
print("\nVALIDATION metrics (per target):", flush=True)
for t, m in val_metrics.items():
    print(f"  {t}: {m}", flush=True)

with open(os.path.join(OUTPUT_DIR, 'val_metrics.json'), 'w') as f:
    json.dump(val_metrics, f, indent=2)

print(f"\nAll outputs saved to {OUTPUT_DIR}/", flush=True)
