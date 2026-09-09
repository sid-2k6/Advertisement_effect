"""
Trains the UPGRADED model using the FROZEN final configuration selected in
this project: seq_len=21 (Section 11 window search), loss=mse (Section 12
loss search), and the model's DEFAULT hyperparameters (hidden=48,
dropout=0.25, lr=1e-3, weight_decay=1e-4, lag_window=21, batch_size=32).

The default hyperparameters were selected INSTEAD OF the inner-fold
HP-search winner because, when both were retrained under the identical full
training budget (max_epochs=120, patience=15) on train.csv and evaluated on
val.csv, the defaults achieved a LOWER (better) validation loss
(0.12733 vs 0.13418) - see experiments/hp_search_outcome.log for the full
comparison. This is the FINAL checkpoint later evaluated on test.csv in
run_final_test_evaluation.py.
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

SEQ_LEN = 21
LOSS_NAME = 'mse'
DEFAULT_HP = {'hidden': 48, 'dropout': 0.25, 'lr': 1e-3, 'weight_decay': 1e-4, 'lag_window': 21, 'batch_size': 32}
print('Using DEFAULT hyperparameters:', DEFAULT_HP, flush=True)

OUTPUT_DIR = 'outputs/upgraded_final'
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df, val_df, test_df = P.load_splits()

train_ds = P.MultiStreamDataset(train_df, SEQ_LEN, include_target_history=True, n_context=0)
val_ds = P.MultiStreamDataset(val_df, SEQ_LEN, include_target_history=True, n_context=P.CONTEXT_ROWS)
print(f"train windows={len(train_ds)} val windows={len(val_ds)}", flush=True)

model = M.UpgradedForecaster(len(P.DEMAND_COLS), len(P.REVCYCLE_COLS), len(P.EXPENSE_COLS),
                               hidden=DEFAULT_HP['hidden'], dropout=DEFAULT_HP['dropout'],
                               lag_window=DEFAULT_HP['lag_window'])
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}", flush=True)

t0 = time.time()
best_val_loss, best_state, best_epoch, history = TU.run_training(
    model, train_ds, val_ds, device,
    loss_name=LOSS_NAME, lr=DEFAULT_HP['lr'], weight_decay=DEFAULT_HP['weight_decay'],
    batch_size=DEFAULT_HP['batch_size'], max_epochs=120, patience=15,
    use_history_stream=True, verbose=True, return_history=True
)
dt = time.time() - t0
print(f"\nTraining complete in {dt:.1f}s. Best val_loss={best_val_loss:.5f} at epoch {best_epoch}", flush=True)

torch.save(best_state, os.path.join(OUTPUT_DIR, 'best_model.pt'))
pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, 'history.csv'), index=False)

config = {
    'seq_len': SEQ_LEN, 'loss': LOSS_NAME, **DEFAULT_HP,
    'n_params': n_params, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
    'train_windows': len(train_ds), 'val_windows': len(val_ds),
    'seed': P.SEED, 'training_time_s': dt,
}
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

model.load_state_dict(best_state)
model.to(device)
y_true_s, y_pred_s, anom_flags = TU.predict(model, val_ds, device, use_history_stream=True)
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
