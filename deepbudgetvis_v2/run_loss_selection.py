import sys, time, json
sys.path.insert(0, '.')
import pipeline as P
import models as M
import train_utils as TU
import torch

P.set_seed()
device = P.get_device()
print('device', device, flush=True)

train_df, val_df, test_df = P.load_splits()
SEQ_LEN = 21  # selected in window-selection experiment (train/val only)

train_ds = P.MultiStreamDataset(train_df, SEQ_LEN, include_target_history=True, n_context=0)
val_ds = P.MultiStreamDataset(val_df, SEQ_LEN, include_target_history=True, n_context=P.CONTEXT_ROWS)
print('train windows', len(train_ds), 'val windows', len(val_ds), flush=True)

results = []
for loss_name in ['mse', 'huber']:
    P.set_seed()
    model = M.UpgradedForecaster(len(P.DEMAND_COLS), len(P.REVCYCLE_COLS), len(P.EXPENSE_COLS))
    t0 = time.time()
    best_val_loss, best_state, best_epoch, _ = TU.run_training(
        model, train_ds, val_ds, device,
        loss_name=loss_name, lr=1e-3, weight_decay=1e-4, batch_size=32,
        max_epochs=50, patience=10, use_history_stream=True, verbose=False
    )
    dt = time.time() - t0
    # Also compute per-target MAE/R2 in ORIGINAL units on validation for a
    # more interpretable comparison than the raw (loss-dependent) loss value.
    model.load_state_dict(best_state)
    model.to(device)
    y_true_s, y_pred_s, _ = TU.predict(model, val_ds, device, use_history_stream=True)
    target_stats = P.compute_target_scaler_stats(train_df)
    y_true = P.inverse_targets(y_true_s, target_stats)
    y_pred = P.inverse_targets(y_pred_s, target_stats)
    per_target = P.compute_metrics_per_target(y_true, y_pred, y_true_s, y_pred_s)

    print(f"loss={loss_name:6s} | best_val_loss={best_val_loss:.5f} best_epoch={best_epoch} time={dt:.1f}s", flush=True)
    for t, m in per_target.items():
        print(f"   {t}: MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} R2={m['R2']:.4f}", flush=True)

    results.append({'loss': loss_name, 'best_val_loss': best_val_loss, 'best_epoch': best_epoch,
                     'time_s': dt, 'val_metrics_per_target': per_target})

with open('experiments/loss_selection_results.json', 'w') as f:
    json.dump(results, f, indent=2)

best = min(results, key=lambda r: r['best_val_loss'])
print()
print('SELECTED LOSS (by validation combined loss):', best['loss'], flush=True)
