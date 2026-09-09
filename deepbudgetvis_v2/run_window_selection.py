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

candidates = [7, 14, 21, 30, 45, 60]
results = []

for seq_len in candidates:
    P.set_seed()
    train_ds = P.MultiStreamDataset(train_df, seq_len, include_target_history=True, n_context=0)
    val_ds = P.MultiStreamDataset(val_df, seq_len, include_target_history=True, n_context=P.CONTEXT_ROWS)

    model = M.UpgradedForecaster(len(P.DEMAND_COLS), len(P.REVCYCLE_COLS), len(P.EXPENSE_COLS))
    t0 = time.time()
    best_val_loss, best_state, best_epoch, _ = TU.run_training(
        model, train_ds, val_ds, device,
        loss_name='mse', lr=1e-3, weight_decay=1e-4, batch_size=32,
        max_epochs=40, patience=8, use_history_stream=True, verbose=False
    )
    dt = time.time() - t0
    print(f"seq_len={seq_len:3d} | train_windows={len(train_ds):5d} val_windows={len(val_ds):4d} | "
          f"best_val_loss={best_val_loss:.5f} best_epoch={best_epoch} time={dt:.1f}s", flush=True)
    results.append({'seq_len': seq_len, 'best_val_loss': best_val_loss, 'best_epoch': best_epoch,
                     'train_windows': len(train_ds), 'val_windows': len(val_ds), 'time_s': dt})

with open('experiments/window_selection_results.json', 'w') as f:
    json.dump(results, f, indent=2)

best = min(results, key=lambda r: r['best_val_loss'])
print()
print('BEST WINDOW:', best, flush=True)
