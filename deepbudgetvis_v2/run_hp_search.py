"""
Time-aware random hyperparameter search.

Design (per task spec): use the 84-month TRAIN block only, with an inner
EXPANDING validation scheme (3 folds), to select hyperparameters. The
official 18-month val.csv is reserved for early-stopping/checkpoint
selection during FINAL training (a separate script), and the 18-month
test.csv is never touched here.

Inner folds (all strictly inside train.csv, chronological, expanding train):
  Fold 1: inner-train = months  1-48 (2016-01..2019-12), inner-val = months 49-60 (2020-01..2020-12)
  Fold 2: inner-train = months  1-60 (2016-01..2020-12), inner-val = months 61-72 (2021-01..2021-12)
  Fold 3: inner-train = months  1-72 (2016-01..2021-12), inner-val = months 73-84 (2022-01..2022-12)

For each candidate hyperparameter configuration, a fresh model is trained on
each fold's inner-train and evaluated on that fold's inner-val; the 3 fold
losses are averaged. The configuration with the lowest average inner-val loss
is selected. This never uses val.csv or test.csv.
"""
import sys, time, json, random
sys.path.insert(0, '.')
import pipeline as P
import models as M
import train_utils as TU
import torch
import pandas as pd

P.set_seed()
device = P.get_device()
print('device', device, flush=True)

SEQ_LEN = 21  # selected in window-selection experiment

train_df, val_df, test_df = P.load_splits()
train_df = train_df.copy()
train_df['ym'] = train_df['DATE'].dt.to_period('M')
months = sorted(train_df['ym'].unique())
assert len(months) == 84

FOLD_DEFS = [(48, 60), (60, 72), (72, 84)]


def make_fold_dfs(start_train_m, end_val_m):
    train_months = months[:start_train_m]
    val_months = months[start_train_m:end_val_m]
    tr = train_df[train_df['ym'].isin(train_months)].drop(columns=['ym']).reset_index(drop=True)
    va = train_df[train_df['ym'].isin(val_months)].drop(columns=['ym']).reset_index(drop=True)
    return tr, va


SEARCH_SPACE = {
    'hidden': [32, 48, 64],
    'dropout': [0.20, 0.30, 0.40],
    'lr': [3e-4, 5e-4, 1e-3, 2e-3],
    'weight_decay': [1e-5, 1e-4, 5e-4],
    'lag_window': [14, 21, 28],
    'batch_size': [16, 32],
}
N_TRIALS = 10
TUNE_MAX_EPOCHS = 15
TUNE_PATIENCE = 5

rng = random.Random(P.SEED)


def sample_config():
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


results = []
fold_data = []
for (s, e) in FOLD_DEFS:
    tr, va = make_fold_dfs(s, e)
    fold_data.append((tr, va))
    print(f"fold train_months={s} val_months={e - s} | rows tr={len(tr)} va={len(va)}", flush=True)

for trial in range(1, N_TRIALS + 1):
    cfg = sample_config()
    print(f"\nTrial {trial:02d}/{N_TRIALS} | {cfg}", flush=True)
    fold_losses = []
    t0 = time.time()
    for fold_idx, (tr_df, va_df) in enumerate(fold_data):
        P.set_seed()
        train_ds = P.MultiStreamDataset(tr_df, SEQ_LEN, include_target_history=True, n_context=0)
        val_ds = P.MultiStreamDataset(va_df, SEQ_LEN, include_target_history=True, n_context=0)
        # NOTE: inner-val folds have NO copied context rows (they are
        # contiguous slices of train.csv), so the first `SEQ_LEN` days of
        # each inner-val fold cannot be used as prediction targets (their
        # window would reach back into inner-train, which the dataset
        # correctly disallows via n_context=0 + seq_len skip is not
        # actually needed here - the slice itself starts exactly after the
        # inner-train boundary, so all its rows ARE genuinely inner-val;
        # windows for the earliest targets just reach back across the fold
        # boundary into inner-train data, which is legitimate since that
        # data was already used for inner-train fitting, not future info).
        model = M.UpgradedForecaster(len(P.DEMAND_COLS), len(P.REVCYCLE_COLS), len(P.EXPENSE_COLS),
                                       hidden=cfg['hidden'], dropout=cfg['dropout'], lag_window=cfg['lag_window'])
        best_val_loss, _, best_epoch, _ = TU.run_training(
            model, train_ds, val_ds, device,
            loss_name='mse', lr=cfg['lr'], weight_decay=cfg['weight_decay'], batch_size=cfg['batch_size'],
            max_epochs=TUNE_MAX_EPOCHS, patience=TUNE_PATIENCE, use_history_stream=True, verbose=False
        )
        fold_losses.append(best_val_loss)
        print(f"   fold{fold_idx+1}: best_val_loss={best_val_loss:.5f} (epoch {best_epoch})", flush=True)
    avg_loss = sum(fold_losses) / len(fold_losses)
    dt = time.time() - t0
    print(f"   -> avg_inner_val_loss={avg_loss:.5f} | time={dt:.1f}s", flush=True)
    results.append({'trial': trial, **cfg, 'fold_losses': fold_losses, 'avg_inner_val_loss': avg_loss, 'time_s': dt})

results_sorted = sorted(results, key=lambda r: r['avg_inner_val_loss'])
with open('experiments/hp_search_results.json', 'w') as f:
    json.dump(results_sorted, f, indent=2)

best = results_sorted[0]
print("\n" + "=" * 70)
print("BEST CONFIG (by mean inner-val loss across 3 expanding folds, train.csv only):")
for k in SEARCH_SPACE:
    print(f"  {k}: {best[k]}")
print(f"  avg_inner_val_loss: {best['avg_inner_val_loss']:.5f}")

with open('experiments/best_hyperparameters.json', 'w') as f:
    json.dump({k: best[k] for k in SEARCH_SPACE}, f, indent=2)
