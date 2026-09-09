"""
Training / evaluation loop utilities shared across all experiments
(window selection, loss selection, hyperparameter search, final training).

All functions here train on train_ds / evaluate on val_ds ONLY, unless
explicitly documented otherwise (final test evaluation is a separate,
one-shot call in the final notebook - never inside this module's tuning
helpers).
"""
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

import pipeline as P


def make_loss(name):
    if name == 'mse':
        return torch.nn.MSELoss()
    elif name == 'huber':
        return torch.nn.HuberLoss(delta=1.0)
    else:
        raise ValueError(name)


def run_training(model, train_ds, val_ds, device, loss_name='mse', lr=1e-3, weight_decay=1e-4,
                  batch_size=32, max_epochs=60, patience=10, anomaly_weight=0.1,
                  use_history_stream=True, verbose=False, warmup_epochs=3,
                  return_history=False):
    """Generic train/val loop with warmup + plateau scheduler + early stopping
    on validation combined loss (forecast loss + anomaly_weight * anomaly BCE),
    matching the training protocol already used in Prop_model_v1 (1).ipynb.

    Returns: best_val_loss, best_state_dict (CPU), history (list of dict, only if return_history)
    """
    model = model.to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion_forecast = make_loss(loss_name)
    criterion_anomaly = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def warmup_lr(epoch):
        return min(1.0, (epoch + 1) / max(1, warmup_epochs))
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float('inf')
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for xd, xr, xe, xh, y, is_anom in train_loader:
            xd, xr, xe, xh, y, is_anom = xd.to(device), xr.to(device), xe.to(device), xh.to(device), y.to(device), is_anom.to(device)
            optimizer.zero_grad()
            if use_history_stream:
                pred, anom_logit = model(xd, xr, xe, xh)
            else:
                pred, anom_logit = model(xd, xr, xe)
            loss = criterion_forecast(pred, y) + anomaly_weight * criterion_anomaly(anom_logit, is_anom)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        if epoch < warmup_epochs:
            warmup_scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xd, xr, xe, xh, y, is_anom in val_loader:
                xd, xr, xe, xh, y, is_anom = xd.to(device), xr.to(device), xe.to(device), xh.to(device), y.to(device), is_anom.to(device)
                if use_history_stream:
                    pred, anom_logit = model(xd, xr, xe, xh)
                else:
                    pred, anom_logit = model(xd, xr, xe)
                loss = criterion_forecast(pred, y) + anomaly_weight * criterion_anomaly(anom_logit, is_anom)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        if epoch >= warmup_epochs:
            plateau_scheduler.step(val_loss)

        if return_history:
            history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                             'lr': optimizer.param_groups[0]['lr']})

        if verbose:
            print(f"epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return best_val_loss, best_state, best_epoch, history


@torch.no_grad()
def predict(model, ds, device, use_history_stream=True, batch_size=32):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    all_true, all_pred, all_anom = [], [], []
    for xd, xr, xe, xh, y, is_anom in loader:
        xd, xr, xe, xh = xd.to(device), xr.to(device), xe.to(device), xh.to(device)
        if use_history_stream:
            pred, _ = model(xd, xr, xe, xh)
        else:
            pred, _ = model(xd, xr, xe)
        all_true.append(y.numpy())
        all_pred.append(pred.cpu().numpy())
        all_anom.append(is_anom.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_anom)
