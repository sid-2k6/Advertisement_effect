"""
Model definitions for DeepBudget-Vis v2.

Contains:
  - BaselineForecaster: exact reimplementation of Prop_model_v1 (1).ipynb's
    DeepBudgetVisForecaster (Cells 5-6), used ONLY to retrain the ORIGINAL
    v1 architecture on the new 84/18/18 split for a fair baseline comparison
    (Task: "Retrain original v1 architecture on the SAME split"). No
    architecture changes here - this is a faithful port, not a redesign.

  - UpgradedForecaster: the v2 model, containing only the upgrades that were
    justified by the data/EDA audit (see final report "WHAT CHANGED AND WHY"):
      * Upgrade 1: dedicated financial-history encoder over past target
        values (strictly historical, never IS_ANOMALY, never future target).
      * Upgrade 2: learned attention pooling over the full valid sequence
        for every stream, replacing "take only the last hidden state".
      * Upgrade 3: target-specific forecast heads (Revenue head / Expense
        head) instead of one shared linear output layer.
      * Upgrade 6: anomaly probability explicitly gates the fused
        representation before the forecast heads (anomaly-CONDITIONED gate,
        not just a jointly-trained gate that happens to share input).
    Everything else (DemandEncoder CNN+GRU, RevCycleEncoder lag-attention,
    ExpenseEncoder GRU, cross-stream fusion attention) is retained unchanged
    from v1 because no data/EDA evidence was found to justify replacing it.
"""
import torch
import torch.nn as nn


# ============================================================
# Shared encoder building blocks (used by both Baseline and Upgraded models)
# ============================================================
class DemandEncoder(nn.Module):
    """1D-CNN (kernel=7, weekly local pattern) + 2-layer GRU. Unchanged from v1."""
    def __init__(self, n_features, hidden=48, dropout=0.25):
        super().__init__()
        self.conv = nn.Conv1d(n_features, 32, kernel_size=7, padding=3)
        self.act = nn.ReLU()
        self.gru = nn.GRU(32, hidden, batch_first=True, dropout=dropout, num_layers=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.act(self.conv(x))
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        return self.drop(out)  # (batch, seq, hidden)


class RevCycleEncoder(nn.Module):
    """GRU + lag-attention over the last `lag_window` steps. Unchanged from v1.
    The lag window is sized around the empirically verified ~14-day
    TOTAL_GROSS_CHARGES -> CASH_COLLECTED correlation peak (measured directly
    from train.csv, see final report "Upgrade 7 / lag verification")."""
    def __init__(self, n_features, hidden=48, lag_window=21, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True, dropout=dropout, num_layers=2)
        self.lag_window = lag_window
        self.lag_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        w = min(self.lag_window, out.size(1))
        recent = out[:, -w:, :]
        attn_out, _ = self.lag_attn(recent, recent, recent)
        fused = out.clone()
        fused[:, -w:, :] = fused[:, -w:, :] + attn_out
        return self.drop(fused)  # (batch, seq, hidden)


class ExpenseEncoder(nn.Module):
    """Plain 2-layer GRU. Unchanged from v1."""
    def __init__(self, n_features, hidden=48, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True, dropout=dropout, num_layers=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.drop(out)  # (batch, seq, hidden)


class HistoryEncoder(nn.Module):
    """Upgrade 1: dedicated small GRU encoder over past target history
    (NET_PATIENT_REVENUE_scaled, TOTAL_OPERATING_EXP_scaled), strictly
    t-seq_len..t-1. This is a SEPARATE stream from Demand/RevCycle/Expense,
    fused later via the same cross-stream attention mechanism - it does not
    duplicate targets into the other streams."""
    def __init__(self, n_targets=2, hidden=48, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(n_targets, hidden, batch_first=True, dropout=dropout, num_layers=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.drop(out)  # (batch, seq, hidden)


class AttnPool(nn.Module):
    """Upgrade 2: learned additive attention pooling over the time axis of a
    single stream's encoder output, replacing `out[:, -1, :]`."""
    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, seq):  # seq: (batch, seq_len, hidden)
        scores = self.score(seq).squeeze(-1)          # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)          # (batch, seq_len)
        pooled = torch.bmm(weights.unsqueeze(1), seq).squeeze(1)  # (batch, hidden)
        return pooled, weights


# ============================================================
# BASELINE (v1, unmodified) - used only for the fair-comparison retrain
# ============================================================
class BaselineForecaster(nn.Module):
    """Faithful reimplementation of Prop_model_v1 (1).ipynb's
    DeepBudgetVisForecaster. Used to retrain v1 on the 84/18/18 split."""
    def __init__(self, n_demand, n_rev, n_exp, n_targets, hidden=48, dropout=0.25, lag_window=21):
        super().__init__()
        self.demand_enc = DemandEncoder(n_demand, hidden, dropout)
        self.rev_enc = RevCycleEncoder(n_rev, hidden, lag_window, dropout)
        self.exp_enc = ExpenseEncoder(n_exp, hidden, dropout)
        self.fusion_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.anomaly_head = nn.Linear(hidden, 1)
        self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_targets)
        )

    def forward(self, xd, xr, xe, xh=None):
        hd = self.demand_enc(xd)
        hr = self.rev_enc(xr)
        he = self.exp_enc(xe)
        stacked = torch.stack([hd[:, -1, :], hr[:, -1, :], he[:, -1, :]], dim=1)
        fused, _ = self.fusion_attn(stacked, stacked, stacked)
        fused = self.fusion_norm(fused + stacked)
        fused_pooled = fused.mean(dim=1)
        anomaly_logit = self.anomaly_head(fused_pooled).squeeze(-1)
        gate_weights = self.gate(fused_pooled)
        gated = fused_pooled * gate_weights
        forecast = self.forecast_head(gated)
        return forecast, anomaly_logit


# ============================================================
# UPGRADED (v2) model
# ============================================================
class UpgradedForecaster(nn.Module):
    def __init__(self, n_demand, n_rev, n_exp, n_targets=2, hidden=48, dropout=0.25,
                 lag_window=21, use_history_stream=True):
        super().__init__()
        self.use_history_stream = use_history_stream
        self.demand_enc = DemandEncoder(n_demand, hidden, dropout)
        self.rev_enc = RevCycleEncoder(n_rev, hidden, lag_window, dropout)
        self.exp_enc = ExpenseEncoder(n_exp, hidden, dropout)
        if use_history_stream:
            self.hist_enc = HistoryEncoder(n_targets, hidden, dropout)

        # Upgrade 2: per-stream learned attention pooling (replaces out[:, -1, :])
        self.pool_demand = AttnPool(hidden)
        self.pool_rev = AttnPool(hidden)
        self.pool_exp = AttnPool(hidden)
        if use_history_stream:
            self.pool_hist = AttnPool(hidden)

        n_streams = 4 if use_history_stream else 3
        self.n_streams = n_streams

        # Cross-stream fusion attention (unchanged mechanism from v1, but now
        # operating on attention-pooled tokens instead of last-hidden-state
        # tokens, and over 4 tokens instead of 3 when history stream is used)
        self.fusion_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden)

        # Upgrade 6: anomaly probability explicitly conditions a gate applied
        # to the fused representation BEFORE the forecast heads. IS_ANOMALY
        # itself is never fed as an input anywhere in this model - only used
        # as the auxiliary supervision target for anomaly_head during
        # training (see training loop).
        self.anomaly_head = nn.Linear(hidden, 1)
        self.anomaly_gate_proj = nn.Linear(1, hidden)  # broadcast anomaly prob into gate space

        self.forecast_dropout = nn.Dropout(dropout)

        # Upgrade 3: target-specific forecast heads (Revenue / Expense),
        # each a small 2-layer MLP off the SAME shared fused representation.
        self.revenue_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        self.expense_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, xd, xr, xe, xh=None):
        hd = self.demand_enc(xd)
        hr = self.rev_enc(xr)
        he = self.exp_enc(xe)

        pd_, _ = self.pool_demand(hd)
        pr_, _ = self.pool_rev(hr)
        pe_, _ = self.pool_exp(he)

        tokens = [pd_, pr_, pe_]
        if self.use_history_stream:
            assert xh is not None, "History stream enabled but xh not provided"
            hh = self.hist_enc(xh)
            ph_, _ = self.pool_hist(hh)
            tokens.append(ph_)

        stacked = torch.stack(tokens, dim=1)  # (batch, n_streams, hidden)
        fused, _ = self.fusion_attn(stacked, stacked, stacked)
        fused = self.fusion_norm(fused + stacked)
        fused_pooled = fused.mean(dim=1)  # (batch, hidden)

        # Anomaly-conditioned gate: anomaly probability (detached from the
        # forecast path's gradient with respect to the LABEL, but the gate
        # itself is fully differentiable and trained jointly with the
        # forecast loss) modulates the fused representation.
        anomaly_logit = self.anomaly_head(fused_pooled).squeeze(-1)
        anomaly_prob = torch.sigmoid(anomaly_logit).unsqueeze(-1)  # (batch, 1)
        gate = torch.sigmoid(self.anomaly_gate_proj(1.0 - anomaly_prob))  # down-weight when anomaly_prob high
        gated = fused_pooled * gate
        gated = self.forecast_dropout(gated)

        revenue = self.revenue_head(gated)
        expense = self.expense_head(gated)
        forecast = torch.cat([revenue, expense], dim=-1)  # (batch, 2) -> [NET_PATIENT_REVENUE, TOTAL_OPERATING_EXP]

        return forecast, anomaly_logit
