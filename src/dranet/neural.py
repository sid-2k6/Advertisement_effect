"""
Neural CATE / uplift models (PyTorch, CPU) with the shared model interface.

Implemented (faithful, compact reimplementations; documented as such):
  * TARNet    - Shalit et al. 2017: shared representation + two outcome heads.
  * CFRNet    - Shalit et al. 2017: TARNet + representation balancing (linear MMD/IPM).
  * DragonNet - Shi et al. 2019: shared rep + two outcome heads + propensity head
                + targeted regularization.
  * DESCN-style - Zhong et al. 2022 (KDD): entire-space cross network; here a compact
                variant with propensity + two response heads trained on entire-space
                (e-weighted) objective. Labelled RECENT.
  * CHAUN-style - attention-based uplift: feature self-attention encoder feeding
                TARNet-style heads. Labelled RECENT (attention).

Category labels: STRONG_MODERN (TARNet/CFRNet/DragonNet), RECENT (DESCN/CHAUN).
All train on a documented subsample (neural_train_n) and predict on the FULL locked
val/test splits in mini-batches. Features are standardized (scaler fit on train only).
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from . import config as C

torch.set_num_threads(8)


def _mlp(inp, hidden, depth, out, act=nn.ELU):
    layers, d = [], inp
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers += [nn.Linear(d, out)]
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# Network definitions
# --------------------------------------------------------------------------- #
class _TARNetNet(nn.Module):
    def __init__(self, d_in, rep=128, hid=64, depth=3, dragon=False):
        super().__init__()
        self.rep = _mlp(d_in, rep, depth, rep)
        self.h1 = _mlp(rep, hid, 2, 1)
        self.h0 = _mlp(rep, hid, 2, 1)
        self.dragon = dragon
        if dragon:
            self.t_head = nn.Sequential(nn.Linear(rep, 1))
            self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        r = self.rep(x)
        y1 = self.h1(r).squeeze(-1)
        y0 = self.h0(r).squeeze(-1)
        if self.dragon:
            t = self.t_head(r).squeeze(-1)
            return y1, y0, t, r
        return y1, y0, r


class _AttnNet(nn.Module):
    """CHAUN-style: per-feature embedding + self-attention encoder -> TARNet heads."""
    def __init__(self, d_in, emb=16, heads=4, hid=64):
        super().__init__()
        self.d_in = d_in; self.emb = emb
        self.feat_emb = nn.Linear(1, emb)                       # shared scalar->emb
        self.pos = nn.Parameter(torch.randn(d_in, emb) * 0.1)
        self.attn = nn.MultiheadAttention(emb, heads, batch_first=True)
        self.ln = nn.LayerNorm(emb)
        rep = d_in * emb
        self.h1 = _mlp(rep, hid, 2, 1)
        self.h0 = _mlp(rep, hid, 2, 1)

    def forward(self, x):
        b = x.shape[0]
        tok = self.feat_emb(x.unsqueeze(-1)) + self.pos.unsqueeze(0)   # (b, d_in, emb)
        a, _ = self.attn(tok, tok, tok)
        z = self.ln(a + tok).reshape(b, -1)
        return self.h1(z).squeeze(-1), self.h0(z).squeeze(-1), z


class _DESCNNet(nn.Module):
    """DESCN-style: shared rep -> propensity + entire-space response heads."""
    def __init__(self, d_in, rep=128, hid=64, depth=3):
        super().__init__()
        self.rep = _mlp(d_in, rep, depth, rep)
        self.p_head = nn.Linear(rep, 1)          # propensity logit
        self.mu1 = _mlp(rep, hid, 2, 1)          # P(Y|T=1)
        self.mu0 = _mlp(rep, hid, 2, 1)          # P(Y|T=0)

    def forward(self, x):
        r = self.rep(x)
        return self.mu1(r).squeeze(-1), self.mu0(r).squeeze(-1), self.p_head(r).squeeze(-1), r


# --------------------------------------------------------------------------- #
# Wrapper with shared training loop
# --------------------------------------------------------------------------- #
class NeuralCATE:
    def __init__(self, cfg: C.RunConfig, kind: str = "tarnet",
                 epochs: int = 20, batch: int = 4096, lr: float = 1e-3,
                 alpha_ipm: float = 1.0, beta_tr: float = 1.0):
        self.cfg = cfg; self.kind = kind
        self.epochs = epochs; self.batch = batch; self.lr = lr
        self.alpha_ipm = alpha_ipm; self.beta_tr = beta_tr
        self.scaler = StandardScaler()
        self.net = None
        self.category = {"tarnet": "STRONG_MODERN", "cfrnet": "STRONG_MODERN",
                         "dragonnet": "STRONG_MODERN", "descn": "RECENT",
                         "chaun": "RECENT"}[kind]
        self.name = {"tarnet": "TARNet", "cfrnet": "CFRNet", "dragonnet": "DragonNet",
                     "descn": "DESCN-style", "chaun": "CHAUN-attn"}[kind]

    def _build(self, d_in):
        torch.manual_seed(self.cfg.seed)
        if self.kind in ("tarnet", "cfrnet"):
            return _TARNetNet(d_in, dragon=False)
        if self.kind == "dragonnet":
            return _TARNetNet(d_in, dragon=True)
        if self.kind == "chaun":
            return _AttnNet(d_in)
        if self.kind == "descn":
            return _DESCNNet(d_in)
        raise ValueError(self.kind)

    @staticmethod
    def _linear_mmd(r, t):
        rt = r[t == 1]; rc = r[t == 0]
        if len(rt) == 0 or len(rc) == 0:
            return torch.tensor(0.0)
        return ((rt.mean(0) - rc.mean(0)) ** 2).sum()

    def fit(self, X, T, Y, val_frac: float = 0.1):
        X = self.scaler.fit_transform(np.asarray(X, np.float64)).astype(np.float32)
        T = np.asarray(T, np.float32); Y = np.asarray(Y, np.float32)
        n = len(X); rng = np.random.default_rng(self.cfg.seed)
        idx = rng.permutation(n); nv = int(val_frac * n)
        vi, ti = idx[:nv], idx[nv:]
        Xt = torch.tensor(X[ti]); Tt = torch.tensor(T[ti]); Yt = torch.tensor(Y[ti])
        Xv = torch.tensor(X[vi]); Tv = torch.tensor(T[vi]); Yv = torch.tensor(Y[vi])
        self.net = self._build(X.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        bce = nn.BCEWithLogitsLoss()
        best_val = np.inf; best_state = None; patience = 4; bad = 0
        nb = int(np.ceil(len(ti) / self.batch))
        for ep in range(self.epochs):
            self.net.train(); perm = torch.randperm(len(ti))
            for b in range(nb):
                bi = perm[b * self.batch:(b + 1) * self.batch]
                xb, tb, yb = Xt[bi], Tt[bi], Yt[bi]
                opt.zero_grad()
                loss = self._loss(xb, tb, yb, bce)
                loss.backward(); opt.step()
            # validation (factual BCE)
            self.net.eval()
            with torch.no_grad():
                vloss = self._loss(Xv, Tv, Yv, bce, train=False).item()
            if vloss < best_val - 1e-5:
                best_val = vloss; best_state = {k: v.clone() for k, v in self.net.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _loss(self, xb, tb, yb, bce, train=True):
        if self.kind in ("tarnet", "cfrnet"):
            y1, y0, r = self.net(xb)
            yhat = torch.where(tb == 1, y1, y0)
            loss = bce(yhat, yb)
            if self.kind == "cfrnet" and train:
                loss = loss + self.alpha_ipm * self._linear_mmd(r, tb)
            return loss
        if self.kind == "dragonnet":
            y1, y0, tlogit, r = self.net(xb)
            yhat = torch.where(tb == 1, y1, y0)
            loss = bce(yhat, yb) + bce(tlogit, tb)
            if train and self.beta_tr > 0:
                e = torch.sigmoid(tlogit).clamp(1e-3, 1 - 1e-3)
                q = torch.sigmoid(yhat)
                h = tb / e - (1 - tb) / (1 - e)
                y_pert = q + self.net.eps * h
                loss = loss + self.beta_tr * ((yb - y_pert) ** 2).mean()
            return loss
        if self.kind == "chaun":
            y1, y0, _ = self.net(xb)
            yhat = torch.where(tb == 1, y1, y0)
            return bce(yhat, yb)
        if self.kind == "descn":
            mu1, mu0, tlogit, r = self.net(xb)
            yhat = torch.where(tb == 1, mu1, mu0)
            # factual outcome + propensity + entire-space consistency
            e = torch.sigmoid(tlogit).clamp(1e-3, 1 - 1e-3)
            es = e * torch.sigmoid(mu1) + (1 - e) * torch.sigmoid(mu0)   # marginal P(Y)
            loss = bce(yhat, yb) + bce(tlogit, tb) + ((torch.sigmoid(yb) - es) ** 2).mean() * 0.1
            return loss
        raise ValueError(self.kind)

    def _forward_probs(self, X):
        Xs = self.scaler.transform(np.asarray(X, np.float64)).astype(np.float32)
        self.net.eval(); outs1 = []; outs0 = []
        with torch.no_grad():
            for b in range(0, len(Xs), 200000):
                xb = torch.tensor(Xs[b:b + 200000])
                out = self.net(xb)
                y1, y0 = out[0], out[1]
                outs1.append(torch.sigmoid(y1).numpy()); outs0.append(torch.sigmoid(y0).numpy())
        return np.concatenate(outs1), np.concatenate(outs0)

    def predict_cate(self, X):
        p1, p0 = self._forward_probs(X)
        return (p1 - p0).astype(np.float64)

    def predict_outcome(self, X, T):
        p1, p0 = self._forward_probs(X)
        T = np.asarray(T)
        return np.where(T == 1, p1, p0).astype(np.float64)
