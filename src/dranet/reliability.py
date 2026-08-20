"""
Reliability-specific metrics that directly test the DRA-Net contribution.

Everything here is RCT-valid (within-group treated-minus-control rates). Because
individual counterfactual effects are unobservable, "positive-effect precision"
etc. are defined at the GROUP level (observed uplift among flagged users). We do
NOT assert individual calibration; we report group diagnostics honestly.
"""
from __future__ import annotations
from typing import Dict

import numpy as np
from scipy.stats import norm, spearmanr

from . import metrics as M


def _obs_uplift(Y, T):
    Y = np.asarray(Y, np.float64); T = np.asarray(T)
    n1 = (T == 1).sum(); n0 = (T == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float(Y[T == 1].mean() - Y[T == 0].mean())


def reliability_report(cate, spread, reliability, Y, T, e, c, gamma, z,
                       target_fracs=(0.05, 0.10, 0.20, 0.30), n_bins=10) -> Dict:
    Y = np.asarray(Y); T = np.asarray(T)
    cate = np.asarray(cate); spread = np.asarray(spread); r = np.asarray(reliability)
    out: Dict = {}

    # interval width + group coverage
    width = 2 * z * spread
    out["mean_interval_width"] = float(width.mean())
    out["median_interval_width"] = float(np.median(width))

    # confident-positive / confident-negative / uncertain shares
    r_neg = norm.cdf((-c - cate) / spread)                 # P(tau < -c)
    out["confident_positive_pct"] = float((r >= gamma).mean())
    out["confident_negative_pct"] = float((r_neg >= gamma).mean())
    out["uncertain_pct"] = float(1.0 - (r >= gamma).mean() - (r_neg >= gamma).mean())

    # positive-effect precision/recall at group level
    flagged = r >= gamma
    out["precision_high_r_obs_uplift"] = _obs_uplift(Y[flagged], T[flagged]) if flagged.sum() > 0 else np.nan
    # recall proxy: share of total incremental captured by flagged vs treat-all
    inc_flag = M.policy_incremental_outcome(Y, T, flagged.astype(float), e)
    inc_all = M.policy_incremental_outcome(Y, T, np.ones(len(Y)), e)
    out["recall_incremental_captured"] = float(inc_flag / inc_all) if inc_all != 0 else np.nan

    # observed uplift by reliability decile (high r first) and by CATE decile
    out["uplift_by_r_decile"] = _decile_uplift(Y, T, r, n_bins)
    out["uplift_by_cate_decile"] = _decile_uplift(Y, T, cate, n_bins)

    # policy value by reliability decile
    out["policy_value_by_r_decile"] = _decile_policy(Y, T, r, e, n_bins)

    # reliability calibration diagnostic: rank correlation between r-decile and observed uplift
    rd = out["uplift_by_r_decile"]["uplift"]
    ranks = list(range(len(rd), 0, -1))                    # decile 1 (high r) -> highest rank
    valid = [(rk, u) for rk, u in zip(ranks, rd) if u == u]
    if len(valid) >= 3:
        rr, uu = zip(*valid)
        rho, _ = spearmanr(rr, uu)
        out["reliability_monotonicity_spearman"] = float(rho)
    else:
        out["reliability_monotonicity_spearman"] = float("nan")
    return out


def _decile_uplift(Y, T, score, n_bins):
    Y = np.asarray(Y, np.float64); T = np.asarray(T)
    order = np.argsort(-np.asarray(score))
    n = len(score); edges = np.linspace(0, n, n_bins + 1).astype(int)
    dec, up, ns, mean_s = [], [], [], []
    for b in range(n_bins):
        sl = order[edges[b]:edges[b + 1]]
        dec.append(b + 1); up.append(_obs_uplift(Y[sl], T[sl])); ns.append(int(len(sl)))
        mean_s.append(float(np.asarray(score)[sl].mean()))
    return {"decile": dec, "uplift": up, "n": ns, "mean_score": mean_s}


def _decile_policy(Y, T, score, e, n_bins):
    Y = np.asarray(Y, np.float64); T = np.asarray(T)
    order = np.argsort(-np.asarray(score))
    n = len(score); edges = np.linspace(0, n, n_bins + 1).astype(int)
    vals = []
    for b in range(n_bins):
        sl = order[edges[b]:edges[b + 1]]
        pi = np.zeros(n); pi[sl] = 1.0
        vals.append(M.policy_value_ipw(Y, T, pi, e))
    return vals
