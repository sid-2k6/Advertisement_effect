"""
Treatment / propensity / overlap diagnostics.

CRITICAL FRAMING (per user correction #4):
  * Criteo is a randomized incrementality experiment. Unconfoundedness holds BY DESIGN.
  * The propensity model e(x)=P(T=1|X) is estimated ONLY for: treatment-balance
    diagnostics, overlap / common-support checks, robustness, and doubly-robust
    (AIPW) estimation. IPTW is NOT required for identification here, and we do not
    claim it is. The treatment marginal is imbalanced (~85% treated) but assignment
    is (near) random; a propensity AUC ~0.5 is EVIDENCE of randomization, and is
    NOT itself a reason to prefer any particular CATE learner.
"""
from __future__ import annotations
from typing import Dict

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import config as C


def fit_propensity(X, T, params: dict, method: str = "lgbm"):
    """Fit e(x)=P(T=1|X). Returns fitted model. Used for diagnostics + DR only."""
    if method == "logistic":
        m = LogisticRegression(max_iter=1000, n_jobs=8)
        m.fit(X, T)
    else:
        m = LGBMClassifier(**{**params, "objective": "binary"})
        m.fit(X, T)
    return m


def propensity_scores(model, X) -> np.ndarray:
    p = model.predict_proba(X)[:, 1]
    return np.clip(p, 1e-4, 1 - 1e-4).astype(np.float64)


def standardized_mean_diff(X, T, weights=None) -> np.ndarray:
    """Per-feature standardized mean difference between arms (optionally weighted)."""
    X = np.asarray(X, np.float64); T = np.asarray(T)
    t_mask = T == 1; c_mask = T == 0
    if weights is None:
        w = np.ones(len(X))
    else:
        w = np.asarray(weights, np.float64)
    def wmean(a, m):
        return np.average(a[m], axis=0, weights=w[m])
    def wvar(a, m):
        mu = wmean(a, m)
        return np.average((a[m] - mu) ** 2, axis=0, weights=w[m])
    mt, mc = wmean(X, t_mask), wmean(X, c_mask)
    vt, vc = wvar(X, t_mask), wvar(X, c_mask)
    pooled = np.sqrt((vt + vc) / 2.0) + 1e-12
    return (mt - mc) / pooled


def stabilized_weights(T, e, clip: float = 0.01) -> np.ndarray:
    """Stabilized IPTW with clipping. (Used for robustness/diagnostics, not identification.)"""
    T = np.asarray(T, np.float64); e = np.asarray(e, np.float64)
    p_t = T.mean()
    e = np.clip(e, clip, 1 - clip)
    sw = np.where(T == 1, p_t / e, (1 - p_t) / (1 - e))
    return sw


def effective_sample_size(w) -> float:
    w = np.asarray(w, np.float64)
    return float((w.sum() ** 2) / np.sum(w ** 2))


def propensity_report(model, X, T, n_bins: int = 30) -> Dict:
    e = propensity_scores(model, X)
    T = np.asarray(T)
    auc = float(roc_auc_score(T, e)) if len(np.unique(T)) > 1 else float("nan")
    smd_before = standardized_mean_diff(X, T, weights=None)
    sw = stabilized_weights(T, e)
    smd_after = standardized_mean_diff(X, T, weights=sw)
    ess = effective_sample_size(sw)
    # common support: overlap of e-distributions across arms
    e_t = e[T == 1]; e_c = e[T == 0]
    lo = max(e_t.min(), e_c.min()); hi = min(e_t.max(), e_c.max())
    frac_in_support = float(((e >= lo) & (e <= hi)).mean())
    rep = {
        "propensity_auc": auc,
        "e_min": float(e.min()), "e_max": float(e.max()),
        "e_mean": float(e.mean()), "e_std": float(e.std()),
        "e_mean_treated": float(e_t.mean()), "e_mean_control": float(e_c.mean()),
        "common_support_range": [float(lo), float(hi)],
        "frac_in_common_support": frac_in_support,
        "max_abs_smd_before": float(np.max(np.abs(smd_before))),
        "max_abs_smd_after": float(np.max(np.abs(smd_after))),
        "smd_before": smd_before.tolist(),
        "smd_after": smd_after.tolist(),
        "stabilized_weight_min": float(sw.min()),
        "stabilized_weight_max": float(sw.max()),
        "effective_sample_size": ess,
        "n": int(len(T)),
        "note": ("Randomized assignment: propensity used for diagnostics/robustness/DR "
                 "only; IPTW not required for identification."),
    }
    return rep, e
