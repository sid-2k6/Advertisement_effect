"""
Evaluation metrics for causal / uplift / policy / reliability analysis.

All uplift metrics are computed from a *ranking* induced by a model's targeting
score s(x) (higher = target first) together with the observed (T, Y) on a held-out
split. They are RCT-valid because Criteo assignment is randomized.

IMPORTANT (per brief): Qini/AUUC magnitudes are implementation-normalized; there
is NO universal "good" threshold. We therefore (a) fix ONE consistent definition
used for every model, (b) also report the random-relative normalized Qini, and
(c) attach bootstrap CIs for all comparisons.

Definitions used
----------------
Rank units by score descending. For a prefix containing n_t treated and n_c control
with r_t, r_c responders (sum of Y):
  * Qini curve:   q(k) = r_t(k) - r_c(k) * n_t(k)/n_c(k)
  * Uplift curve: u(k) = ( r_t(k)/n_t(k) - r_c(k)/n_c(k) ) * (n_t(k)+n_c(k))
  * Uplift@f    : observed incremental RATE in the top-f fraction targeted =
                  r_t/n_t - r_c/n_c  (i.e. the within-group ATE among targeted users)
  * qini_coef   : area between q(.) and the random diagonal, normalised by the
                  random-to-perfect range  (dimensionless, comparable across models)
  * auuc        : area under the uplift curve divided by N (per-sample)
Policy value uses an IPW estimator for a binary policy pi(x) in {0,1}.
"""
from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score, average_precision_score, log_loss,
    brier_score_loss, confusion_matrix,
)


# --------------------------------------------------------------------------- #
# Core cumulative curves
# --------------------------------------------------------------------------- #
def _cumulative(y: np.ndarray, t: np.ndarray, score: np.ndarray
                ) -> Dict[str, np.ndarray]:
    """Sort by score desc and return cumulative counts/responders + population frac."""
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="mergesort")   # stable
    y, t = y[order], t[order]
    n_t = np.cumsum(t)
    n_c = np.cumsum(1.0 - t)
    r_t = np.cumsum(y * t)
    r_c = np.cumsum(y * (1.0 - t))
    n = len(y)
    frac = np.arange(1, n + 1) / n
    return {"n_t": n_t, "n_c": n_c, "r_t": r_t, "r_c": r_c, "frac": frac, "N": n}


def qini_curve(y, t, score) -> Tuple[np.ndarray, np.ndarray]:
    cu = _cumulative(y, t, score)
    n_c_safe = np.where(cu["n_c"] == 0, 1.0, cu["n_c"])
    q = cu["r_t"] - cu["r_c"] * (cu["n_t"] / n_c_safe)
    x = np.concatenate([[0.0], cu["frac"]])
    q = np.concatenate([[0.0], q])
    return x, q


def uplift_curve(y, t, score) -> Tuple[np.ndarray, np.ndarray]:
    cu = _cumulative(y, t, score)
    n_t_safe = np.where(cu["n_t"] == 0, 1.0, cu["n_t"])
    n_c_safe = np.where(cu["n_c"] == 0, 1.0, cu["n_c"])
    u = (cu["r_t"] / n_t_safe - cu["r_c"] / n_c_safe) * (cu["n_t"] + cu["n_c"])
    x = np.concatenate([[0.0], cu["frac"]])
    u = np.concatenate([[0.0], u])
    return x, u


_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _area(x, yv) -> float:
    return float(_trapz(yv, x))


def qini_coefficient(y, t, score) -> float:
    """Random-relative normalised Qini in (-inf, 1]; 0=random, 1=perfect ordering."""
    x, q = qini_curve(y, t, score)
    q_end = q[-1]
    rand = x * q_end                                  # diagonal from (0,0)->(1,q_end)
    area_model = _area(x, q)
    area_rand = _area(x, rand)
    # perfect ordering: all treated responders first, then controls non-responders last
    xp, qp = qini_curve(y, t, _perfect_score(y, t))
    area_perf = _area(xp, qp)
    denom = area_perf - area_rand
    if abs(denom) < 1e-12:
        return 0.0
    return float((area_model - area_rand) / denom)


def qini_area(y, t, score) -> float:
    """Unnormalised area between qini curve and random diagonal, divided by N."""
    x, q = qini_curve(y, t, score)
    rand = x * q[-1]
    return float((_area(x, q) - _area(x, rand)) / _cumulative(y, t, score)["N"])


def auuc(y, t, score) -> float:
    """Area under uplift curve divided by N (per-sample)."""
    x, u = uplift_curve(y, t, score)
    return float(_area(x, u) / len(y))


def fast_uplift(y, t, score, fracs=(0.05, 0.10, 0.20, 0.30)) -> Dict[str, float]:
    """Single-sort computation of qini_area, auuc, and uplift@k (for fast bootstrap)."""
    cu = _cumulative(y, t, score)
    n_t, n_c, r_t, r_c, frac, N = (cu["n_t"], cu["n_c"], cu["r_t"], cu["r_c"], cu["frac"], cu["N"])
    n_c_safe = np.where(n_c == 0, 1.0, n_c); n_t_safe = np.where(n_t == 0, 1.0, n_t)
    q = r_t - r_c * (n_t / n_c_safe)
    x = np.concatenate([[0.0], frac]); qv = np.concatenate([[0.0], q])
    qini_area = float((_area(x, qv) - _area(x, x * qv[-1])) / N)
    u = (r_t / n_t_safe - r_c / n_c_safe) * (n_t + n_c)
    auuc = float(_area(x, np.concatenate([[0.0], u])) / N)
    out = {"qini": qini_area, "auuc": auuc}
    for f in fracs:
        k = max(1, int(round(f * N)))
        rt = r_t[k - 1] / n_t_safe[k - 1]; rc = r_c[k - 1] / n_c_safe[k - 1]
        out[f"uplift@{int(f*100)}"] = float(rt - rc)
    return out


def _perfect_score(y, t) -> np.ndarray:
    """Oracle ranking for Qini: treated-responders & control-non-responders first."""
    y = np.asarray(y); t = np.asarray(t)
    s = np.where((t == 1) & (y == 1), 3,
        np.where((t == 0) & (y == 0), 2,
        np.where((t == 1) & (y == 0), 1, 0)))
    return s.astype(np.float64) + np.random.default_rng(0).random(len(y)) * 1e-6


def uplift_at_fraction(y, t, score, frac: float) -> Dict[str, float]:
    """Observed incremental RATE and cumulative gain in the top-`frac` targeted."""
    y = np.asarray(y, np.float64); t = np.asarray(t, np.float64)
    order = np.argsort(-np.asarray(score, np.float64), kind="mergesort")
    k = max(1, int(round(frac * len(y))))
    top = order[:k]
    yt, tt = y[top], t[top]
    n_t = tt.sum(); n_c = (1 - tt).sum()
    rate_t = (yt * tt).sum() / n_t if n_t > 0 else 0.0
    rate_c = (yt * (1 - tt)).sum() / n_c if n_c > 0 else 0.0
    uplift_rate = rate_t - rate_c
    return {
        "uplift_rate": float(uplift_rate),
        "n_treated": int(n_t), "n_control": int(n_c),
        "rate_treated": float(rate_t), "rate_control": float(rate_c),
    }


def uplift_by_decile(y, t, score, n_bins: int = 10) -> Dict[str, list]:
    """Observed uplift (treated-rate minus control-rate) within score deciles."""
    y = np.asarray(y, np.float64); t = np.asarray(t, np.float64)
    score = np.asarray(score, np.float64)
    order = np.argsort(-score, kind="mergesort")
    y, t = y[order], t[order]
    n = len(y)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    out = {"decile": [], "uplift": [], "n": [], "rate_t": [], "rate_c": []}
    for b in range(n_bins):
        sl = slice(edges[b], edges[b + 1])
        yt, tt = y[sl], t[sl]
        n_t = tt.sum(); n_c = (1 - tt).sum()
        rt = (yt * tt).sum() / n_t if n_t > 0 else np.nan
        rc = (yt * (1 - tt)).sum() / n_c if n_c > 0 else np.nan
        out["decile"].append(b + 1)
        out["uplift"].append(float(rt - rc))
        out["n"].append(int(len(yt)))
        out["rate_t"].append(float(rt))
        out["rate_c"].append(float(rc))
    return out


# --------------------------------------------------------------------------- #
# Policy value (IPW), incremental outcome
# --------------------------------------------------------------------------- #
def policy_value_ipw(y, t, pi, e) -> float:
    """IPW estimate of E[Y under policy pi] on an RCT split.

    V(pi) = mean[ Y * ( T*pi/e + (1-T)*(1-pi)/(1-e) ) ].
    With near-constant e this is stable; e may be scalar or per-unit.
    """
    y = np.asarray(y, np.float64); t = np.asarray(t, np.float64)
    pi = np.asarray(pi, np.float64)
    e = np.asarray(e, np.float64) if np.ndim(e) else np.full_like(y, float(e))
    e = np.clip(e, 1e-3, 1 - 1e-3)
    w = t * pi / e + (1 - t) * (1 - pi) / (1 - e)
    return float(np.mean(y * w))


def policy_incremental_outcome(y, t, pi, e) -> float:
    """Incremental outcome of pi vs treat-nobody, per targeted user summed:
    difference of IPW value(pi) and IPW value(all-zeros) times N."""
    v_pi = policy_value_ipw(y, t, pi, e)
    v_none = policy_value_ipw(y, t, np.zeros_like(pi), e)
    return float((v_pi - v_none) * len(y))


# --------------------------------------------------------------------------- #
# CATE calibration (GATES-style, RCT-valid) and PEHE (synthetic only)
# --------------------------------------------------------------------------- #
def gates_calibration(y, t, cate_hat, n_bins: int = 20) -> Dict:
    """Bin by predicted CATE; observed within-bin ATE = treated-rate - control-rate.
    Returns per-bin predicted vs observed, calibration error (mean abs), and slope.
    """
    y = np.asarray(y, np.float64); t = np.asarray(t, np.float64)
    cate_hat = np.asarray(cate_hat, np.float64)
    order = np.argsort(cate_hat)
    y, t, ch = y[order], t[order], cate_hat[order]
    n = len(y)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    pred, obs, ns = [], [], []
    for b in range(n_bins):
        sl = slice(edges[b], edges[b + 1])
        yt, tt, cc = y[sl], t[sl], ch[sl]
        n_t = tt.sum(); n_c = (1 - tt).sum()
        if n_t == 0 or n_c == 0:
            continue
        rt = (yt * tt).sum() / n_t
        rc = (yt * (1 - tt)).sum() / n_c
        pred.append(float(cc.mean()))
        obs.append(float(rt - rc))
        ns.append(int(len(yt)))
    pred = np.array(pred); obs = np.array(obs)
    cal_err = float(np.mean(np.abs(pred - obs))) if len(pred) else float("nan")
    # calibration slope: regress observed on predicted (weighted by bin size)
    if len(pred) >= 2 and np.std(pred) > 1e-9:
        w = np.array(ns, np.float64)
        pm = np.average(pred, weights=w); om = np.average(obs, weights=w)
        slope = float(np.sum(w * (pred - pm) * (obs - om)) /
                      np.sum(w * (pred - pm) ** 2))
    else:
        slope = float("nan")
    return {"pred": pred.tolist(), "obs": obs.tolist(), "n": ns,
            "calibration_error": cal_err, "calibration_slope": slope}


# --------------------------------------------------------------------------- #
# Classification / prediction diagnostics (supporting only)
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, p_hat, threshold: float = 0.5) -> Dict:
    y_true = np.asarray(y_true, np.int64)
    p_hat = np.asarray(p_hat, np.float64)
    p_hat = np.clip(p_hat, 1e-7, 1 - 1e-7)
    y_pred = (p_hat >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    assert tn + fp + fn + tp == len(y_true), "confusion matrix does not sum to N"
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p_hat)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, p_hat)) if len(np.unique(y_true)) > 1 else float("nan"),
        "log_loss": float(log_loss(y_true, p_hat, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p_hat)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return out


def cate_distribution_stats(cate_hat) -> Dict:
    c = np.asarray(cate_hat, np.float64)
    pct = np.percentile(c, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "cate_mean": float(c.mean()), "cate_std": float(c.std()),
        "cate_min": float(c.min()), "cate_max": float(c.max()),
        "pct_positive": float((c > 0).mean()), "pct_negative": float((c < 0).mean()),
        "percentiles": {str(p): float(v) for p, v in
                        zip([1, 5, 10, 25, 50, 75, 90, 95, 99], pct)},
    }
