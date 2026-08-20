"""
Shared evaluation harness: given any model exposing predict_cate (and optionally
predict_outcome) plus an evaluation split (X, T, Y) and propensity e, compute the
full metric suite (causal + uplift + policy + calibration + classification).

The targeting policy for a CATE model at fraction f targets the top-f users by
predicted CATE (this is the standard "CATE ranking" policy that DRA-Net is tested
against). Policy value is the RCT-valid IPW estimate.
"""
from __future__ import annotations
from typing import Dict, Optional

import numpy as np

from . import metrics as M
from . import config as C


def top_fraction_policy(score: np.ndarray, frac: float) -> np.ndarray:
    n = len(score); k = max(1, int(round(frac * n)))
    thr = np.partition(score, n - k)[n - k]
    return (score >= thr).astype(np.int8)


def evaluate_cate(cate: np.ndarray, Y, T, e, cfg: C.RunConfig,
                  p_outcome: Optional[np.ndarray] = None,
                  compute_policy: bool = True) -> Dict:
    """Full metric dict for a precomputed CATE vector on an eval split."""
    Y = np.asarray(Y); T = np.asarray(T)
    cate = np.asarray(cate, np.float64)
    assert len(cate) == len(Y) == len(T), "eval length mismatch"

    out: Dict = {}
    out["ate_hat"] = float(cate.mean())
    out.update(M.cate_distribution_stats(cate))

    # uplift metrics (ranking by CATE, higher targeted first)
    out["qini_coef"] = M.qini_coefficient(Y, T, cate)
    out["qini_area"] = M.qini_area(Y, T, cate)
    out["auuc"] = M.auuc(Y, T, cate)
    for f in cfg.target_fracs:
        u = M.uplift_at_fraction(Y, T, cate, f)
        out[f"uplift@{int(f*100)}"] = u["uplift_rate"]

    # calibration (GATES, RCT-valid)
    cal = M.gates_calibration(Y, T, cate, n_bins=cfg.n_gates_bins)
    out["cate_calibration_error"] = cal["calibration_error"]
    out["cate_calibration_slope"] = cal["calibration_slope"]

    # policy value at each target fraction (IPW), plus best fraction
    if compute_policy:
        for f in cfg.target_fracs:
            pi = top_fraction_policy(cate, f)
            out[f"policy_value@{int(f*100)}"] = M.policy_value_ipw(Y, T, pi, e)
            out[f"incremental@{int(f*100)}"] = M.policy_incremental_outcome(Y, T, pi, e)
        out["policy_value_treat_all"] = M.policy_value_ipw(Y, T, np.ones(len(Y)), e)
        out["policy_value_treat_none"] = M.policy_value_ipw(Y, T, np.zeros(len(Y)), e)

    # classification (supporting only)
    if p_outcome is not None:
        out.update({f"clf_{k}": v for k, v in
                    M.classification_metrics(Y, p_outcome).items()})
    return out, cal


def summarize_for_table(metrics: Dict) -> Dict:
    """Pick the master-table columns from a full metrics dict (policy@20 as headline)."""
    row = {
        "ATE": metrics.get("ate_hat"),
        "CATE_Mean": metrics.get("cate_mean"),
        "CATE_Std": metrics.get("cate_std"),
        "AUUC": metrics.get("auuc"),
        "Qini": metrics.get("qini_coef"),
        "Uplift@5%": metrics.get("uplift@5"),
        "Uplift@10%": metrics.get("uplift@10"),
        "Uplift@20%": metrics.get("uplift@20"),
        "Uplift@30%": metrics.get("uplift@30"),
        "PolicyValue": metrics.get("policy_value@20"),
        "PR_AUC": metrics.get("clf_pr_auc"),
        "ROC_AUC": metrics.get("clf_roc_auc"),
        "F1": metrics.get("clf_f1"),
        "Brier": metrics.get("clf_brier"),
        "CATE_CalErr": metrics.get("cate_calibration_error"),
        "Pos_CATE_%": metrics.get("pct_positive"),
        "Neg_CATE_%": metrics.get("pct_negative"),
    }
    return row
