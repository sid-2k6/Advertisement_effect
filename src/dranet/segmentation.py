"""
User segmentation: the classical persuadable framework (Kane et al. 2014) PLUS a
reliability layer (the DRA-Net contribution). We then test whether the reliability
layer actually improves targeting quality (observed uplift of reliable-persuadables
vs all-persuadables at matched or comparable size).
"""
from __future__ import annotations
from typing import Dict

import numpy as np
from scipy.stats import norm


def segment(cate, mu0, spread, c, gamma, z) -> Dict[str, np.ndarray]:
    cate = np.asarray(cate, np.float64); mu0 = np.asarray(mu0, np.float64)
    spread = np.asarray(spread, np.float64)
    r_pos = norm.cdf((cate - c) / spread)        # P(tau > c)
    r_neg = norm.cdf((-c - cate) / spread)       # P(tau < -c)
    base_hi = mu0 >= np.median(mu0)

    base = np.empty(len(cate), dtype=object)
    base[:] = "unassigned"
    persuadable = cate > c
    sleeping = cate < -c
    flat = ~persuadable & ~sleeping
    base[persuadable] = "Persuadable"
    base[sleeping] = "SleepingDog"
    base[flat & base_hi] = "SureThing"
    base[flat & ~base_hi] = "LostCause"

    rel = np.empty(len(cate), dtype=object)
    rel[:] = "n/a"
    rel[persuadable & (r_pos >= gamma)] = "ReliablePersuadable"
    rel[persuadable & (r_pos < gamma)] = "UncertainPersuadable"
    rel[sleeping & (r_neg >= gamma)] = "ReliableSleepingDog"
    rel[sleeping & (r_neg < gamma)] = "UncertainNegative"
    return {"base": base, "reliability": rel, "r_pos": r_pos, "r_neg": r_neg}


def segment_summary(seg, Y, T, e) -> Dict:
    from . import metrics as M
    Y = np.asarray(Y, np.float64); T = np.asarray(T)
    base, rel = seg["base"], seg["reliability"]
    n = len(Y)

    def obs_uplift(mask):
        if mask.sum() == 0: return np.nan
        yy, tt = Y[mask], T[mask]
        if (tt == 1).sum() == 0 or (tt == 0).sum() == 0: return np.nan
        return float(yy[tt == 1].mean() - yy[tt == 0].mean())

    out = {"base_counts": {}, "base_uplift": {}, "reliability_counts": {}, "reliability_uplift": {}}
    for lab in ["Persuadable", "SleepingDog", "SureThing", "LostCause"]:
        m = base == lab
        out["base_counts"][lab] = int(m.sum())
        out["base_pct"] = out.get("base_pct", {}); out["base_pct"][lab] = float(m.mean())
        out["base_uplift"][lab] = obs_uplift(m)
    for lab in ["ReliablePersuadable", "UncertainPersuadable", "ReliableSleepingDog", "UncertainNegative"]:
        m = rel == lab
        out["reliability_counts"][lab] = int(m.sum())
        out["reliability_uplift"][lab] = obs_uplift(m)

    # DOES THE RELIABILITY LAYER HELP? observed uplift: all persuadables vs reliable persuadables
    out["persuadable_all_uplift"] = obs_uplift(base == "Persuadable")
    out["persuadable_reliable_uplift"] = obs_uplift(rel == "ReliablePersuadable")
    out["reliability_layer_gain"] = (
        out["persuadable_reliable_uplift"] - out["persuadable_all_uplift"]
        if (out["persuadable_reliable_uplift"] == out["persuadable_reliable_uplift"]
            and out["persuadable_all_uplift"] == out["persuadable_all_uplift"]) else np.nan
    )
    return out
