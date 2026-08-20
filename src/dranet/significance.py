"""
Bootstrap confidence intervals and paired-bootstrap comparisons.

Resamples rows with replacement (full test size) and recomputes uplift metrics.
For each replicate and each score we sort once and derive qini/auuc/uplift@k
together for efficiency. Paired bootstrap uses the SAME resampled indices for two
scores so that the difference CI accounts for correlation.
"""
from __future__ import annotations
from typing import Callable, Dict, List

import numpy as np

from . import metrics as M


def _metrics_for_score(y, t, score, fracs) -> Dict[str, float]:
    # single-sort fast metrics (qini here is the per-sample qini AREA, comparable across models)
    return M.fast_uplift(y, t, score, fracs)


def bootstrap_scores(Y, T, scores: Dict[str, np.ndarray], e,
                     n_boot=200, seed=42, fracs=(0.05, 0.10, 0.20, 0.30),
                     policy_pi: Dict[str, np.ndarray] | None = None) -> Dict:
    """Paired bootstrap over rows. `scores` maps model-name -> targeting score.
    Optional `policy_pi` maps model-name -> binary policy (for policy-value CI).
    Returns per-model metric distributions and pairwise diffs vs the first key.
    """
    Y = np.asarray(Y, np.float64); T = np.asarray(T); e_arr = np.asarray(e, np.float64) if np.ndim(e) else None
    n = len(Y); rng = np.random.default_rng(seed)
    names = list(scores.keys())
    metrics = ["qini", "auuc"] + [f"uplift@{int(f*100)}" for f in fracs]
    dist = {nm: {m: [] for m in metrics} for nm in names}
    if policy_pi:
        for nm in policy_pi: dist[nm]["policy_value"] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, tb = Y[idx], T[idx]
        eb = e_arr[idx] if e_arr is not None else float(np.mean(tb))
        for nm in names:
            md = _metrics_for_score(yb, tb, scores[nm][idx], fracs)
            for m in metrics:
                dist[nm][m].append(md[m])
            if policy_pi and nm in policy_pi:
                dist[nm]["policy_value"].append(M.policy_value_ipw(yb, tb, policy_pi[nm][idx], eb))

    def summ(arr):
        a = np.asarray(arr)
        return {"mean": float(a.mean()), "lo95": float(np.percentile(a, 2.5)),
                "hi95": float(np.percentile(a, 97.5)), "std": float(a.std())}

    out = {"per_model": {}, "diff_vs_first": {}, "n_boot": n_boot, "reference": names[0]}
    for nm in names:
        out["per_model"][nm] = {m: summ(v) for m, v in dist[nm].items()}
    ref = names[0]
    for nm in names[1:]:
        out["diff_vs_first"][nm] = {}
        for m in dist[nm]:
            if m in dist[ref]:
                d = np.asarray(dist[nm][m]) - np.asarray(dist[ref][m])
                out["diff_vs_first"][nm][m] = {
                    "mean_diff": float(d.mean()),
                    "lo95": float(np.percentile(d, 2.5)),
                    "hi95": float(np.percentile(d, 97.5)),
                    "p_gt_0": float((d > 0).mean()),
                    "significant_95": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
                }
    return out
