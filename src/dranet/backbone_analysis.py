"""
Reliability-layer generality + validation-based backbone selection.

The DRA-Net contribution is a decision LAYER: r(x)=P(tau(x)>c) built on ANY point-CATE
backbone plus the AIPW conditional-dispersion spread s(x). This module:
  (1) applies the reliability layer to several backbones (whose val/test CATE are already
      checkpointed) using DRA-Net's spread, calibrating each and selecting (c,kappa) on VAL;
  (2) compares, per backbone, tau_hat-ranking vs r(x)-ranking on TEST (the core hypothesis);
  (3) selects the DRA-Net* backbone by validation r-AUUC (validation-based model selection);
  (4) bootstraps significance of r vs tau_hat for the selected backbone.

Everything reuses cached predictions; the only new computation is s(x) on val/test.
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression
from scipy.stats import norm

from . import config as C, data, metrics as M, significance as SIG
from .run_experiment import ART, _load_arr, _load_json


def _iso_calibrate(cate_val, Tval, Yval, cate_test, n_bins):
    cal = M.gates_calibration(Yval, Tval, cate_val, n_bins=n_bins)
    pred = np.array(cal["pred"]); obs = np.array(cal["obs"])
    if len(pred) >= 2 and np.std(pred) > 1e-9:
        iso = IsotonicRegression(out_of_bounds="clip").fit(pred, obs)
        return iso.predict(cate_val), iso.predict(cate_test)
    return cate_val, cate_test


def run(outcome: str, cfg: C.RunConfig,
        backbones=("DR-Learner", "S-Learner", "X-Learner", "T-Learner", "TARNet", "CFRNet", "CausalForest")):
    net = joblib.load(os.path.join(ART(outcome), "dranet_object.joblib"))
    Xva, Tva, Yva = data.load_split("val", outcome)
    Xte, Tte, Yte = data.load_split("test", outcome)
    e_te = float(Tte.mean())
    z = net.z
    s_val = net.kappa_ * net._s_raw(Xva)
    s_test = net.kappa_ * net._s_raw(Xte)
    val_ate = float(Yva[Tva == 1].mean() - Yva[Tva == 0].mean())
    c_cand = sorted(set([round(x, 6) for x in cfg.c_grid] + [round(val_ate, 6)]))
    kappa_grid = np.concatenate([[1e-6], np.geomspace(0.01, 3.0, 20)])

    rows = []
    best = {"val_r_auuc": -1e9}
    for name in backbones:
        vp = os.path.join(ART(outcome), f"cate_val__{name}.npy")
        tp = os.path.join(ART(outcome), f"cate_test__{name}.npy")
        if not (os.path.exists(vp) and os.path.exists(tp)):
            continue
        cate_val = np.load(vp).astype(np.float64); cate_test = np.load(tp).astype(np.float64)
        tv, tt = _iso_calibrate(cate_val, Tva, Yva, cate_test, cfg.n_gates_bins)
        # select (c, kappa) on validation by AUUC of r-ranking
        bc, bk, bau = c_cand[0], 1e-6, -1e9
        for c in c_cand:
            for k in kappa_grid:
                r = norm.cdf((tv - c) / (k * (s_val / net.kappa_)))
                a = M.auuc(Yva, Tva, r)
                if a > bau:
                    bau, bc, bk = a, c, k
        s_te_b = bk * (s_test / net.kappa_)
        r_test = norm.cdf((tt - bc) / s_te_b)
        rec = {"backbone": name, "c_star": bc, "kappa_star": float(bk), "val_r_auuc": float(bau)}
        # test metrics: tau ranking vs r ranking
        for tag, score in [("tau", tt), ("r", r_test)]:
            fm = M.fast_uplift(Yte, Tte, score, cfg.target_fracs)
            rec[f"{tag}_qini"] = M.qini_coefficient(Yte, Tte, score)
            rec[f"{tag}_auuc"] = fm["auuc"]
            for f in cfg.target_fracs:
                rec[f"{tag}_uplift@{int(f*100)}"] = fm[f"uplift@{int(f*100)}"]
        rec["r_minus_tau_qini"] = rec["r_qini"] - rec["tau_qini"]
        rec["r_minus_tau_uplift@20"] = rec["r_uplift@20"] - rec["tau_uplift@20"]
        rec["r_test"] = r_test; rec["tau_test"] = tt
        rows.append(rec)
        if bau > best["val_r_auuc"]:
            best = rec

    # significance for selected backbone: r vs tau (paired bootstrap on test)
    sig = SIG.bootstrap_scores(Yte, Tte, {"r": best["r_test"], "tau": best["tau_test"]},
                               e_te, n_boot=cfg.n_bootstrap, seed=cfg.seed, fracs=cfg.target_fracs)
    out_rows = [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in rows]
    df = pd.DataFrame(out_rows).sort_values("val_r_auuc", ascending=False)
    df.to_csv(os.path.join(C.RESULTS_DIR, f"backbone_generality_{outcome}.csv"), index=False)
    summary = {"selected_backbone": best["backbone"], "c_star": best["c_star"],
               "kappa_star": best["kappa_star"],
               "test_qini_tau": best["tau_qini"], "test_qini_r": best["r_qini"],
               "test_uplift@20_tau": best["tau_uplift@20"], "test_uplift@20_r": best["r_uplift@20"],
               "significance_r_vs_tau": sig["diff_vs_first"].get("tau", {})}
    with open(os.path.join(C.RESULTS_DIR, f"backbone_generality_{outcome}.json"), "w") as fh:
        json.dump(_jsonify(summary), fh, indent=2)
    print(f"[BACKBONE GENERALITY {outcome}] selected={best['backbone']} "
          f"c*={best['c_star']} kappa*={best['kappa_star']:.4f}")
    print(df[["backbone", "val_r_auuc", "tau_qini", "r_qini", "r_minus_tau_qini",
              "tau_uplift@20", "r_uplift@20", "r_minus_tau_uplift@20"]].round(5).to_string(index=False))
    print("significance r vs tau (selected backbone), diff=tau-r (negative => r better):")
    print(json.dumps(_jsonify(sig["diff_vs_first"].get("tau", {})), indent=1))
    return df, summary


def _jsonify(o):
    if isinstance(o, dict): return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonify(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
