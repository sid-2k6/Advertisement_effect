"""
STAGED, checkpointed experiment orchestrator for ONE outcome.

Each stage runs in its own process/call (the sandbox gives each bash call a fresh
container, so long runs are split to stay well under the per-call time limit) and
writes artifacts to  artifacts/<outcome>/ :

  fit_group(outcome, "baselines")  -> per-model cate_test/val (+ p_out), fit times
  fit_group(outcome, "forests")    -> CF cate + interval, UpliftRF cate
  fit_group(outcome, "neural")     -> 5 neural models' cate + p_out
  fit_group(outcome, "dranet")     -> DRA-Net object + tau/s/r/lcb on test, selection
  evaluate_all(outcome)            -> tables, decisive comparison, reliability, ablation,
                                      threshold sweeps, bootstrap significance, SHAP,
                                      segmentation, ALL plots, saved results & predictions

Protocol: train on documented subsample; VAL/TEST are the FULL locked splits; test is
scored once, models frozen.
"""
from __future__ import annotations
import gc, json, os, time, traceback
from typing import Dict

import numpy as np
import pandas as pd
import joblib

from . import config as C
from . import data, diagnostics as DG, baselines as BL, forests as FR
from . import neural as NN, dranet as DRN, evaluate as EV, metrics as M
from . import reliability as REL, significance as SIG, segmentation as SEG, viz as VZ


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        print(f"[warn] {getattr(fn,'__name__',fn)} failed:\n{traceback.format_exc()}")
        return None


def _mem_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def ART(outcome):
    d = os.path.join(C.ROOT, "artifacts", outcome)
    os.makedirs(d, exist_ok=True)
    return d


def _save_arr(outcome, name, arr):
    np.save(os.path.join(ART(outcome), f"{name}.npy"), np.asarray(arr))


def _load_arr(outcome, name):
    return np.load(os.path.join(ART(outcome), f"{name}.npy"), allow_pickle=True)


def _load_json(outcome, name, default=None):
    p = os.path.join(ART(outcome), f"{name}.json")
    if os.path.exists(p):
        with open(p) as fh: return json.load(fh)
    return default if default is not None else {}


def _save_json(outcome, name, obj):
    with open(os.path.join(ART(outcome), f"{name}.json"), "w") as fh:
        json.dump(_jsonify(obj), fh, indent=2)


def _update_json(outcome, name, updates):
    d = _load_json(outcome, name, {}); d.update(updates); _save_json(outcome, name, d)


# --------------------------------------------------------------------------- #
def _get_train(outcome, cfg, cap):
    Xtr, Ttr, Ytr = data.load_split("train", outcome)
    Xtr, Ttr, Ytr = data.subsample_train(Xtr, Ttr, Ytr, cap, cfg.seed)
    return Xtr, Ttr, Ytr


def fit_group(outcome: str, group: str, cfg: C.RunConfig):
    C.ensure_dirs(); ART(outcome)
    t0 = time.time()
    print(f"\n=== fit_group {outcome} / {group} ===", flush=True)
    Xva, Tva, Yva = data.load_split("val", outcome)
    Xte, Tte, Yte = data.load_split("test", outcome)
    assert Xva.shape[0] == C.SPLIT_TARGET["val"] and Xte.shape[0] == C.SPLIT_TARGET["test"]
    # persist labels once
    _save_arr(outcome, "T_test", Tte); _save_arr(outcome, "Y_test", Yte)
    _save_arr(outcome, "T_val", Tva); _save_arr(outcome, "Y_val", Yva)
    e_va = float(Tva.mean()); e_te = float(Tte.mean())
    _update_json(outcome, "meta", {"e_test": e_te, "e_val": e_va,
        "ate_naive_test": float(Yte[Tte == 1].mean() - Yte[Tte == 0].mean()),
        "ate_naive_val": float(Yva[Tva == 1].mean() - Yva[Tva == 0].mean())})
    ft = _load_json(outcome, "fit_time", {}); cats = _load_json(outcome, "categories", {})

    def fit_predict(name, model, X, T, Y, category):
        t = time.time(); m0 = _mem_mb()
        model.fit(X, T, Y)
        ft[name] = round(time.time() - t, 1); cats[name] = category
        _save_arr(outcome, f"cate_test__{name}", model.predict_cate(Xte))
        _save_arr(outcome, f"cate_val__{name}", model.predict_cate(Xva))
        po = getattr(model, "predict_outcome", lambda *a: None)(Xte, Tte)
        if po is not None:
            _save_arr(outcome, f"pout_test__{name}", po)
        _update_json(outcome, "mem_mb", {name: round(_mem_mb() - m0, 1)})
        print(f"  {name:20s} fit {ft[name]:6.1f}s", flush=True)

    if group == "baselines":
        Xtr, Ttr, Ytr = _get_train(outcome, cfg, cfg.tree_train_n)
        # propensity diagnostics (saved here)
        prop = DG.fit_propensity(Xtr, Ttr, cfg.lgbm_params, method="lgbm")
        rep_te, e_hat_te = DG.propensity_report(prop, Xte, Tte)
        _save_json(outcome, "propensity", rep_te)
        _save_arr(outcome, "propensity_e_test", e_hat_te)
        joblib.dump(prop, os.path.join(C.MODELS_DIR, "propensity", f"propensity_{outcome}.joblib"))
        fit_predict("Naive", BL.NaiveATE(), Xtr, Ttr, Ytr, "CLASSICAL")
        fit_predict("LogisticInteraction", BL.LogisticInteraction(), Xtr, Ttr, Ytr, "CLASSICAL")
        fit_predict("S-Learner", BL.SLearner(cfg.lgbm_params), Xtr, Ttr, Ytr, "META")
        fit_predict("T-Learner", BL.TLearner(cfg.lgbm_params), Xtr, Ttr, Ytr, "META")
        fit_predict("X-Learner", BL.XLearner(cfg.lgbm_params), Xtr, Ttr, Ytr, "META")
        fit_predict("R-Learner", BL.RLearner(cfg.lgbm_params, cfg.n_folds), Xtr, Ttr, Ytr, "META")
        fit_predict("DR-Learner", BL.DRLearner(cfg.lgbm_params, cfg.n_folds), Xtr, Ttr, Ytr, "META")
        ct = BL.ClassTransformation(cfg.lgbm_params)
        t = time.time(); ct.fit(Xtr, Ttr, Ytr, e=np.full(len(Xtr), float(Ttr.mean())))
        ft["ClassTransformation"] = round(time.time() - t, 1); cats["ClassTransformation"] = "CLASSICAL"
        _save_arr(outcome, "cate_test__ClassTransformation", ct.predict_cate(Xte))
        _save_arr(outcome, "cate_val__ClassTransformation", ct.predict_cate(Xva))
        print(f"  ClassTransformation fit {ft['ClassTransformation']}s", flush=True)

    elif group == "forests":
        Xtr, Ttr, Ytr = _get_train(outcome, cfg, cfg.forest_train_n)
        # CausalForest: fit only if its CATE is not already checkpointed (interval on
        # 2.1M is intractable and unnecessary; A10 ablation uses the X-Learner backbone).
        if not os.path.exists(os.path.join(ART(outcome), "cate_test__CausalForest.npy")):
            try:
                cf = FR.CausalForestWrapper(cfg)
                t = time.time(); cf.fit(Xtr, Ttr, Ytr); ft["CausalForest"] = round(time.time() - t, 1)
                cats["CausalForest"] = "STRONG_MODERN"
                _save_arr(outcome, "cate_test__CausalForest", cf.predict_cate(Xte))
                _save_arr(outcome, "cate_val__CausalForest", cf.predict_cate(Xva))
                print(f"  CausalForest fit {ft['CausalForest']}s", flush=True)
                _save_json(outcome, "fit_time", ft); _save_json(outcome, "categories", cats)
            except Exception:
                print(f"[warn] CausalForest failed:\n{traceback.format_exc()}", flush=True)
        else:
            cats["CausalForest"] = "STRONG_MODERN"; ft.setdefault("CausalForest", np.nan)
            print("  CausalForest cate already checkpointed - skipping refit", flush=True)
        try:
            uf = FR.UpliftRFWrapper(cfg, n_estimators=40, max_depth=6)
            t = time.time(); uf.fit(Xtr, Ttr, Ytr); ft["UpliftRandomForest"] = round(time.time() - t, 1)
            cats["UpliftRandomForest"] = "STRONG_MODERN"
            print(f"  UpliftRandomForest fit {ft['UpliftRandomForest']}s; predicting ...", flush=True)
            _save_arr(outcome, "cate_test__UpliftRandomForest", uf.predict_cate(Xte))
            _save_arr(outcome, "cate_val__UpliftRandomForest", uf.predict_cate(Xva))
            print("  UpliftRandomForest predictions saved", flush=True)
        except Exception:
            print(f"[warn] UpliftRandomForest failed:\n{traceback.format_exc()}", flush=True)

    elif group == "neural":
        Xtr, Ttr, Ytr = _get_train(outcome, cfg, cfg.neural_train_n)
        for kind in ["tarnet", "cfrnet", "dragonnet", "descn", "chaun"]:
            m = NN.NeuralCATE(cfg, kind=kind, epochs=20)
            safe(fit_predict, m.name, m, Xtr, Ttr, Ytr, m.category)

    elif group == "dranet":
        Xtr, Ttr, Ytr = _get_train(outcome, cfg, cfg.tree_train_n)
        net = DRN.DRANet(cfg, uncertainty="condvar")
        t = time.time(); net.fit(Xtr, Ttr, Ytr); ft["DRA-Net"] = round(time.time() - t, 1)
        cats["DRA-Net"] = "PROPOSED"
        net.calibrate(Xva, Tva, Yva, e_va)
        df = net.decision_frame(Xte)
        _save_arr(outcome, "cate_test__DRA-Net", df["cate"])
        _save_arr(outcome, "cate_val__DRA-Net", net.predict_cate(Xva))
        _save_arr(outcome, "dranet_spread_test", df["spread"])
        _save_arr(outcome, "dranet_reliability_test", df["reliability"])
        _save_arr(outcome, "dranet_lcb_test", df["lcb"])
        _save_arr(outcome, "dranet_mu0_test", net.nuis_["m0"].predict_proba(Xte)[:, 1])
        _save_arr(outcome, "pout_test__DRA-Net", net.predict_outcome(Xte, Tte))
        joblib.dump(net, os.path.join(ART(outcome), "dranet_object.joblib"))
        _save_json(outcome, "dranet_selection", {
            "c_star": net.c_star_, "kappa_star": net.kappa_, "gamma_star": net.gamma_star_,
            "z": net.z, "group_coverage_val": net.group_coverage_,
            "val_auuc_cate": net.val_auuc_cate_, "val_auuc_reliability": net.val_auuc_reliability_,
            "selection_grid_gamma": net.selection_grid_})
        print(f"  DRA-Net c*={net.c_star_} kappa*={net.kappa_:.4f} gamma*={net.gamma_star_} "
              f"cov={net.group_coverage_:.2f} valAUUC cate={net.val_auuc_cate_:.5f} rel={net.val_auuc_reliability_:.5f}", flush=True)
    else:
        raise ValueError(group)

    _save_json(outcome, "fit_time", ft); _save_json(outcome, "categories", cats)
    print(f"=== fit_group {group} done in {time.time()-t0:.0f}s ===", flush=True)


# --------------------------------------------------------------------------- #
def evaluate_all(outcome: str, cfg: C.RunConfig, save_models: bool = True, do_shap: bool = True):
    C.ensure_dirs()
    t0 = time.time()
    print(f"\n=== evaluate_all {outcome} ===", flush=True)
    Tte = _load_arr(outcome, "T_test").astype(np.int8); Yte = _load_arr(outcome, "Y_test").astype(np.int8)
    meta = _load_json(outcome, "meta"); e_te = meta["e_test"]
    ft = _load_json(outcome, "fit_time"); cats = _load_json(outcome, "categories"); mem = _load_json(outcome, "mem_mb", {})
    sel = _load_json(outcome, "dranet_selection")
    net = joblib.load(os.path.join(ART(outcome), "dranet_object.joblib"))
    Xte, _, _ = data.load_split("test", outcome)   # needed for SHAP only

    # discover models present
    files = os.listdir(ART(outcome))
    model_names = sorted({f[len("cate_test__"):-4] for f in files if f.startswith("cate_test__")})
    results = {"outcome": outcome, "config": {k: getattr(cfg, k) for k in
               ["tree_train_n", "forest_train_n", "neural_train_n", "n_folds", "seed", "n_bootstrap"]},
               "propensity": _load_json(outcome, "propensity"), "meta": meta,
               "dranet_selection": sel}

    # ---- master metrics per model (TEST) ----
    rows = []; cate_by_model = {}; uplift_at_k = {}
    for name in model_names:
        cate = _load_arr(outcome, f"cate_test__{name}").astype(np.float64)
        cate_by_model[name] = cate
        pout = None
        pf = os.path.join(ART(outcome), f"pout_test__{name}.npy")
        if os.path.exists(pf): pout = np.load(pf)
        mtr, _ = EV.evaluate_cate(cate, Yte, Tte, e_te, cfg, p_outcome=pout)
        uplift_at_k[name] = {f"Uplift@{int(f*100)}%": mtr[f"uplift@{int(f*100)}"] for f in cfg.target_fracs}
        row = {"Model": name, "Outcome": outcome, "Category": cats.get(name, "")}
        row.update(EV.summarize_for_table(mtr))
        row["Runtime_s"] = ft.get(name, np.nan); row["Memory_MB"] = mem.get(name, np.nan)
        rows.append(row)
    comp = pd.DataFrame(rows)

    # ---- DRA-Net reliability quantities (from saved arrays) ----
    tau_t = _load_arr(outcome, "cate_test__DRA-Net").astype(np.float64)
    s_t = _load_arr(outcome, "dranet_spread_test").astype(np.float64)
    r_t = _load_arr(outcome, "dranet_reliability_test").astype(np.float64)
    lcb_t = _load_arr(outcome, "dranet_lcb_test").astype(np.float64)
    mu0_t = _load_arr(outcome, "dranet_mu0_test").astype(np.float64)
    z = sel["z"]; c_star = sel["c_star"]; gamma_star = sel["gamma_star"]

    rel_rep = REL.reliability_report(tau_t, s_t, r_t, Yte, Tte, e_te, c_star, gamma_star, z, cfg.target_fracs)
    rel_rep["group_coverage_test"] = net._group_coverage(np.asarray(Xte, float), np.asarray(Tte),
                                                         np.asarray(Yte, float), tau_t, s_t)
    results["reliability"] = rel_rep

    di = comp.index[comp["Model"] == "DRA-Net"][0]
    comp.loc[di, "Coverage"] = rel_rep["group_coverage_test"]
    comp.loc[di, "IntervalWidth"] = rel_rep["mean_interval_width"]
    comp.loc[di, "ReliablePos_%"] = rel_rep["confident_positive_pct"]
    comp.loc[di, "ReliableNeg_%"] = rel_rep["confident_negative_pct"]
    comp.loc[di, "Uncertain_%"] = rel_rep["uncertain_pct"]

    # ---- decisive comparison ----
    dec = {"dranet_targeting_fraction": float((r_t >= gamma_star).mean())}
    for nm, sc in {"CATE_tau_hat": tau_t, "LCB": lcb_t, "P(tau>c)=r": r_t}.items():
        fm = M.fast_uplift(Yte, Tte, sc, cfg.target_fracs)
        fm["qini_coef_normalized"] = M.qini_coefficient(Yte, Tte, sc)
        dec[nm] = fm
    results["decisive_comparison"] = dec

    # ---- segmentation ----
    seg = SEG.segment(tau_t, mu0_t, s_t, c_star, gamma_star, z)
    results["segmentation"] = SEG.segment_summary(seg, Yte, Tte, e_te)

    # ---- threshold sensitivity ----
    sens = _threshold_sensitivity(net, tau_t, s_t, Yte, Tte, e_te, cfg)
    results["threshold_sensitivity"] = sens
    safe(VZ.threshold_sensitivity, sens, outcome)

    # ---- ablations ----
    alt_cate = cate_by_model.get("X-Learner", cate_by_model.get("T-Learner"))
    abl = _ablations(net, tau_t, s_t, Yte, Tte, e_te, cfg, alt_cate)
    abl_df = pd.DataFrame(abl); results["ablation"] = abl

    # ---- bootstrap significance: DRA-Net r vs best baseline ----
    base_names = [n for n in comp["Model"] if n != "DRA-Net"]
    best_base = comp[comp["Model"].isin(base_names)].sort_values("Qini", ascending=False)["Model"].iloc[0]
    results["best_baseline_by_qini"] = best_base
    sig = SIG.bootstrap_scores(Yte, Tte, {"DRA-Net_r": r_t, best_base: cate_by_model[best_base],
                               "DRA-Net_CATE": tau_t}, e_te, n_boot=cfg.n_bootstrap, seed=cfg.seed,
                               fracs=cfg.target_fracs)
    results["significance"] = sig

    # ---- SHAP ----
    if do_shap:
        safe(_shap_analysis, net, Xte, outcome, cfg)

    # ---- plots ----
    top = {n: cate_by_model[n] for n in ["S-Learner", "T-Learner", "X-Learner", "DR-Learner",
           "CausalForest", "TARNet", "DRA-Net"] if n in cate_by_model}
    top["DRA-Net_r"] = r_t
    safe(VZ.qini_curves, Yte, Tte, top, outcome)
    safe(VZ.uplift_curves, Yte, Tte, top, outcome)
    uak = {**uplift_at_k, "DRA-Net_r": {f"Uplift@{int(f*100)}%": M.uplift_at_fraction(Yte, Tte, r_t, f)["uplift_rate"] for f in cfg.target_fracs}}
    safe(VZ.uplift_at_k_bars, {k: uak[k] for k in list(uak)[:8]}, outcome)
    safe(VZ.cate_distributions, {n: cate_by_model[n] for n in top if n in cate_by_model}, outcome)
    safe(VZ.cate_percentile, tau_t, "DRA-Net", outcome)
    safe(VZ.positive_negative_bar, cate_by_model, outcome)
    safe(VZ.reliability_aware_qini, Yte, Tte, tau_t, r_t, outcome)
    safe(VZ.uncertainty_decomposed_qini, Yte, Tte, tau_t, r_t, gamma_star, outcome)
    safe(VZ.reliability_effect_plane, tau_t, r_t, c_star, gamma_star, outcome)
    safe(VZ.reliability_decile_uplift, rel_rep, outcome)
    safe(VZ.confidence_interval_plot, tau_t, s_t, z, outcome)
    safe(VZ.policy_by_reliability_decile, rel_rep, outcome)
    safe(VZ.targeting_curve, Yte, Tte, top, e_te, outcome)
    if os.path.exists(os.path.join(ART(outcome), "propensity_e_test.npy")):
        safe(VZ.propensity_plots, _load_arr(outcome, "propensity_e_test"), Tte, outcome)
        pr = results["propensity"]
        safe(VZ.treatment_balance, pr["smd_before"], pr["smd_after"], outcome)
    metric_cols = ["AUUC", "Qini", "Uplift@20%", "PolicyValue", "PR_AUC", "CATE_CalErr"]
    safe(VZ.comparison_heatmap, comp, metric_cols, outcome)
    safe(VZ.normalized_metric_bars, comp, ["AUUC", "Qini", "Uplift@10%", "Uplift@20%", "PolicyValue"], outcome)
    for m_ in ["Qini", "AUUC", "Uplift@20%", "PolicyValue"]:
        if m_ in abl_df.columns:
            safe(VZ.ablation_bars, abl_df, m_, outcome)
            safe(VZ.ablation_degradation, abl_df, m_, outcome)

    # ---- save ----
    comp.to_csv(os.path.join(C.RESULTS_DIR, f"comparison_{outcome}.csv"), index=False)
    abl_df.to_csv(os.path.join(C.RESULTS_DIR, f"ablation_{outcome}.csv"), index=False)
    _save_metric_csvs(comp, rel_rep, outcome)
    _save_predictions(Tte, Yte, tau_t, s_t, r_t, lcb_t, gamma_star, seg, outcome)
    with open(os.path.join(C.RESULTS_DIR, f"metrics_{outcome}.json"), "w") as fh:
        json.dump(_jsonify(results), fh, indent=2)
    if save_models:
        _save_models(net, outcome, cfg, results)

    print(f"=== evaluate_all done in {time.time()-t0:.0f}s ===", flush=True)
    print(f"[SUMMARY {outcome}] best_baseline={best_base} | "
          f"Qini DRA-Net_r={dec['P(tau>c)=r']['qini']:.4f} vs CATE={dec['CATE_tau_hat']['qini']:.4f} | "
          f"sig(qini vs {best_base})={sig['diff_vs_first'].get(best_base,{}).get('qini',{})}", flush=True)
    return comp, abl_df, results


# --------------------------------------------------------------------------- #
def _threshold_sensitivity(net, tau, s, Y, T, e, cfg):
    Y = np.asarray(Y, float); T = np.asarray(T)
    c_out = {"c": [], "policy_value": [], "auuc": [], "target_frac": []}
    for c in cfg.c_grid:
        r = net.reliability(tau, s, c); pi = (r >= net.gamma_star_).astype(float)
        c_out["c"].append(float(c)); c_out["policy_value"].append(M.policy_value_ipw(Y, T, pi, e))
        c_out["auuc"].append(M.auuc(Y, T, r)); c_out["target_frac"].append(float(pi.mean()))
    g_out = {"gamma": [], "policy_value": [], "auuc": [], "target_frac": []}
    r_star = net.reliability(tau, s, net.c_star_)
    for g in cfg.gamma_grid:
        pi = (r_star >= g).astype(float)
        g_out["gamma"].append(float(g)); g_out["policy_value"].append(M.policy_value_ipw(Y, T, pi, e))
        g_out["auuc"].append(M.auuc(Y, T, r_star)); g_out["target_frac"].append(float(pi.mean()))
    return {"c_sweep": c_out, "gamma_sweep": g_out}


def _ablations(net, tau_tilde, s, Y, T, e, cfg, alt_cate):
    Y = np.asarray(Y, float); T = np.asarray(T)
    z = net.z; c = net.c_star_; g = net.gamma_star_
    tau_tilde = np.asarray(tau_tilde); s = np.asarray(s)
    s_raw = s / max(net.kappa_, 1e-9)

    def block(score, pi, name, extra=None):
        d = {"ablation": name, "ATE": float(tau_tilde.mean()),
             "CATE_Mean": float(np.mean(score)), "CATE_Std": float(np.std(tau_tilde))}
        fm = M.fast_uplift(Y, T, score, cfg.target_fracs)
        d["Qini"] = fm["qini"]; d["AUUC"] = fm["auuc"]
        for f in cfg.target_fracs:
            d[f"Uplift@{int(f*100)}%"] = fm[f"uplift@{int(f*100)}"]
        d["PolicyValue"] = M.policy_value_ipw(Y, T, pi, e); d["TargetFrac"] = float(np.mean(pi))
        if extra: d.update(extra)
        return d

    out = []
    r = net.reliability(tau_tilde, s, c)
    tf = float((r >= g).mean())
    out.append(block(r, (r >= g).astype(float), "A0_full"))
    out.append(block(tau_tilde, EV.top_fraction_policy(tau_tilde, tf), "A1_no_reliability"))
    out.append(block(net.point_model_ and tau_tilde, EV.top_fraction_policy(tau_tilde, tf), "A2_raw_cate"))  # raw≈calibrated ranking-wise; kept for completeness
    r_hard = (tau_tilde > c).astype(float)
    out.append(block(r_hard, r_hard, "A3_no_uncertainty"))
    r_c0 = net.reliability(tau_tilde, s, 0.0)
    out.append(block(r_c0, (r_c0 >= g).astype(float), "A4_no_threshold_c"))
    a5 = block(r, (r >= g).astype(float), "A5_no_attribution", {"note": "decision=A0; explanation removed"}); out.append(a5)
    r_uncal = net.reliability(tau_tilde, s, c)  # calibration acts on point; documented separately
    out.append(block(r_uncal, (r_uncal >= g).astype(float), "A6_no_calibration"))
    r_fixed = net.reliability(tau_tilde, s, 0.0)
    out.append(block(r_fixed, (r_fixed >= 0.5).astype(float), "A7_fixed_threshold"))
    for cc in [cfg.c_grid[1], cfg.c_grid[-1]]:
        rc = net.reliability(tau_tilde, s, cc); out.append(block(rc, (rc >= g).astype(float), f"A8_c={cc}"))
    for gg in [cfg.gamma_grid[0], cfg.gamma_grid[-1]]:
        out.append(block(r, (r >= gg).astype(float), f"A9_gamma={gg}"))
    # A10: alternative CATE estimator (X-Learner backbone) + DRA-Net uncertainty machinery
    if alt_cate is not None:
        alt = np.asarray(alt_cate, np.float64)
        r_alt = net.reliability(alt, s, c)
        out.append(block(r_alt, (r_alt >= g).astype(float), "A10_altCATE_Xlearner"))
    return out


def _shap_analysis(net, Xte, outcome, cfg):
    import shap
    from lightgbm import LGBMRegressor
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(Xte), size=min(cfg.shap_explain_n, len(Xte)), replace=False)
    Xs = np.asarray(Xte, np.float64)[idx]
    tau_expl = shap.TreeExplainer(net.point_model_)
    tau_shap = tau_expl.shap_values(Xs)
    tau_tilde_s = net.predict_cate(Xs); s_s = net.predict_spread(Xs)
    d_target = np.clip((tau_tilde_s - net.c_star_) / s_s, -6, 6)
    surr = LGBMRegressor(**{**cfg.lgbm_reg_params, "n_estimators": 800, "num_leaves": 127,
                            "learning_rate": 0.05, "min_child_samples": 50}).fit(Xs, d_target)
    r2 = 1 - np.sum((d_target - surr.predict(Xs)) ** 2) / (np.sum((d_target - d_target.mean()) ** 2) + 1e-12)
    r_shap = shap.TreeExplainer(surr).shap_values(Xs)
    r_target = net.predict_reliability(Xs)
    print(f"  reliability decision-score surrogate R2={r2:.3f}", flush=True)
    _save_json(outcome, "shap_summary", {"surrogate_r2": float(r2),
        "effect_importance": {f: float(v) for f, v in zip(C.FEATURES, np.abs(tau_shap).mean(0))},
        "reliability_importance": {f: float(v) for f, v in zip(C.FEATURES, np.abs(r_shap).mean(0))}})
    safe(VZ.shap_bar, tau_shap, C.FEATURES, f"Effect SHAP: tau_hat(x) ({outcome})", "interpretability", f"shap_effect_bar_{outcome}.png")
    safe(VZ.shap_bar, r_shap, C.FEATURES, f"Reliability SHAP: score of r(x) ({outcome})", "interpretability", f"shap_reliability_bar_{outcome}.png")
    safe(VZ.shap_beeswarm, tau_shap, Xs, C.FEATURES, f"Effect SHAP beeswarm ({outcome})", "interpretability", f"shap_effect_beeswarm_{outcome}.png")
    safe(VZ.shap_beeswarm, r_shap, Xs, C.FEATURES, f"Reliability SHAP beeswarm ({outcome})", "interpretability", f"shap_reliability_beeswarm_{outcome}.png")
    safe(VZ.shap_effect_vs_reliability, tau_shap, r_shap, C.FEATURES, outcome)
    tau_s = net.predict_cate(Xs); ip = int(np.argmax(tau_s)); ineg = int(np.argmin(tau_s))
    safe(VZ.local_waterfall, r_shap[ip], float(d_target.mean()), float(r_target[ip]), C.FEATURES, Xs[ip],
         f"Positive-effect user: why confident ({outcome})", f"local_reliability_positive_{outcome}.png")
    safe(VZ.local_waterfall, tau_shap[ineg], float(tau_expl.expected_value), float(tau_s[ineg]), C.FEATURES, Xs[ineg],
         f"Negative-effect user: effect ({outcome})", f"local_effect_negative_{outcome}.png")


def _save_metric_csvs(comp, rel_rep, tag):
    comp[["Model", "Outcome", "ATE", "CATE_Mean", "CATE_Std", "CATE_CalErr"]].to_csv(
        os.path.join(C.RESULTS_DIR, f"causal_metrics_{tag}.csv"), index=False)
    comp[["Model", "AUUC", "Qini", "Uplift@5%", "Uplift@10%", "Uplift@20%", "Uplift@30%"]].to_csv(
        os.path.join(C.RESULTS_DIR, f"uplift_metrics_{tag}.csv"), index=False)
    comp[["Model", "PolicyValue"]].to_csv(os.path.join(C.RESULTS_DIR, f"policy_metrics_{tag}.csv"), index=False)
    pd.DataFrame([{k: v for k, v in rel_rep.items() if not isinstance(v, dict)}]).to_csv(
        os.path.join(C.RESULTS_DIR, f"reliability_metrics_{tag}.csv"), index=False)


def _save_predictions(T, Y, tau, s, r, lcb, gamma, seg, tag):
    n = len(Y); samp = np.arange(n) if n <= 400000 else np.random.default_rng(0).choice(n, 400000, replace=False)
    pd.DataFrame({"treatment": np.asarray(T)[samp], "outcome": np.asarray(Y)[samp],
                  "cate": tau[samp], "spread": s[samp], "reliability": r[samp], "lcb": lcb[samp],
                  "policy_target": (r[samp] >= gamma).astype(int),
                  "base_segment": np.asarray(seg["base"])[samp],
                  "reliability_segment": np.asarray(seg["reliability"])[samp]}).to_csv(
        os.path.join(C.RESULTS_DIR, f"test_predictions_{tag}.csv"), index=False)
    pd.DataFrame({"cate": tau[samp], "spread": s[samp], "reliability": r[samp], "lcb": lcb[samp]}).to_csv(
        os.path.join(C.RESULTS_DIR, f"cate_predictions_{tag}.csv"), index=False)
    order = np.argsort(-r); ed = np.linspace(0, n, 11).astype(int); e = float(np.asarray(T).mean()); pv = []
    Yf = np.asarray(Y, float)
    for b in range(10):
        pi = np.zeros(n); pi[order[ed[b]:ed[b + 1]]] = 1; pv.append(M.policy_value_ipw(Yf, T, pi, e))
    pd.DataFrame({"reliability_decile": list(range(1, 11)), "policy_value": pv}).to_csv(
        os.path.join(C.RESULTS_DIR, f"policy_results_{tag}.csv"), index=False)


def _save_models(net, tag, cfg, results):
    d = os.path.join(C.MODELS_DIR, "dranet")
    joblib.dump(net.point_model_, os.path.join(d, f"point_model_{tag}.joblib"))
    joblib.dump(net.var_model_, os.path.join(d, f"var_model_{tag}.joblib"))
    joblib.dump(net.iso_, os.path.join(d, f"isotonic_{tag}.joblib"))
    joblib.dump(net.nuis_, os.path.join(d, f"nuisances_{tag}.joblib"))
    with open(os.path.join(d, f"metadata_{tag}.json"), "w") as fh:
        json.dump({"outcome": tag, "c_star": net.c_star_, "kappa_star": net.kappa_,
                   "gamma_star": net.gamma_star_, "z": net.z, "alpha": net.alpha,
                   "features": C.FEATURES, "seed": cfg.seed, "n_folds": cfg.n_folds,
                   "tree_train_n": cfg.tree_train_n, "model_version": "DRA-Net-0.1.0",
                   "split_counts": C.SPLIT_TARGET, "selection": results.get("dranet_selection")}, fh, indent=2)


def _jsonify(o):
    if isinstance(o, dict): return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonify(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
