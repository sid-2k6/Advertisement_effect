"""
Publication-quality visualizations. Every function saves a PNG under plots/<subdir>/
and is wrapped by `safe()` in the orchestrator so a single failure never aborts a run.

Novel figures (the DRA-Net contribution):
  * reliability_effect_plane  -> plots/reliability/  (CATE vs r(x), 4 decision regions)
  * uncertainty_decomposed_qini -> plots/qini/       (confident vs speculative gain)
  * reliability_aware_qini    -> plots/qini/         (r-ranking vs CATE-ranking Qini)
"""
from __future__ import annotations
import os
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import metrics as M

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})
PALETTE = plt.get_cmap("tab10").colors


def _p(sub, name):
    return os.path.join(C.PLOTS_DIR, sub, name)


# --------------------------------------------------------------------------- #
# Causal identification / propensity
# --------------------------------------------------------------------------- #
def treatment_balance(smd_before, smd_after, outcome, tag=""):
    feats = C.FEATURES
    y = np.arange(len(feats))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(smd_before, y, label="unweighted", color=PALETTE[0])
    ax.scatter(smd_after, y, label="stabilized IPTW", color=PALETTE[1], marker="s")
    ax.axvline(0.1, ls="--", c="grey"); ax.axvline(-0.1, ls="--", c="grey")
    ax.set_yticks(y); ax.set_yticklabels(feats)
    ax.set_xlabel("Standardized mean difference"); ax.set_title(f"Covariate balance ({outcome})")
    ax.legend()
    fig.savefig(_p("causal", f"balance_{outcome}{tag}.png")); plt.close(fig)


def propensity_plots(e, T, outcome):
    T = np.asarray(T)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(e, bins=60, color=PALETTE[0]); ax[0].set_title("Propensity e(x)=P(T=1|X)")
    ax[0].set_xlabel("e(x)"); ax[0].set_ylabel("count")
    ax[1].hist(e[T == 1], bins=60, alpha=0.6, label="treated", density=True, color=PALETTE[1])
    ax[1].hist(e[T == 0], bins=60, alpha=0.6, label="control", density=True, color=PALETTE[2])
    ax[1].set_title("Propensity by arm (overlap)"); ax[1].set_xlabel("e(x)"); ax[1].legend()
    fig.suptitle(f"Propensity diagnostics ({outcome}) - randomized assignment")
    fig.savefig(_p("propensity", f"propensity_{outcome}.png")); plt.close(fig)


# --------------------------------------------------------------------------- #
# CATE distributions
# --------------------------------------------------------------------------- #
def cate_distributions(cate_by_model: Dict[str, np.ndarray], outcome):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (nm, c) in enumerate(cate_by_model.items()):
        c = np.asarray(c)
        lo, hi = np.percentile(c, [0.5, 99.5])
        ax.hist(np.clip(c, lo, hi), bins=80, histtype="step", density=True,
                label=nm, color=PALETTE[i % 10])
    ax.axvline(0, ls="--", c="k", lw=1)
    ax.set_xlabel("estimated CATE"); ax.set_ylabel("density")
    ax.set_title(f"CATE distributions ({outcome})"); ax.legend(fontsize=8)
    fig.savefig(_p("cate", f"cate_distributions_{outcome}.png")); plt.close(fig)


def cate_percentile(cate, name, outcome):
    c = np.sort(np.asarray(cate))
    q = np.linspace(0, 100, len(c))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(q, c, color=PALETTE[0]); ax.axhline(0, ls="--", c="k", lw=1)
    ax.set_xlabel("percentile"); ax.set_ylabel("CATE")
    ax.set_title(f"CATE percentile curve - {name} ({outcome})")
    fig.savefig(_p("cate", f"cate_percentile_{name}_{outcome}.png")); plt.close(fig)


def positive_negative_bar(cate_by_model: Dict[str, np.ndarray], outcome):
    names = list(cate_by_model); pos = [np.mean(np.asarray(c) > 0) for c in cate_by_model.values()]
    neg = [np.mean(np.asarray(c) < 0) for c in cate_by_model.values()]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(6, len(names)), 4))
    ax.bar(x - 0.2, pos, 0.4, label="% positive CATE", color=PALETTE[2])
    ax.bar(x + 0.2, neg, 0.4, label="% negative CATE", color=PALETTE[3])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title(f"Positive vs negative treatment-effect population ({outcome})"); ax.legend()
    fig.savefig(_p("cate", f"pos_neg_population_{outcome}.png")); plt.close(fig)


# --------------------------------------------------------------------------- #
# Uplift / Qini
# --------------------------------------------------------------------------- #
def qini_curves(Y, T, scores: Dict[str, np.ndarray], outcome, fname="qini_curves"):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (nm, s) in enumerate(scores.items()):
        x, q = M.qini_curve(Y, T, s)
        ax.plot(x, q, label=f"{nm} (Q={M.qini_coefficient(Y,T,s):.3f})", color=PALETTE[i % 10])
    xr = np.linspace(0, 1, 2); ax.plot(xr, xr * q[-1], ls="--", c="grey", label="random")
    ax.set_xlabel("fraction targeted"); ax.set_ylabel("cumulative incremental (Qini)")
    ax.set_title(f"Qini curves ({outcome})"); ax.legend(fontsize=8)
    fig.savefig(_p("qini", f"{fname}_{outcome}.png")); plt.close(fig)


def uplift_curves(Y, T, scores: Dict[str, np.ndarray], outcome):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (nm, s) in enumerate(scores.items()):
        x, u = M.uplift_curve(Y, T, s)
        ax.plot(x, u, label=f"{nm} (AUUC={M.auuc(Y,T,s):.4f})", color=PALETTE[i % 10])
    ax.set_xlabel("fraction targeted"); ax.set_ylabel("cumulative uplift")
    ax.set_title(f"Uplift curves ({outcome})"); ax.legend(fontsize=8)
    fig.savefig(_p("uplift", f"uplift_curves_{outcome}.png")); plt.close(fig)


def uplift_at_k_bars(uplift_at_k: Dict[str, Dict[str, float]], outcome):
    ks = ["Uplift@5%", "Uplift@10%", "Uplift@20%", "Uplift@30%"]
    names = list(uplift_at_k)
    x = np.arange(len(ks)); w = 0.8 / max(1, len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, nm in enumerate(names):
        vals = [uplift_at_k[nm].get(k, np.nan) for k in ks]
        ax.bar(x + i * w, vals, w, label=nm, color=PALETTE[i % 10])
    ax.set_xticks(x + w * len(names) / 2); ax.set_xticklabels(ks)
    ax.set_ylabel("observed uplift rate"); ax.set_title(f"Uplift@k ({outcome})")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(_p("uplift", f"uplift_at_k_{outcome}.png")); plt.close(fig)


# ---- NOVEL: reliability-aware Qini & uncertainty-decomposed Qini ---- #
def reliability_aware_qini(Y, T, cate, reliability, outcome):
    fig, ax = plt.subplots(figsize=(7, 6))
    for nm, s, col in [("CATE ranking", cate, PALETTE[0]),
                       ("Reliability r(x) ranking", reliability, PALETTE[3])]:
        x, q = M.qini_curve(Y, T, s)
        ax.plot(x, q, label=f"{nm} (Q={M.qini_coefficient(Y,T,s):.3f})", color=col)
    ax.plot([0, 1], [0, q[-1]], ls="--", c="grey", label="random")
    ax.set_xlabel("fraction targeted"); ax.set_ylabel("cumulative incremental (Qini)")
    ax.set_title(f"Reliability-aware Qini vs CATE Qini ({outcome})"); ax.legend()
    fig.savefig(_p("qini", f"reliability_aware_qini_{outcome}.png")); plt.close(fig)


def uncertainty_decomposed_qini(Y, T, cate, reliability, gamma, outcome):
    """Rank by CATE; decompose the cumulative Qini increment into contributions from
    CONFIDENT users (r>=gamma) vs SPECULATIVE users (r<gamma)."""
    Y = np.asarray(Y, np.float64); T = np.asarray(T, np.float64)
    order = np.argsort(-np.asarray(cate), kind="mergesort")
    y, t, r = Y[order], T[order], np.asarray(reliability)[order]
    conf = (r >= gamma).astype(np.float64)
    n_t = np.cumsum(t); n_c = np.cumsum(1 - t)
    n_c_safe = np.where(n_c == 0, 1.0, n_c)
    # confident-only responders and control adjustment
    r_t_conf = np.cumsum(y * t * conf); r_c_conf = np.cumsum(y * (1 - t) * conf)
    r_t_all = np.cumsum(y * t); r_c_all = np.cumsum(y * (1 - t))
    q_all = r_t_all - r_c_all * (n_t / n_c_safe)
    q_conf = r_t_conf - r_c_conf * (n_t / n_c_safe)
    q_spec = q_all - q_conf
    x = np.arange(1, len(y) + 1) / len(y)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill_between(x, 0, q_conf, color=PALETTE[2], alpha=0.6, label=f"confident (r>={gamma:.2f})")
    ax.fill_between(x, q_conf, q_all, color=PALETTE[1], alpha=0.5, label="speculative")
    ax.plot(x, q_all, color="k", lw=1, label="total Qini")
    ax.set_xlabel("fraction targeted (CATE order)"); ax.set_ylabel("cumulative incremental (Qini)")
    ax.set_title(f"Uncertainty-decomposed Qini ({outcome})"); ax.legend()
    fig.savefig(_p("qini", f"uncertainty_decomposed_qini_{outcome}.png")); plt.close(fig)


# --------------------------------------------------------------------------- #
# NOVEL: Reliability-Effect plane
# --------------------------------------------------------------------------- #
def reliability_effect_plane(cate, reliability, c, gamma, outcome, n_sample=20000, seed=42):
    cate = np.asarray(cate); r = np.asarray(reliability)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cate), size=min(n_sample, len(cate)), replace=False)
    cx, ry = cate[idx], r[idx]
    # region colors
    region = np.where((cx > c) & (ry >= gamma), 0,          # strongly target
             np.where((cx > c) & (ry < gamma), 1,           # uncertain target
             np.where((cx <= c) & (ry >= gamma), 2, 3)))    # avoid / uncertain-no
    labels = ["HIGH effect / HIGH reliability -> TARGET",
              "HIGH effect / LOW reliability -> uncertain",
              "LOW effect / HIGH reliability -> avoid",
              "LOW effect / LOW reliability -> no target"]
    cols = [PALETTE[2], PALETTE[1], PALETTE[3], PALETTE[7]]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for k in range(4):
        m = region == k
        ax.scatter(cx[m], ry[m], s=4, alpha=0.4, color=cols[k], label=labels[k])
    ax.axvline(c, ls="--", c="k"); ax.axhline(gamma, ls="--", c="k")
    ax.set_xlabel("estimated CATE  tau_tilde(x)"); ax.set_ylabel("reliability  r(x)=P(tau>c)")
    ax.set_title(f"DRA-Net Reliability-Effect plane ({outcome})\nc={c:.4f}, gamma={gamma:.2f}")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(_p("reliability", f"reliability_effect_plane_{outcome}.png")); plt.close(fig)


def reliability_decile_uplift(rel_report: Dict, outcome):
    rd = rel_report["uplift_by_r_decile"]; cd = rel_report["uplift_by_cate_decile"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rd["decile"], rd["uplift"], "-o", label="by reliability r(x)", color=PALETTE[3])
    ax.plot(cd["decile"], cd["uplift"], "-s", label="by CATE", color=PALETTE[0])
    ax.axhline(0, ls="--", c="k", lw=1)
    ax.set_xlabel("decile (1 = highest score)"); ax.set_ylabel("observed uplift (RCT-valid)")
    ax.set_title(f"Observed uplift by decile ({outcome})\nSpearman(r-decile,uplift)="
                 f"{rel_report.get('reliability_monotonicity_spearman', float('nan')):.2f}")
    ax.legend()
    fig.savefig(_p("reliability", f"reliability_decile_uplift_{outcome}.png")); plt.close(fig)


def confidence_interval_plot(cate, spread, z, outcome, n_sample=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cate), size=min(n_sample, len(cate)), replace=False)
    c = np.asarray(cate)[idx]; s = np.asarray(spread)[idx]
    o = np.argsort(c); c, s = c[o], s[o]
    x = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(x, c, yerr=z * s, fmt="o", ms=2, elinewidth=0.5, alpha=0.6, color=PALETTE[0])
    ax.axhline(0, ls="--", c="k", lw=1)
    ax.set_xlabel(f"users (sorted by CATE, sample of {len(c)})")
    ax.set_ylabel("CATE with confidence interval"); ax.set_title(f"CATE uncertainty intervals ({outcome})")
    fig.savefig(_p("reliability", f"cate_intervals_{outcome}.png")); plt.close(fig)


def threshold_sensitivity(sens: Dict, outcome):
    """sens has arrays for c-sweep and gamma-sweep with policy_value, auuc, target_frac."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    cs = sens["c_sweep"]
    ax[0, 0].plot(cs["c"], cs["policy_value"], "-o", color=PALETTE[0]); ax[0, 0].set_title("c vs policy value")
    ax[0, 1].plot(cs["c"], cs["auuc"], "-o", color=PALETTE[1]); ax[0, 1].set_title("c vs AUUC")
    ax[0, 2].plot(cs["c"], cs["target_frac"], "-o", color=PALETTE[2]); ax[0, 2].set_title("c vs targeting fraction")
    gs = sens["gamma_sweep"]
    ax[1, 0].plot(gs["gamma"], gs["policy_value"], "-o", color=PALETTE[0]); ax[1, 0].set_title("gamma vs policy value")
    ax[1, 1].plot(gs["gamma"], gs["auuc"], "-o", color=PALETTE[1]); ax[1, 1].set_title("gamma vs AUUC")
    ax[1, 2].plot(gs["gamma"], gs["target_frac"], "-o", color=PALETTE[2]); ax[1, 2].set_title("gamma vs targeting fraction")
    for a in ax.ravel(): a.set_xlabel("threshold")
    fig.suptitle(f"DRA-Net threshold sensitivity ({outcome})")
    fig.savefig(_p("reliability", f"threshold_sensitivity_{outcome}.png")); plt.close(fig)


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
def targeting_curve(Y, T, scores: Dict[str, np.ndarray], e, outcome, n_points=25):
    from .evaluate import top_fraction_policy
    fracs = np.linspace(0.02, 1.0, n_points)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for i, (nm, s) in enumerate(scores.items()):
        pv, inc = [], []
        for f in fracs:
            pi = top_fraction_policy(np.asarray(s), f)
            pv.append(M.policy_value_ipw(Y, T, pi, e))
            inc.append(M.policy_incremental_outcome(Y, T, pi, e))
        ax[0].plot(fracs, pv, label=nm, color=PALETTE[i % 10])
        ax[1].plot(fracs, inc, label=nm, color=PALETTE[i % 10])
    ax[0].set_title("Policy value vs targeting fraction"); ax[0].set_xlabel("fraction targeted"); ax[0].set_ylabel("E[Y] under policy (IPW)")
    ax[1].set_title("Incremental outcome vs targeting fraction"); ax[1].set_xlabel("fraction targeted"); ax[1].set_ylabel("total incremental outcome")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.suptitle(f"Targeting / policy curves ({outcome})")
    fig.savefig(_p("policy", f"targeting_curves_{outcome}.png")); plt.close(fig)


def policy_by_reliability_decile(rel_report, outcome):
    pv = rel_report["policy_value_by_r_decile"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(1, len(pv) + 1), pv, color=PALETTE[4])
    ax.set_xlabel("reliability decile (1 = highest r)"); ax.set_ylabel("policy value if target only this decile")
    ax.set_title(f"Policy value by reliability decile ({outcome})")
    fig.savefig(_p("policy", f"policy_by_r_decile_{outcome}.png")); plt.close(fig)


# --------------------------------------------------------------------------- #
# Interpretability (SHAP arrays precomputed in orchestrator)
# --------------------------------------------------------------------------- #
def shap_bar(shap_vals, feat_names, title, sub, fname):
    imp = np.abs(np.asarray(shap_vals)).mean(0)
    o = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(feat_names)[o], imp[o], color=PALETTE[0])
    ax.set_xlabel("mean |SHAP|"); ax.set_title(title)
    fig.savefig(_p(sub, fname)); plt.close(fig)


def shap_beeswarm(shap_vals, X_sample, feat_names, title, sub, fname):
    try:
        import shap
        fig = plt.figure(figsize=(7, 5))
        shap.summary_plot(np.asarray(shap_vals), features=np.asarray(X_sample),
                          feature_names=feat_names, show=False, plot_size=(7, 5))
        plt.title(title); plt.savefig(_p(sub, fname), bbox_inches="tight"); plt.close("all")
    except Exception:
        shap_bar(shap_vals, feat_names, title, sub, fname)


def shap_effect_vs_reliability(tau_shap, r_shap, feat_names, outcome):
    it = np.abs(np.asarray(tau_shap)).mean(0); ir = np.abs(np.asarray(r_shap)).mean(0)
    it = it / (it.sum() + 1e-12); ir = ir / (ir.sum() + 1e-12)
    y = np.arange(len(feat_names))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(y - 0.2, it, 0.4, label="effect  tau_hat(x)", color=PALETTE[0])
    ax.barh(y + 0.2, ir, 0.4, label="reliability  r(x)", color=PALETTE[3])
    ax.set_yticks(y); ax.set_yticklabels(feat_names)
    ax.set_xlabel("normalized mean |SHAP|")
    ax.set_title(f"What drives the EFFECT vs the CONFIDENCE ({outcome})"); ax.legend()
    fig.savefig(_p("interpretability", f"effect_vs_reliability_shap_{outcome}.png")); plt.close(fig)


def local_waterfall(shap_row, base_value, pred_value, feat_names, x_row, title, fname):
    order = np.argsort(-np.abs(shap_row))
    sv = np.asarray(shap_row)[order]; fn = np.array(feat_names)[order]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [PALETTE[2] if v > 0 else PALETTE[3] for v in sv]
    ax.barh(range(len(sv))[::-1], sv, color=colors)
    ax.set_yticks(range(len(sv))[::-1]); ax.set_yticklabels(fn)
    ax.set_xlabel("SHAP contribution")
    ax.set_title(f"{title}\nbase={base_value:.4f} -> pred={pred_value:.4f}")
    fig.savefig(_p("interpretability", fname)); plt.close(fig)


# --------------------------------------------------------------------------- #
# Comparison + ablation
# --------------------------------------------------------------------------- #
def comparison_heatmap(df, metric_cols, outcome):
    import matplotlib.colors as mcolors
    M_ = df[metric_cols].astype(float).to_numpy()
    norm = np.zeros_like(M_)
    for j in range(M_.shape[1]):
        col = M_[:, j]
        rng = np.nanmax(col) - np.nanmin(col)
        norm[:, j] = 0.5 if rng == 0 else (col - np.nanmin(col)) / rng
    fig, ax = plt.subplots(figsize=(1.1 * len(metric_cols) + 3, 0.5 * len(df) + 2))
    im = ax.imshow(norm, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(metric_cols))); ax.set_xticklabels(metric_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df["Model"].tolist())
    for i in range(len(df)):
        for j in range(len(metric_cols)):
            ax.text(j, i, f"{M_[i,j]:.3g}", ha="center", va="center",
                    color="white" if norm[i, j] < 0.5 else "black", fontsize=6)
    ax.set_title(f"Model comparison (min-max normalized colors) - {outcome}")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.savefig(_p("comparison", f"comparison_heatmap_{outcome}.png"), bbox_inches="tight"); plt.close(fig)


def normalized_metric_bars(df, metric_cols, outcome):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df)); w = 0.8 / len(metric_cols)
    for j, m in enumerate(metric_cols):
        col = df[m].astype(float).to_numpy()
        rng = np.nanmax(col) - np.nanmin(col)
        v = np.zeros_like(col) if rng == 0 else (col - np.nanmin(col)) / rng
        ax.bar(x + j * w, v, w, label=m, color=PALETTE[j % 10])
    ax.set_xticks(x + w * len(metric_cols) / 2); ax.set_xticklabels(df["Model"], rotation=45, ha="right")
    ax.set_ylabel("min-max normalized"); ax.set_title(f"Normalized metric comparison ({outcome})")
    ax.legend(fontsize=7, ncol=3)
    fig.savefig(_p("comparison", f"normalized_bars_{outcome}.png"), bbox_inches="tight"); plt.close(fig)


def ablation_bars(abl_df, metric, outcome):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(abl_df["ablation"], abl_df[metric].astype(float), color=PALETTE[5])
    ax.set_xticklabels(abl_df["ablation"], rotation=45, ha="right")
    ax.set_ylabel(metric); ax.set_title(f"Ablation: {metric} ({outcome})")
    fig.savefig(_p("ablation", f"ablation_{metric}_{outcome}.png"), bbox_inches="tight"); plt.close(fig)


def ablation_degradation(abl_df, metric, outcome):
    v = abl_df[metric].astype(float).to_numpy()
    full = v[abl_df["ablation"].tolist().index("A0_full")] if "A0_full" in abl_df["ablation"].tolist() else v[0]
    deg = (v - full) / (abs(full) + 1e-12) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = [PALETTE[3] if d < 0 else PALETTE[2] for d in deg]
    ax.bar(abl_df["ablation"], deg, color=bar_colors)
    ax.set_ylabel(f"% change in {metric} vs A0"); ax.set_xticklabels(abl_df["ablation"], rotation=45, ha="right")
    ax.set_title(f"Ablation degradation ({outcome})")
    fig.savefig(_p("ablation", f"ablation_degradation_{metric}_{outcome}.png"), bbox_inches="tight"); plt.close(fig)
