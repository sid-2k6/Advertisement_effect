"""
DRA-Net interactive dashboard — the "Interpretable Visualization" component of
"Causal Inference-Driven Interpretable Visualization for Advertising Effect Analysis".

Run:
    cd Advertisement_effect
    PYTHONPATH=src .venv/bin/streamlit run app/streamlit_app.py

Reads saved results/ + plots/ + artifacts/ (no retraining). Page 6 loads the frozen
DRA-Net object for live per-user counterfactual + reliability computation.

DISCLAIMER shown in-app: estimates are causal only insofar as Criteo's randomized design
and the estimator assumptions hold; reliability r(x)=P(tau>c) is a validated DECISION score,
not a guarantee about any individual.
"""
import json, os, sys
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS = os.path.join(ROOT, "results"); PLOTS = os.path.join(ROOT, "plots")
ART = os.path.join(ROOT, "artifacts")

st.set_page_config(page_title="DRA-Net | Advertising Effect Analysis", layout="wide")


@st.cache_data
def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else {}


@st.cache_data
def load_csv(p):
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def img(path, cap=""):
    if os.path.exists(path):
        st.image(path, caption=cap, use_container_width=True)
    else:
        st.info(f"figure not found: {os.path.basename(path)}")


st.sidebar.title("DRA-Net")
st.sidebar.caption("Decision-Reliability Attribution")
outcome = st.sidebar.radio("Outcome (experiment)", ["visit", "conversion"],
                           help="A = visit (primary); B = conversion (secondary stress test)")
page = st.sidebar.radio("Page", [
    "1 · Dataset / experiment overview", "2 · Causal diagnostics", "3 · CATE analysis",
    "4 · Advertising targeting", "5 · Model comparison", "6 · Individual user explanation",
    "7 · Interpretation summary"])
st.sidebar.markdown("---")
st.sidebar.warning("Estimates are causal only under Criteo's randomized design + estimator "
                   "assumptions. r(x)=P(τ>c) is a validated **decision score**, not an individual "
                   "guarantee.")

metrics = load_json(os.path.join(RESULTS, f"metrics_{outcome}.json"))
sel = metrics.get("dranet_selection", {})
rel = metrics.get("reliability", {})
comp = load_csv(os.path.join(RESULTS, "comparison.csv"))
preds = load_csv(os.path.join(RESULTS, f"test_predictions_{outcome}.csv"))

# =============================================================== Page 1
if page.startswith("1"):
    st.title("Criteo Uplift v2.1 — experiment overview")
    meta = metrics.get("meta", {}); prop = metrics.get("propensity", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users (full)", "13,979,592")
    c2.metric("Treated share", "85.0%")
    c3.metric(f"Naive ATE ({outcome})", f"{meta.get('ate_naive_test', float('nan')):+.5f}")
    c4.metric("Test rows (locked)", "2,096,939")
    st.markdown(
        "- **Features** f0–f11 (anonymized). **Treatment** randomized (85% treated). "
        "**`exposure` dropped** (post-treatment mediator).\n"
        "- Split **70/15/15 = 9,785,714 / 2,096,939 / 2,096,939**. Validation used for all "
        "selection; **test scored once**.\n"
        f"- **Primary = visit (4.70%)**, secondary = **conversion (0.29%, extremely rare)**.")
    st.subheader("DRA-Net operating point (selected on validation)")
    c1, c2, c3 = st.columns(3)
    c1.metric("c* (meaningful effect)", f"{sel.get('c_star', float('nan'))}")
    c2.metric("κ* (spread scale)", f"{sel.get('kappa_star', float('nan')):.4f}")
    c3.metric("γ* (reliability threshold)", f"{sel.get('gamma_star', float('nan'))}")

# =============================================================== Page 2
elif page.startswith("2"):
    st.title("Causal diagnostics")
    prop = metrics.get("propensity", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Propensity AUC", f"{prop.get('propensity_auc', float('nan')):.4f}", help="≈0.5 ⇒ randomized")
    c2.metric("Common support", f"{prop.get('frac_in_common_support', float('nan')):.3f}")
    c3.metric("Max |SMD|", f"{prop.get('max_abs_smd_before', float('nan')):.4f}")
    c4.metric("Effective sample size", f"{prop.get('effective_sample_size', float('nan')):,.0f}")
    st.caption("Propensity is used for diagnostics / doubly-robust estimation only — NOT for "
               "identification (assignment is randomized).")
    col1, col2 = st.columns(2)
    with col1: img(os.path.join(PLOTS, "propensity", f"propensity_{outcome}.png"), "Propensity overlap")
    with col2: img(os.path.join(PLOTS, "causal", f"balance_{outcome}.png"), "Covariate balance")

# =============================================================== Page 3
elif page.startswith("3"):
    st.title("CATE analysis")
    if not preds.empty:
        cate = preds["cate"].to_numpy()
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean CATE", f"{cate.mean():+.5f}")
        c2.metric("% positive", f"{(cate>0).mean()*100:.1f}%")
        c3.metric("% negative", f"{(cate<0).mean()*100:.1f}%")
        lo, hi = st.slider("CATE percentile window", 0, 100, (1, 99))
        a, b = np.percentile(cate, [lo, hi])
        st.bar_chart(pd.Series(np.clip(cate, a, b)).sample(min(20000, len(cate)), random_state=0)
                     .value_counts(bins=60).sort_index())
        st.caption(f"CATE window [{a:.4f}, {b:.4f}] (P{lo}–P{hi}).")
    col1, col2 = st.columns(2)
    with col1: img(os.path.join(PLOTS, "cate", f"cate_distributions_{outcome}.png"), "CATE distributions")
    with col2: img(os.path.join(PLOTS, "reliability", f"cate_intervals_{outcome}.png"), "CATE uncertainty intervals")

# =============================================================== Page 4
elif page.startswith("4"):
    st.title("Advertising targeting")
    dec = metrics.get("decisive_comparison", {})
    st.subheader("Realised uplift by targeting rule (locked test)")
    rows = []
    for rule in ["CATE_tau_hat", "LCB", "P(tau>c)=r"]:
        d = dec.get(rule, {})
        rows.append({"ranking": rule, "Qini": d.get("qini_coef_normalized"),
                     "Uplift@5%": d.get("uplift@5"), "Uplift@10%": d.get("uplift@10"),
                     "Uplift@20%": d.get("uplift@20"), "Uplift@30%": d.get("uplift@30")})
    st.dataframe(pd.DataFrame(rows).round(5), use_container_width=True)
    st.caption("Reliability r(x)=P(τ>c) vs point-CATE vs lower-confidence-bound ranking.")
    col1, col2 = st.columns(2)
    with col1: img(os.path.join(PLOTS, "policy", f"targeting_curves_{outcome}.png"), "Policy value & incremental vs targeting fraction")
    with col2: img(os.path.join(PLOTS, "reliability", f"threshold_sensitivity_{outcome}.png"), "Threshold sensitivity (c, γ)")
    img(os.path.join(PLOTS, "qini", f"reliability_aware_qini_{outcome}.png"), "Reliability-aware Qini")

# =============================================================== Page 5
elif page.startswith("5"):
    st.title("Model comparison — baselines vs strong/recent vs DRA-Net")
    if not comp.empty:
        sub = comp[comp["Outcome"] == outcome].copy()
        show = ["Model", "Category", "Qini", "AUUC", "Uplift@10%", "Uplift@20%",
                "PolicyValue", "PR_AUC", "Coverage", "ReliablePos_%"]
        st.dataframe(sub[show].round(5).reset_index(drop=True), use_container_width=True, height=520)
    img(os.path.join(PLOTS, "comparison", f"comparison_heatmap_{outcome}.png"), "Normalized comparison heatmap")
    img(os.path.join(PLOTS, "comparison", "dranet_architecture.png"), "DRA-Net architecture")

# =============================================================== Page 6
elif page.startswith("6"):
    st.title("Individual user explanation")
    try:
        import joblib
        from dranet import config as C, data as D
        net = joblib.load(os.path.join(ART, outcome, "dranet_object.joblib"))
        Xte, Tte, Yte = D.load_split("test", outcome)

        @st.cache_data
        def sample_idx(n, k=3000):
            return np.random.default_rng(0).choice(n, k, replace=False)
        idx = sample_idx(len(Xte))
        i = st.selectbox("Pick a sampled test user", list(range(len(idx))),
                         format_func=lambda k: f"user #{int(idx[k])}")
        x = Xte[idx[i]:idx[i] + 1].astype(float)
        tau = float(net.predict_cate(x)[0]); s = float(net.predict_spread(x)[0])
        r = float(net.predict_reliability(x)[0]); lcb = float(net.predict_lcb(x)[0])
        mu1 = float(net.nuis_["m1"].predict_proba(x)[0, 1]); mu0 = float(net.nuis_["m0"].predict_proba(x)[0, 1])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted CATE τ̂", f"{tau:+.4f}")
        c2.metric("Reliability r=P(τ>c)", f"{r:.3f}")
        c3.metric("P(outcome | treated)", f"{mu1:.4f}")
        c4.metric("P(outcome | control)", f"{mu0:.4f}")
        target = r >= sel.get("gamma_star", 0.5)
        st.success(f"**Recommended action: {'TARGET (show ad)' if target else 'do NOT target'}** "
                   f"— r={r:.3f} vs γ*={sel.get('gamma_star')}, CATE 90% CI ≈ [{lcb:+.4f}, {tau+ (tau-lcb):+.4f}].")
        st.caption("Counterfactual P(outcome) under treatment vs control from the frozen nuisance models; "
                   "reliability uses the DRA-Net conditional-dispersion + calibration.")
    except Exception as e:
        st.error(f"Live per-user view unavailable ({e}). Showing saved local explanations instead.")
    col1, col2 = st.columns(2)
    with col1: img(os.path.join(PLOTS, "interpretability", f"local_reliability_positive_{outcome}.png"), "Why confident (positive-effect user)")
    with col2: img(os.path.join(PLOTS, "interpretability", f"local_effect_negative_{outcome}.png"), "Effect explanation (negative-effect user)")
    img(os.path.join(PLOTS, "interpretability", f"effect_vs_reliability_shap_{outcome}.png"), "What drives the EFFECT vs the CONFIDENCE")

# =============================================================== Page 7
elif page.startswith("7"):
    st.title("Interpretation summary — who, and how confidently")
    seg = metrics.get("segmentation", {})
    bc = seg.get("base_counts", {}); bu = seg.get("base_uplift", {})
    st.subheader("Classical segments (Kane framework)")
    st.dataframe(pd.DataFrame({"segment": list(bc.keys()),
        "count": list(bc.values()),
        "observed_uplift": [bu.get(k) for k in bc]}).round(5), use_container_width=True)
    st.subheader("Reliability layer")
    rc = seg.get("reliability_counts", {}); ru = seg.get("reliability_uplift", {})
    st.dataframe(pd.DataFrame({"segment": list(rc.keys()), "count": list(rc.values()),
        "observed_uplift": [ru.get(k) for k in rc]}).round(5), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Confident persuadables", f"{rel.get('confident_positive_pct', float('nan'))*100:.1f}%")
    c2.metric("Uncertain", f"{rel.get('uncertain_pct', float('nan'))*100:.1f}%")
    c3.metric("r-decile ↔ uplift (Spearman)", f"{rel.get('reliability_monotonicity_spearman', float('nan')):.2f}")
    col1, col2 = st.columns(2)
    with col1: img(os.path.join(PLOTS, "reliability", f"reliability_effect_plane_{outcome}.png"), "Reliability–Effect plane")
    with col2: img(os.path.join(PLOTS, "reliability", f"reliability_decile_uplift_{outcome}.png"), "Observed uplift by reliability vs CATE decile")
