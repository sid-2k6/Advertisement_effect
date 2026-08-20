"""Generate the executable research notebook DRA_Net_Research.ipynb (sections 0-32).

Executable cells load the checkpointed artifacts/results and render tables + figures in
seconds. Heavy model fitting is shown and callable but gated by RUN_HEAVY (default False);
the actual results were produced by the staged pipeline (run_pipeline.py)."""
import json, os

cells = []


def md(text): cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepdims=True) if hasattr(text, "splitlines") else text})
def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in text.strip("\n").split("\n")]})
def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": [l + "\n" for l in src.strip("\n").split("\n")]})


md("""# Causal Inference-Driven Interpretable Visualization for Advertising Effect Analysis
## DRA-Net (Decision-Reliability Attribution Network) — executable research notebook

This notebook is **executable in seconds**: it loads the checkpointed results/artifacts produced by
the staged pipeline (`run_pipeline.py`) and renders every table and figure. Heavy model fitting is
shown and callable but gated by `RUN_HEAVY` (default `False`). Validation/test are the full locked
splits; the test split was scored once with all selection frozen on validation.""")

code("""import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from IPython.display import Image, display
from dranet import config as C
RUN_HEAVY = False   # set True to refit (uses run_experiment stages; hours on CPU)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 60)
def show(path, w=900):
    display(Image(path, width=w)) if os.path.exists(path) else print("missing:", path)
def load_metrics(oc): return json.load(open(f"results/metrics_{oc}.json"))
print("features:", C.FEATURES); print("splits:", C.SPLIT_TARGET)""")

# 0
md("""## 0. Research Objective
Estimate, explain, and **act on** the incremental effect of advertising on the Criteo Uplift v2.1 RCT
(13,979,592 users). Proposed contribution: a reliability-aware decision quantity **r(x)=P(τ(x)>c)** that
drives targeting, attribution, visualization, and segmentation — tested vs point-CATE ranking, LCB
ranking, classical/meta/strong-modern/recent baselines, and a proposed network.""")

# 1
md("""## 1. Literature / Methodological Context
See `RESEARCH_PLAN.md` (Parts A–B): 41 works across causal inference, uplift/advertising, interpretable
causal ML, causal visual analytics, CATE uncertainty, and policy learning. Key prior art fence:
UpliftBench (arXiv 2604.06123) already does S/T/X + Causal Forest + SHAP + persuadables on this exact
dataset — so that *combination* is explicitly not our novelty. Our contribution is the reliability
estimand r(x)=P(τ>c) as a unified policy/attribution/visualization object.""")

# 2
md("## 2. Environment and Dependencies")
code("""import numpy, pandas, sklearn, lightgbm, scipy, shap, econml, torch
print("numpy", numpy.__version__, "| lightgbm", lightgbm.__version__, "| torch", torch.__version__)
print("econml", econml.__version__, "| shap", shap.__version__)""")

# 3
md("## 3. Configuration (single source of truth)")
code("""cfg = C.RunConfig()
print("tree_train_n =", cfg.tree_train_n, "| n_folds =", cfg.n_folds, "| seed =", cfg.seed)
print("c_grid =", cfg.c_grid, "| gamma_grid =", cfg.gamma_grid, "| alpha =", cfg.alpha)
print("VAL/TEST are FULL locked splits:", C.SPLIT_TARGET)""")

# 4
md("""## 4. Data Loading
Full gzip loaded once, `exposure` dropped (post-treatment mediator), float32, split 70/15/15 to
parquet. (Executed once via `dranet.data.prepare_and_save()`.)""")
code("""from dranet import data
if RUN_HEAVY and not os.path.exists(C.split_path("train")):
    data.prepare_and_save()
rep = json.load(open(os.path.join(C.DATA_DIR, "data_validation_report.json")))
print("rows:", rep["n_rows"], "| split_counts:", rep["split_counts"])
print("treatment_mean:", round(rep["treatment_mean"], 4))
for y in C.OUTCOMES: print(f"  {y}: rate={rep[f'{y}_mean']:.5f}  naive_ATE={rep[f'{y}_naive_ate']:+.6f}")""")

# 5
md("""## 5. Data Validation
Missing/duplicate/balance/cardinality + train/val/test consistency. Note: ~1.66M identical feature
rows are an inherent property of Criteo's coarse anonymized features (documented threat to validity),
not a bug — duplicates are NOT dropped (would distort the experiment).""")
code("""print("missing (nonzero):", {k:v for k,v in rep["missing_per_col"].items() if v>0} or "none")
print("duplicate feature-rows:", rep["n_duplicate_rows"])
print("split marginals:"); [print("  ", s, m) for s,m in rep["split_marginals"].items()]""")

# 6
md("## 6. EDA — outcome rates & treatment imbalance")
code("""eda = pd.DataFrame({y: {"rate": rep[f"{y}_mean"], "rate_treated": rep[f"{y}_rate_treated"],
        "rate_control": rep[f"{y}_rate_control"], "naive_ATE": rep[f"{y}_naive_ate"]} for y in C.OUTCOMES}).T
display(eda.round(6))
print("Treatment is imbalanced (85% treated) but RANDOMIZED; conversion is extremely rare (0.29%).")""")

# 7
md("## 7. Treatment / Outcome Diagnostics")
code("""for oc in ["visit","conversion"]:
    m = load_metrics(oc)["propensity"]
    print(f"[{oc}] propensity AUC={m['propensity_auc']:.4f} (≈0.5 ⇒ randomized) | "
          f"common support={m['frac_in_common_support']:.3f} | max|SMD|={m['max_abs_smd_before']:.4f}")""")

# 8
md("""## 8. Propensity Score Analysis
e(x)=P(T=1|X) for **diagnostics / doubly-robust** use only — NOT for identification (Criteo is
randomized). Stabilized IPTW, ESS, overlap reported.""")
code("""for oc in ["visit","conversion"]:
    show(f"plots/propensity/propensity_{oc}.png")
    show(f"plots/causal/balance_{oc}.png")""")

# 9
md("## 9. Train/Validation/Test Split (locked)")
code("""print("Split counts (exact):", C.SPLIT_TARGET, "-> sums to", sum(C.SPLIT_TARGET.values()))
print("Training subsample for fair comparison:", cfg.tree_train_n, "(val/test remain FULL & locked)")""")

# 10-16 baselines
md("""## 10–16. Baselines: Naive, Logistic-interaction, S/T/X/R/DR-Learner, Class-Transformation,
Causal Forest, Uplift RF
Full implementations in `dranet.baselines` and `dranet.forests`. Fitted via
`run_pipeline.py <outcome> baselines|forests`. Below: their test metrics (loaded).""")
code("""comp = pd.read_csv("results/comparison.csv")
show_cols = ["Model","Category","Qini","AUUC","Uplift@10%","Uplift@20%","PolicyValue","PR_AUC","CATE_CalErr"]
display(comp[comp.Outcome=="visit"][show_cols].round(4).reset_index(drop=True))""")

# 17
md("""## 17. Recent SOTA / strong-modern models: TARNet, CFRNet, DragonNet, DESCN-style, CHAUN-attn
PyTorch reimplementations in `dranet.neural` (trained on documented subsample, evaluated on full
locked test). Labelled STRONG_MODERN / RECENT — not indiscriminately "SOTA".""")
code("""display(comp[(comp.Outcome=="visit") & (comp.Category.isin(["STRONG_MODERN","RECENT"]))]
        [show_cols].round(4).reset_index(drop=True))""")

# 18
md("""## 18. Proposed Model — DRA-Net
`r(x)=P(τ(x)>c)=Φ((τ̃(x)−c)/(κ s(x)))`. Backbone: cross-fitted AIPW/DR-Learner; uncertainty:
heterogeneous conditional dispersion; calibration: isotonic GATES; c,κ,γ selected on validation.""")
code("""show("plots/comparison/dranet_architecture.png", w=1100)
for oc in ["visit","conversion"]:
    s = load_metrics(oc)["dranet_selection"]
    print(f"[{oc}] c*={s['c_star']} kappa*={s['kappa_star']:.4f} gamma*={s['gamma_star']} "
          f"| val AUUC: reliability={s['val_auuc_reliability']:.5f} vs CATE={s['val_auuc_cate']:.5f}")""")

# 19
md("## 19. Hyperparameter tuning (LightGBM defaults; c,κ,γ grids)")
code("""print("LightGBM:", cfg.lgbm_params)
print("Reliability grids: c_grid=", cfg.c_grid, "gamma_grid=", cfg.gamma_grid)""")

# 20
md("""## 20. Validation-based model selection
c, κ, γ and the DRA-Net backbone are chosen on validation (never test). Backbone-generality analysis
(`dranet.backbone_analysis`) selects the val-best backbone for DRA-Net*.""")
code("""for oc in ["visit","conversion"]:
    bg = json.load(open(f"results/backbone_generality_{oc}.json"))
    print(f"[{oc}] validation-selected backbone = {bg['selected_backbone']} | "
          f"test Qini: r={bg['test_qini_r']:.4f} vs tau={bg['test_qini_tau']:.4f}")""")

# 21
md("## 21. Causal Evaluation — ATE / CATE")
code("""display(comp[comp.Outcome=="visit"][["Model","ATE","CATE_Mean","CATE_Std","CATE_CalErr","Pos_CATE_%","Neg_CATE_%"]]
        .round(5).reset_index(drop=True))""")

# 22
md("## 22. Prediction Diagnostics (supporting only — accuracy is NOT the headline)")
code("""display(comp[comp.Outcome=="visit"][["Model","PR_AUC","ROC_AUC","F1","Brier"]].round(4).reset_index(drop=True))
print("Criteo is highly imbalanced; PR-AUC/Brier are supporting diagnostics, not the primary result.")""")

# 23
md("""## 23. Interpretability — effect vs confidence (two distinct questions)
SHAP of τ̂(x) = *why the ad helps*; SHAP of the reliability score = *why we are confident*.""")
code("""for oc in ["visit","conversion"]:
    print(oc); show(f"plots/interpretability/effect_vs_reliability_shap_{oc}.png")
sh = json.load(open("results/metrics_visit.json"))  # surrogate fidelity noted in shap_summary
try:
    ss = json.load(open("artifacts/visit/shap_summary.json"))
    print("reliability-score surrogate R2 (visit):", round(ss["surrogate_r2"],3))
except Exception: pass""")

# 24
md("## 24. Visualization — CATE, Qini, uplift, policy")
code("""oc="visit"
for p in [f"plots/cate/cate_distributions_{oc}.png", f"plots/qini/qini_curves_{oc}.png",
          f"plots/uplift/uplift_at_k_{oc}.png", f"plots/policy/targeting_curves_{oc}.png"]:
    show(p)""")

# 25
md("## 25. Ablation Study (A0–A10)")
code("""abl = pd.read_csv("results/ablation_visit.csv")
display(abl[["ablation","Qini","AUUC","Uplift@20%","PolicyValue","TargetFrac"]].round(4))
show("plots/ablation/ablation_Uplift@20%_visit.png")""")

# 26
md("## 26. Robustness / Statistical Significance (bootstrap, paired)")
code("""for oc in ["visit","conversion"]:
    r = load_metrics(oc); bb = r["best_baseline_by_qini"]
    d = r["significance"]["diff_vs_first"]
    print(f"\\n[{oc}] reference=DRA-Net_r ; diff = (other - DRA-Net_r). best baseline={bb}")
    for other in [bb, "DRA-Net_CATE"]:
        q = d.get(other,{}).get("qini",{})
        if q: print(f"  {other}: Δqini={q['mean_diff']:+.5f} 95%CI[{q['lo95']:+.5f},{q['hi95']:+.5f}] sig={q['significant_95']}")""")

# 27
md("""## 27. Final Test Evaluation (once, frozen) + the decisive comparison
Does r(x) beat point-CATE and LCB ranking on the SAME backbone?""")
code("""for oc in ["visit","conversion"]:
    dec = load_metrics(oc)["decisive_comparison"]
    print(f"\\n[{oc}] targeting fraction={dec['dranet_targeting_fraction']:.3f}")
    for rule in ["CATE_tau_hat","LCB","P(tau>c)=r"]:
        v = dec[rule]; print(f"  {rule:14s} Qini={v['qini_coef_normalized']:+.4f} u@20={v['uplift@20']:.5f}")""")

# 28
md("## 28. Master Comparison Table (both outcomes; DRA-Net family included)")
code("""cols = ["Model","Outcome","Category","Qini","AUUC","Uplift@20%","PolicyValue","PR_AUC",
        "Coverage","IntervalWidth","ReliablePos_%","Uncertain_%"]
display(comp[cols].round(4))""")

# 29
md("## 29. Save Best Model (frozen artifacts)")
code("""import glob
print("models/:", [os.path.relpath(p, ".") for p in glob.glob("models/**/*", recursive=True) if os.path.isfile(p)])
print("metadata:", json.load(open("models/dranet/metadata_visit.json")))""")

# 30
md("## 30. Export Results")
code("""print("results/:", sorted(os.listdir("results")))""")

# 31
md("""## 31. Streamlit Dashboard Preparation
`PYTHONPATH=src streamlit run app/streamlit_app.py` — 7 pages: overview, causal diagnostics, CATE,
targeting, model comparison, individual user explanation (live), interpretation summary.""")
code("""print(open("app/streamlit_app.py").read()[:600], "...")""")

# 32
md("""## 32. Final Research Summary
See `REPORT/RESEARCH_SUMMARY.md`. Headline (honest): reliability ranking r(x)=P(τ>c) **significantly
beats point-CATE and LCB ranking on the same backbone**, with the largest gains on noisy estimators
and no-harm behaviour on strong ones; DRA-Net on the DR backbone does **not** beat the best neural
baseline in absolute Qini (reported transparently), while DRA-Net* (validation-selected backbone) is
on par with it and adds calibrated confidence, segmentation, and dual attribution.""")
code("""print(open("REPORT/RESEARCH_SUMMARY.md").read()[:1500])""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
      "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
with open("DRA_Net_Research.ipynb", "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote DRA_Net_Research.ipynb with", len(cells), "cells")
