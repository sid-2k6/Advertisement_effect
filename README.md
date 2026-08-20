# Causal Inference-Driven Interpretable Visualization for Advertising Effect Analysis
### DRA-Net — Decision-Reliability Attribution Network (Criteo Uplift v2.1)

A reliability-aware causal decision framework. Instead of ranking users by the point CATE `τ̂(x)`,
DRA-Net estimates **`r(x) = P(τ(x) > c)`** — the probability that advertising helps a user by more
than a practically-meaningful threshold `c` — and uses this single quantity for **targeting,
treatment recommendation, causal attribution, visualization, and segmentation**.

> **Honest positioning.** Standard components (S/T/X/R/DR-Learners, Causal Forest, Uplift RF, class
> transformation, propensity/IPTW, conformal ideas, CATE uncertainty, SHAP, Qini/AUUC, policy value,
> persuadable/sleeping-dog segmentation) are **not** claimed as novel. The contribution is the
> reliability estimand `r(x)=P(τ>c)` as a *unified* policy / attribution / visualization object,
> evaluated experimentally at 14M-row scale.

## Key results (locked test; see `REPORT/RESEARCH_SUMMARY.md`)
- **Core hypothesis confirmed:** on the same backbone, ranking by `r(x)` **significantly beats** point-CATE
  ranking and lower-confidence-bound ranking (paired bootstrap 95% CI excludes 0). LCB ranking is
  *worse* than CATE — reliability ≠ conservatism.
- **Backbone-generality:** the reliability layer substantially improves *noisy* estimators
  (DR-Learner Qini +0.018 visit / +0.045 conversion) and *gracefully reduces to CATE ranking* on strong
  ones (no harm). Validation selects a strong backbone (DRA-Net* ≈ best baseline).
- **Transparent limitation:** DRA-Net on the DR backbone does **not** top the absolute Qini leaderboard
  (TARNet is higher, significantly) — reported, not hidden.
- Reliability validated: group coverage ≈ nominal; Spearman(r-decile, observed uplift)=0.82.

## Repository layout
```
src/dranet/            core package (config, data, metrics, diagnostics, baselines, forests,
                       neural, dranet, reliability, significance, segmentation, viz,
                       run_experiment [staged], backbone_analysis)
run_pipeline.py        staged runner: python run_pipeline.py <visit|conversion> <stage>
finalize.py            master table + architecture figure + model-dir assembly
make_notebook.py       generates DRA_Net_Research.ipynb
DRA_Net_Research.ipynb executable notebook (sections 0-32; loads checkpointed results)
app/streamlit_app.py   7-page interactive dashboard
RESEARCH_PLAN.md       Parts A-J: literature gap, novelty, architecture, plans
REPORT/RESEARCH_SUMMARY.md   final research summary + title alignment + novelty statement
results/  plots/  models/    saved metrics, figures, frozen artifacts
```

## Reproduce
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt         # torch CPU: --index-url https://download.pytorch.org/whl/cpu
# 1) get data (HuggingFace mirror) -> criteo_full.csv.gz, then build locked splits:
python -c "from dranet import data; data.prepare_and_save()"   # PYTHONPATH=src
# 2) run each experiment stage (checkpointed to artifacts/):
for st in baselines forests neural dranet evaluate; do python run_pipeline.py visit $st; done
python -c "from dranet import config as C, backbone_analysis as BA; BA.run('visit', C.RunConfig())"
python finalize.py
# 3) dashboard
PYTHONPATH=src streamlit run app/streamlit_app.py
```

## Data
Criteo Uplift v2.1 (13,979,592 rows; f0–f11, treatment, visit, conversion, exposure). `exposure` is a
post-treatment mediator and is **dropped**. Split 70/15/15 = 9,785,714 / 2,096,939 / 2,096,939.
Mirror used: HuggingFace `criteo/criteo-uplift`. Data file and `artifacts/` are git-ignored (large).
