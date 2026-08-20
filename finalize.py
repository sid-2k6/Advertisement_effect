"""Assemble the master comparison table, the architecture figure, and complete the
models/ directory structure (dranet/propensity/preprocessing/calibration)."""
import json, os, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from dranet import config as C

C.ensure_dirs()

# ------------------------------------------------------------------ master table
MASTER_COLS = ["Model", "Outcome", "Category", "ATE", "CATE_Mean", "CATE_Std",
    "AUUC", "Qini", "Uplift@5%", "Uplift@10%", "Uplift@20%", "Uplift@30%",
    "PolicyValue", "PR_AUC", "ROC_AUC", "F1", "Brier", "CATE_CalErr",
    "Coverage", "IntervalWidth", "Pos_CATE_%", "Neg_CATE_%",
    "ReliablePos_%", "ReliableNeg_%", "Uncertain_%", "Runtime_s", "Memory_MB"]
frames = []
for oc in ["visit", "conversion"]:
    p = os.path.join(C.RESULTS_DIR, f"comparison_{oc}.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        for col in MASTER_COLS:
            if col not in df.columns:
                df[col] = np.nan
        frames.append(df[MASTER_COLS])
master = pd.concat(frames, ignore_index=True)
# order rows by canonical model order within each outcome
order = ["Naive", "LogisticInteraction", "S-Learner", "T-Learner", "X-Learner", "R-Learner",
         "DR-Learner", "CausalForest", "UpliftRandomForest", "ClassTransformation",
         "TARNet", "CFRNet", "DragonNet", "DESCN-style", "CHAUN-attn", "DRA-Net"]
master["__o"] = master["Model"].apply(lambda m: order.index(m) if m in order else 99)
master = master.sort_values(["Outcome", "__o"]).drop(columns="__o").reset_index(drop=True)
master.to_csv(os.path.join(C.RESULTS_DIR, "comparison.csv"), index=False)
print("wrote results/comparison.csv", master.shape)

# combined ablation
abl = []
for oc in ["visit", "conversion"]:
    p = os.path.join(C.RESULTS_DIR, f"ablation_{oc}.csv")
    if os.path.exists(p):
        d = pd.read_csv(p); d.insert(0, "Outcome", oc); abl.append(d)
if abl:
    pd.concat(abl, ignore_index=True).to_csv(os.path.join(C.RESULTS_DIR, "ablation.csv"), index=False)

# unify metric csv names (drop outcome suffix -> primary visit copies for the spec names)
for base in ["causal_metrics", "uplift_metrics", "policy_metrics", "reliability_metrics",
             "test_predictions", "cate_predictions"]:
    src = os.path.join(C.RESULTS_DIR, f"{base}_visit.csv")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(C.RESULTS_DIR, f"{base}.csv"))

# ------------------------------------------------------------------ model dirs
os.makedirs(os.path.join(C.MODELS_DIR, "calibration"), exist_ok=True)
os.makedirs(os.path.join(C.MODELS_DIR, "preprocessing"), exist_ok=True)
iso = os.path.join(C.MODELS_DIR, "dranet", "isotonic_visit.joblib")
if os.path.exists(iso):
    shutil.copy(iso, os.path.join(C.MODELS_DIR, "calibration", "isotonic_gates_visit.joblib"))
# preprocessing metadata (LightGBM backbones need no scaler; features used as-is, exposure dropped)
with open(os.path.join(C.MODELS_DIR, "preprocessing", "preprocessing.json"), "w") as fh:
    json.dump({"features": C.FEATURES, "dropped_post_treatment": C.POST_TREATMENT,
               "dtype": "float32", "scaler": "none for tree/meta learners; per-model StandardScaler "
               "fit-on-train inside neural models", "treatment": C.TREATMENT,
               "outcomes": C.OUTCOMES, "seed": C.SEED,
               "split_counts": C.SPLIT_TARGET}, fh, indent=2)
print("model dirs completed")

# ------------------------------------------------------------------ architecture figure
fig, ax = plt.subplots(figsize=(15, 9)); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)


def box(x, y, w, h, text, fc, tc="black", fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                 linewidth=1.4, edgecolor="#333", facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def arrow(x1, y1, x2, y2, color="#333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                 lw=1.6, color=color))


std = "#D6EAF8"; nov = "#F9E79F"; out = "#D5F5E3"
box(2, 86, 96, 10, "Criteo Uplift v2.1  -  13,979,592 users  |  f0..f11, treatment (85% treated, randomized),\nvisit / conversion  |  exposure DROPPED (post-treatment mediator)", "#FADBD8", fs=10)
arrow(50, 86, 50, 82)
box(2, 74, 30, 8, "Stage 0-1  Preprocessing +\nCausal identification (RCT):\ntau(x)=E[Y(1)-Y(0)|X]", std)
box(36, 74, 30, 8, "Stage 2  Propensity e(x)\n(DIAGNOSTICS/DR only -\nnot for identification)", std)
box(70, 74, 28, 8, "Stage 3  Cross-fitted\nnuisances mu1,mu0,e (K-fold)", std)
arrow(17, 74, 17, 66); arrow(51, 74, 51, 66); arrow(84, 74, 84, 66)
box(2, 58, 96, 8, "Stage 4  DR backbone: AIPW pseudo-outcome  phi = mu1-mu0 + T(Y-mu1)/e - (1-T)(Y-mu0)/(1-e)  ->  tau_hat(x)   (backbone selectable on validation)", std, fs=10)
arrow(50, 58, 50, 54)
box(14, 40, 72, 12, "Stage 5  * NOVEL * Decision-Reliability head\n"
    "(a) heterogeneous uncertainty s(x)=sqrt(E[(phi-tau_hat)^2|x])\n"
    "(b) isotonic GATES calibration -> tau_tilde(x)\n"
    "(c) RELIABILITY  r(x) = P(tau(x) > c) = Phi( (tau_tilde(x) - c) / (kappa*s(x)) )\n"
    "c*, kappa*, gamma* SELECTED ON VALIDATION", nov, fs=10)
arrow(30, 40, 20, 30); arrow(50, 40, 50, 30); arrow(70, 40, 80, 30)
box(2, 20, 30, 9, "Stage 6  Interpretability\nSHAP of tau_hat (effect)\nvs SHAP of r(x) (confidence)", nov, fs=9)
box(36, 20, 30, 9, "Stage 7  Policy\npi(x)=1{ r(x) >= gamma }\nIPW policy value", nov, fs=9)
box(70, 20, 28, 9, "Stage 8  Visualization\nReliability-Effect plane,\nreliability-aware / UD-Qini", nov, fs=9)
arrow(17, 20, 17, 12); arrow(51, 20, 51, 12); arrow(84, 20, 84, 12)
box(2, 2, 96, 8, "Stage 9  Advertising decision support: WHO to target & WHY, with calibrated confidence  "
    "(Reliable/Uncertain Persuadables, Sleeping Dogs, Sure Things, Lost Causes)  -  Streamlit dashboard", out, fs=10)
ax.text(50, 99, "DRA-Net: Decision-Reliability Attribution framework  (blue = standard causal ML, yellow = novel reliability layer)",
        ha="center", fontsize=12, fontweight="bold")
fig.savefig(os.path.join(C.PLOTS_DIR, "comparison", "dranet_architecture.png"), dpi=160, bbox_inches="tight")
print("wrote architecture figure")
