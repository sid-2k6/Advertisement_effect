"""
Central configuration for the DRA-Net research pipeline.

Research: "Causal Inference-Driven Interpretable Visualization for Advertising
Effect Analysis" on the Criteo Uplift v2.1 dataset (13,979,592 rows).

All paths, seeds, feature lists, split fractions, and the practically-meaningful
effect-threshold grids (c) and reliability-threshold grids (gamma) live here so that
every module and the notebook share one source of truth.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_CSV_GZ = os.path.join(ROOT, "criteo_full.csv.gz")
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")
REPORT_DIR = os.path.join(ROOT, "REPORT")

# Sub-folders required by the deliverable spec (Part 20 of the brief)
PLOT_SUBDIRS = [
    "causal", "propensity", "cate", "uplift", "qini", "reliability",
    "policy", "interpretability", "ablation", "comparison",
]
MODEL_SUBDIRS = ["dranet", "propensity", "preprocessing", "calibration"]

# --------------------------------------------------------------------------- #
# Data schema (verified on the actual file)
# --------------------------------------------------------------------------- #
FEATURES: List[str] = [f"f{i}" for i in range(12)]        # f0..f11
TREATMENT = "treatment"
OUTCOMES = ["visit", "conversion"]
POST_TREATMENT = ["exposure"]                              # mediator/collider -> DROP
PRIMARY_OUTCOME = "visit"        # Experiment A
SECONDARY_OUTCOME = "conversion"  # Experiment B (stress test)

N_TOTAL = 13_979_592

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Split fractions (exact target counts from the brief)
# --------------------------------------------------------------------------- #
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SPLIT_TARGET = {"train": 9_785_714, "val": 2_096_939, "test": 2_096_939}


@dataclass
class RunConfig:
    """Controls compute scale. Test/validation are ALWAYS the full locked splits.

    Only *training* of expensive models is subsampled, and this is documented in
    the results (`train_n`). LightGBM tree/meta learners are trained on the full
    training split by default; forests and neural nets use a documented subsample.
    """
    seed: int = SEED
    fast: bool = False                     # fast=True -> tiny run for smoke tests

    # Training-set caps (None => use full training split).
    # 2,000,000 is a documented, fair, identical training budget for every CATE/meta
    # learner and the DRA-Net nuisances. Validation (2,096,939) and TEST (2,096,939)
    # always remain the FULL locked splits. For 12-feature LightGBM CATE on Criteo the
    # learning curve is flat well before this size, so the subsample is not a limitation
    # on the core comparison while keeping K-fold cross-fitting tractable.
    tree_train_n: int | None = 1_000_000   # S/T/X/R/DR/logistic/class-transform + DRA-Net nuisances
    forest_train_n: int = 300_000          # Causal Forest (EconML) + Uplift RF (causalml)
    neural_train_n: int = 400_000          # TARNet/CFRNet/DragonNet/DESCN/CHAUN
    shap_background_n: int = 2_000          # SHAP background sample
    shap_explain_n: int = 20_000           # SHAP explained sample

    # Cross-fitting folds for nuisance estimation (AIPW / DR-Learner)
    n_folds: int = 3

    # LightGBM defaults (histogram, multi-threaded, early stopping)
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_jobs": 8,
        "verbose": -1,
    })
    lgbm_reg_params: dict = field(default_factory=lambda: {
        "objective": "regression",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_jobs": 8,
        "verbose": -1,
    })

    # -------- DRA-Net decision parameters -------- #
    # c = minimum practically-meaningful ABSOLUTE treatment effect (in outcome units,
    # i.e. probability of visit/conversion). c grid is swept; the operating c is
    # selected on validation. c=0 recovers "any positive effect".
    c_grid: List[float] = field(default_factory=lambda: [0.0, 0.005, 0.01, 0.02, 0.03, 0.05])
    # gamma = reliability threshold on r(x)=P(tau>c). Swept; operating gamma selected on validation.
    gamma_grid: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9])

    # Conformal miscoverage level (1-alpha nominal coverage for CATE intervals)
    alpha: float = 0.1                      # 90% intervals
    # Number of GATES / calibration bins
    n_gates_bins: int = 20
    # Targeting fractions of interest
    target_fracs: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.30])
    # Bootstrap replicates for significance
    n_bootstrap: int = 150
    n_folds_default: int = 3


def ensure_dirs() -> None:
    for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)
    for d in PLOT_SUBDIRS:
        os.makedirs(os.path.join(PLOTS_DIR, d), exist_ok=True)
    for d in MODEL_SUBDIRS:
        os.makedirs(os.path.join(MODELS_DIR, d), exist_ok=True)


def split_path(split: str) -> str:
    return os.path.join(DATA_DIR, f"criteo_{split}.parquet")
