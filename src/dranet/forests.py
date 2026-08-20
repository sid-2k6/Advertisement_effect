"""
Forest-based CATE / uplift baselines with the shared model interface.

  * CausalForestWrapper : EconML CausalForestDML (Wager & Athey 2018; honest forest
    with native pointwise confidence intervals). Used both as a strong baseline AND
    as an alternative uncertainty source for ablation A10 (different estimator).
  * UpliftRFWrapper     : causalml UpliftRandomForestClassifier (Rzepakowski &
    Jaroszewicz style split criteria) - a direct uplift-tree baseline.

Both are trained on a documented subsample (forest_train_n) for tractability; they
are STILL evaluated on the full locked val/test splits.
"""
from __future__ import annotations
import warnings
import numpy as np

from . import config as C

warnings.filterwarnings("ignore")


class CausalForestWrapper:
    category = "STRONG_MODERN"
    name = "CausalForest"

    def __init__(self, cfg: C.RunConfig, n_estimators: int = 300):
        self.cfg = cfg
        self.n_estimators = n_estimators
        self.model_ = None

    def fit(self, X, T, Y):
        from econml.dml import CausalForestDML
        from lightgbm import LGBMClassifier, LGBMRegressor
        X = np.asarray(X, np.float64); T = np.asarray(T); Y = np.asarray(Y, np.float64)
        self.model_ = CausalForestDML(
            model_y=LGBMRegressor(n_estimators=200, num_leaves=31, n_jobs=8, verbose=-1),
            model_t=LGBMClassifier(n_estimators=200, num_leaves=31, n_jobs=8, verbose=-1),
            discrete_treatment=True,
            n_estimators=self.n_estimators,
            min_samples_leaf=50,
            max_depth=None,
            random_state=self.cfg.seed,
            cv=3,
            n_jobs=8,
        )
        self.model_.fit(Y, T, X=X)
        return self

    def predict_cate(self, X, batch=200000):
        X = np.asarray(X, np.float64); out = []
        for i in range(0, len(X), batch):
            out.append(self.model_.effect(X[i:i + batch]).astype(np.float64).ravel())
        return np.concatenate(out)

    def predict_interval(self, X, alpha=0.1, batch=100000):
        """Batched to avoid the large memory footprint of effect_interval on ~2M rows."""
        X = np.asarray(X, np.float64); los = []; his = []
        for i in range(0, len(X), batch):
            lo, hi = self.model_.effect_interval(X[i:i + batch], alpha=alpha)
            los.append(lo.ravel().astype(np.float64)); his.append(hi.ravel().astype(np.float64))
        return np.concatenate(los), np.concatenate(his)

    def predict_outcome(self, X, T):
        return None


class UpliftRFWrapper:
    category = "STRONG_MODERN"
    name = "UpliftRandomForest"

    def __init__(self, cfg: C.RunConfig, n_estimators: int = 100, max_depth: int = 8):
        self.cfg = cfg
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_ = None

    def fit(self, X, T, Y):
        from causalml.inference.tree import UpliftRandomForestClassifier
        X = np.asarray(X, np.float64)
        # causalml expects string treatment labels
        treat = np.where(np.asarray(T) == 1, "treatment", "control").astype(object)
        self.model_ = UpliftRandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=200,
            control_name="control",
            evaluationFunction="KL",
            random_state=self.cfg.seed,
            n_jobs=8,
        )
        self.model_.fit(X, treat, np.asarray(Y))
        return self

    def predict_cate(self, X, batch=200000):
        X = np.asarray(X, np.float64); out = []
        for i in range(0, len(X), batch):
            pred = np.asarray(self.model_.predict(X[i:i + batch]), np.float64)
            out.append(pred.ravel() if pred.ndim > 1 else pred)
        return np.concatenate(out)

    def predict_outcome(self, X, T):
        return None
