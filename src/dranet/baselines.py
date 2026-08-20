"""
Baseline CATE / uplift estimators with a shared interface.

Every estimator implements:
    fit(X, T, Y)              -> self
    predict_cate(X)           -> np.ndarray   (targeting score = estimated CATE)
    predict_outcome(X, T)     -> np.ndarray | None   (P(Y=1|X,T) for supporting
                                                       classification diagnostics)

Base learners are LightGBM (histogram, multi-threaded, float32-friendly) for a fair,
consistent comparison. Cross-fitting (K folds) is used for the DR- and R-learners to
respect Neyman-orthogonality.

Categories (labelled in results):
  CLASSICAL : Naive ATE, Logistic-interaction, Class-Transformation
  META      : S-, T-, X-, R-, DR-Learner
Sources: Kunzel+2019 (S/T/X), Nie&Wager 2021 (R), Kennedy 2023 (DR),
Jaskowski&Jaroszewicz 2012 / transformed-outcome (Class-Transformation).
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from . import config as C


def _clf(params):
    return LGBMClassifier(**{**params, "objective": "binary"})


def _reg(params):
    return LGBMRegressor(**{**params, "objective": "regression"})


# --------------------------------------------------------------------------- #
class NaiveATE:
    """Reference: constant CATE = empirical ATE (treated-rate - control-rate)."""
    category = "CLASSICAL"

    def fit(self, X, T, Y):
        T = np.asarray(T); Y = np.asarray(Y, np.float64)
        self.ate_ = float(Y[T == 1].mean() - Y[T == 0].mean())
        self.p1_ = float(Y[T == 1].mean()); self.p0_ = float(Y[T == 0].mean())
        return self

    def predict_cate(self, X):
        return np.full(len(X), self.ate_, dtype=np.float64)

    def predict_outcome(self, X, T):
        T = np.asarray(T)
        return np.where(T == 1, self.p1_, self.p0_).astype(np.float64)


class LogisticInteraction:
    """Transparent parametric CATE via logistic regression with T and X*T terms."""
    category = "CLASSICAL"

    def fit(self, X, T, Y):
        X = np.asarray(X, np.float64); T = np.asarray(T, np.float64).reshape(-1, 1)
        Xd = np.hstack([X, T, X * T])
        self.model_ = LogisticRegression(max_iter=1000, n_jobs=8)
        self.model_.fit(Xd, np.asarray(Y))
        return self

    def _design(self, X, tval):
        X = np.asarray(X, np.float64)
        T = np.full((len(X), 1), float(tval))
        return np.hstack([X, T, X * T])

    def predict_cate(self, X):
        p1 = self.model_.predict_proba(self._design(X, 1))[:, 1]
        p0 = self.model_.predict_proba(self._design(X, 0))[:, 1]
        return (p1 - p0).astype(np.float64)

    def predict_outcome(self, X, T):
        X = np.asarray(X, np.float64); T = np.asarray(T, np.float64).reshape(-1, 1)
        Xd = np.hstack([X, T, X * T])
        return self.model_.predict_proba(Xd)[:, 1].astype(np.float64)


class SLearner:
    category = "META"

    def __init__(self, params): self.params = params

    def fit(self, X, T, Y):
        X = np.asarray(X, np.float64); T = np.asarray(T, np.float64).reshape(-1, 1)
        self.model_ = _clf(self.params)
        self.model_.fit(np.hstack([X, T]), np.asarray(Y))
        return self

    def predict_cate(self, X):
        X = np.asarray(X, np.float64)
        one = np.ones((len(X), 1)); zero = np.zeros((len(X), 1))
        p1 = self.model_.predict_proba(np.hstack([X, one]))[:, 1]
        p0 = self.model_.predict_proba(np.hstack([X, zero]))[:, 1]
        return (p1 - p0).astype(np.float64)

    def predict_outcome(self, X, T):
        X = np.asarray(X, np.float64); T = np.asarray(T, np.float64).reshape(-1, 1)
        return self.model_.predict_proba(np.hstack([X, T]))[:, 1].astype(np.float64)


class TLearner:
    category = "META"

    def __init__(self, params): self.params = params

    def fit(self, X, T, Y):
        X = np.asarray(X, np.float64); T = np.asarray(T); Y = np.asarray(Y)
        self.m1_ = _clf(self.params).fit(X[T == 1], Y[T == 1])
        self.m0_ = _clf(self.params).fit(X[T == 0], Y[T == 0])
        return self

    def _p(self, m, X):
        return m.predict_proba(np.asarray(X, np.float64))[:, 1]

    def predict_cate(self, X):
        return (self._p(self.m1_, X) - self._p(self.m0_, X)).astype(np.float64)

    def predict_outcome(self, X, T):
        T = np.asarray(T)
        p1 = self._p(self.m1_, X); p0 = self._p(self.m0_, X)
        return np.where(T == 1, p1, p0).astype(np.float64)


class XLearner:
    """Kunzel+2019 X-Learner. Robust under treatment-arm size imbalance."""
    category = "META"

    def __init__(self, params): self.params = params

    def fit(self, X, T, Y):
        X = np.asarray(X, np.float64); T = np.asarray(T); Y = np.asarray(Y, np.float64)
        self.m1_ = _clf(self.params).fit(X[T == 1], Y[T == 1])
        self.m0_ = _clf(self.params).fit(X[T == 0], Y[T == 0])
        # imputed treatment effects
        d1 = Y[T == 1] - self.m0_.predict_proba(X[T == 1])[:, 1]      # treated: Y - mu0
        d0 = self.m1_.predict_proba(X[T == 0])[:, 1] - Y[T == 0]      # control: mu1 - Y
        self.tau1_ = _reg(self.params).fit(X[T == 1], d1)
        self.tau0_ = _reg(self.params).fit(X[T == 0], d0)
        self.p_treat_ = float(T.mean())
        return self

    def predict_cate(self, X):
        X = np.asarray(X, np.float64)
        t1 = self.tau1_.predict(X); t0 = self.tau0_.predict(X)
        g = self.p_treat_                                   # propensity weight (~const)
        return (g * t0 + (1 - g) * t1).astype(np.float64)

    def predict_outcome(self, X, T):
        T = np.asarray(T)
        p1 = self.m1_.predict_proba(np.asarray(X, np.float64))[:, 1]
        p0 = self.m0_.predict_proba(np.asarray(X, np.float64))[:, 1]
        return np.where(T == 1, p1, p0).astype(np.float64)


class ClassTransformation:
    """Transformed-outcome (modified-outcome) uplift for RCT with propensity e.

    Pseudo-outcome Z_i = Y_i * ( T_i/e - (1-T_i)/(1-e) ) has E[Z|X]=CATE.
    A regressor is fit on Z. (Generalises Jaskowski&Jaroszewicz to imbalanced T.)
    """
    category = "CLASSICAL"

    def __init__(self, params, e_hat=None): self.params = params; self.e_hat = e_hat

    def fit(self, X, T, Y, e=None):
        X = np.asarray(X, np.float64); T = np.asarray(T, np.float64); Y = np.asarray(Y, np.float64)
        if e is None:
            e = np.full(len(X), float(T.mean()))
        e = np.clip(np.asarray(e, np.float64), 1e-3, 1 - 1e-3)
        Z = Y * (T / e - (1 - T) / (1 - e))
        self.model_ = _reg(self.params).fit(X, Z)
        return self

    def predict_cate(self, X):
        return self.model_.predict(np.asarray(X, np.float64)).astype(np.float64)

    def predict_outcome(self, X, T):
        return None


# --------------------------------------------------------------------------- #
# Cross-fitted nuisances (shared by R- and DR-learners and by DRA-Net)
# --------------------------------------------------------------------------- #
def cross_fit_nuisances(X, T, Y, params, n_folds=5, seed=C.SEED):
    """Return out-of-fold mu1(x), mu0(x), e(x), m(x)=E[Y|X] via K-fold cross-fitting."""
    X = np.asarray(X, np.float64); T = np.asarray(T); Y = np.asarray(Y, np.float64)
    n = len(X)
    mu1 = np.zeros(n); mu0 = np.zeros(n); ehat = np.zeros(n); mhat = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        Ttr = T[tr]
        m1 = _clf(params).fit(X[tr][Ttr == 1], Y[tr][Ttr == 1])
        m0 = _clf(params).fit(X[tr][Ttr == 0], Y[tr][Ttr == 0])
        eh = _clf(params).fit(X[tr], Ttr)
        mh = _clf(params).fit(X[tr], Y[tr])
        mu1[te] = m1.predict_proba(X[te])[:, 1]
        mu0[te] = m0.predict_proba(X[te])[:, 1]
        ehat[te] = np.clip(eh.predict_proba(X[te])[:, 1], 1e-3, 1 - 1e-3)
        mhat[te] = mh.predict_proba(X[te])[:, 1]
    return mu1, mu0, ehat, mhat


def fit_full_nuisances(X, T, Y, params):
    """Fit nuisance models on all data (for prediction on val/test)."""
    X = np.asarray(X, np.float64); T = np.asarray(T); Y = np.asarray(Y, np.float64)
    m1 = _clf(params).fit(X[T == 1], Y[T == 1])
    m0 = _clf(params).fit(X[T == 0], Y[T == 0])
    eh = _clf(params).fit(X, T)
    mh = _clf(params).fit(X, Y)
    return {"m1": m1, "m0": m0, "e": eh, "m": mh}


class RLearner:
    """Nie & Wager (2021). Residual-on-residual weighted regression of CATE."""
    category = "META"

    def __init__(self, params, n_folds=5): self.params = params; self.n_folds = n_folds

    def fit(self, X, T, Y):
        X = np.asarray(X, np.float64); T = np.asarray(T, np.float64); Y = np.asarray(Y, np.float64)
        _, _, ehat, mhat = cross_fit_nuisances(X, T, Y, self.params, self.n_folds)
        t_res = T - ehat                     # never 0: T in {0,1}, e in (1e-3, 1-1e-3)
        y_res = Y - mhat
        w = t_res ** 2                       # R-loss weights
        pseudo = y_res / t_res               # R-loss target: minimises sum w (pseudo-tau)^2
        self.model_ = _reg(self.params)
        self.model_.fit(X, pseudo, sample_weight=w)
        return self

    def predict_cate(self, X):
        return self.model_.predict(np.asarray(X, np.float64)).astype(np.float64)

    def predict_outcome(self, X, T):
        return None


class DRLearner:
    """Kennedy (2023) doubly-robust learner. AIPW pseudo-outcome regressed on X.

    Also the DRA-Net backbone; here it is exposed as a standalone baseline.
    """
    category = "META"

    def __init__(self, params, n_folds=5): self.params = params; self.n_folds = n_folds

    @staticmethod
    def aipw_pseudo(Y, T, mu1, mu0, e):
        e = np.clip(e, 1e-3, 1 - 1e-3)
        return (mu1 - mu0
                + T * (Y - mu1) / e
                - (1 - T) * (Y - mu0) / (1 - e))

    def fit(self, X, T, Y):
        X = np.asarray(X, np.float64); T = np.asarray(T, np.float64); Y = np.asarray(Y, np.float64)
        mu1, mu0, ehat, _ = cross_fit_nuisances(X, T, Y, self.params, self.n_folds)
        phi = self.aipw_pseudo(Y, T, mu1, mu0, ehat)
        self.model_ = _reg(self.params).fit(X, phi)
        # keep full nuisances for outcome prediction on new data
        self.nuis_ = fit_full_nuisances(X, T, Y, self.params)
        return self

    def predict_cate(self, X):
        return self.model_.predict(np.asarray(X, np.float64)).astype(np.float64)

    def predict_outcome(self, X, T):
        X = np.asarray(X, np.float64); T = np.asarray(T)
        p1 = self.nuis_["m1"].predict_proba(X)[:, 1]
        p0 = self.nuis_["m0"].predict_proba(X)[:, 1]
        return np.where(T == 1, p1, p0).astype(np.float64)
