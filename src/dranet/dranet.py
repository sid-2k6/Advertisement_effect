"""
DRA-Net : Decision-Reliability Attribution framework.

Proposed contribution (to be demonstrated experimentally, NOT assumed):
  Instead of ranking users by the point CATE tau_hat(x), DRA-Net estimates the
  probability that the advertising effect exceeds a practically-meaningful
  threshold c:
        r(x) = P( tau(x) > c )
  and uses r(x) as the UNIFIED object for (1) targeting policy, (2) treatment
  recommendation, (3) causal attribution, (4) visualization, (5) segmentation.

Why r(x) can differ from CATE ranking
--------------------------------------
  r(x) = Phi( (tau_tilde(x) - c) / s(x) ). If the uncertainty s(x) were constant,
  r would be a monotone transform of tau_tilde and rank IDENTICALLY to CATE. The
  contribution is meaningful ONLY when s(x) is HETEROGENEOUS: r then down-weights
  high-CATE-but-high-uncertainty users (whose optimistic estimates may regress to
  the mean) and up-weights moderate-CATE-but-reliable users. Whether this improves
  REALISED uplift is an empirical question answered on held-out data.

Pipeline
--------
  Stage 4  DR backbone   : cross-fitted AIPW pseudo-outcome phi -> tau_hat(x)
  Stage 5a Uncertainty   : heterogeneous conditional-dispersion s_raw(x)=sqrt(v(x)),
                           v(x)=E[(phi-tau_hat(x))^2 | x]  (optionally bootstrap var)
  Stage 5b Calibration   : isotonic GATES map g(.) -> tau_tilde(x);
                           spread scale kappa* and threshold c* chosen on VALIDATION
                           to MAXIMISE decision usefulness (AUUC of the r-ranking).
  Stage 5c Reliability   : r(x) = Phi( (tau_tilde(x) - c*) / (kappa* s_raw(x)) )
  Stage 7  Policy        : pi(x)=1{r(x)>=gamma*}; gamma* maximises VAL net value.

Honesty: r(x) is reported as a validated DECISION SCORE. Individual-level interval
coverage of tau(x) is not verifiable on real data; we report GROUP-level coverage
and reliability-decile-vs-observed-uplift diagnostics, and only use the word
"calibrated" where those diagnostics support it.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

from . import config as C
from . import baselines as B
from . import metrics as M


def _reg(params):
    return LGBMRegressor(**{**params, "objective": "regression"})


class DRANet:
    def __init__(self, cfg: C.RunConfig, uncertainty: str = "condvar",
                 n_bootstrap: int = 0):
        """uncertainty in {"condvar","bootstrap"}. n_bootstrap>0 also fits a bootstrap
        ensemble (used by uncertainty="bootstrap" and by ablation A3)."""
        self.cfg = cfg
        self.uncertainty = uncertainty
        self.n_bootstrap = n_bootstrap
        self.alpha = cfg.alpha
        self.z = float(norm.ppf(1 - cfg.alpha / 2))
        # kappa search grid (small values needed: pseudo-outcome dispersion >> tau scale)
        self.kappa_grid = np.concatenate([[1e-6], np.geomspace(0.01, 3.0, 24)])
        self.point_model_ = None
        self.var_model_ = None
        self.boot_models_: List = []
        self.nuis_ = None
        self.iso_ = None
        self.kappa_ = 1.0
        self.c_star_ = 0.0
        self.gamma_star_ = 0.5
        self.selection_grid_ = None
        self.c_kappa_grid_ = None

    # ------------------------------------------------------------------ #
    def fit(self, X, T, Y, n_folds: Optional[int] = None):
        X = np.asarray(X, np.float64); T = np.asarray(T, np.float64); Y = np.asarray(Y, np.float64)
        n_folds = n_folds or self.cfg.n_folds
        mu1, mu0, ehat, _ = B.cross_fit_nuisances(X, T, Y, self.cfg.lgbm_params, n_folds)
        phi = B.DRLearner.aipw_pseudo(Y, T, mu1, mu0, ehat)
        self.phi_train_ = phi
        self.point_model_ = _reg(self.cfg.lgbm_reg_params).fit(X, phi)
        # heterogeneous conditional dispersion v(x)=E[(phi-tau_hat)^2 | x]
        resid2 = (phi - self.point_model_.predict(X)) ** 2
        self.var_model_ = _reg(self.cfg.lgbm_reg_params).fit(X, resid2)
        # optional bootstrap ensemble (estimation variance of the final regressor)
        if self.n_bootstrap > 0:
            rng = np.random.default_rng(self.cfg.seed); n = len(X)
            self.boot_models_ = []
            for _ in range(self.n_bootstrap):
                idx = rng.integers(0, n, n)
                self.boot_models_.append(_reg(self.cfg.lgbm_reg_params).fit(X[idx], phi[idx]))
        self.nuis_ = B.fit_full_nuisances(X, T, Y, self.cfg.lgbm_params)
        return self

    # ------------------------------------------------------------------ #
    def _s_raw(self, X) -> np.ndarray:
        X = np.asarray(X, np.float64)
        if self.uncertainty == "bootstrap" and self.boot_models_:
            preds = np.column_stack([m.predict(X) for m in self.boot_models_])
            return preds.std(axis=1).astype(np.float64) + 1e-6
        v = np.clip(self.var_model_.predict(X), 1e-10, None)
        return np.sqrt(v).astype(np.float64)

    def predict_raw(self, X) -> Tuple[np.ndarray, np.ndarray]:
        tau_point = self.point_model_.predict(np.asarray(X, np.float64)).astype(np.float64)
        return tau_point, self._s_raw(X)

    def _apply_iso(self, tau_point):
        return tau_point if self.iso_ is None else self.iso_.predict(tau_point).astype(np.float64)

    @staticmethod
    def reliability(tau_tilde, s, c) -> np.ndarray:
        return norm.cdf((np.asarray(tau_tilde) - c) / np.asarray(s)).astype(np.float64)

    # ------------------------------------------------------------------ #
    def calibrate(self, Xval, Tval, Yval, e_val):
        tau_point, s_raw = self.predict_raw(Xval)
        Tval = np.asarray(Tval); Yval = np.asarray(Yval, np.float64)

        # (1) isotonic GATES calibration of point estimate
        cal = M.gates_calibration(Yval, Tval, tau_point, n_bins=self.cfg.n_gates_bins)
        pred_bins = np.array(cal["pred"]); obs_bins = np.array(cal["obs"])
        if len(pred_bins) >= 2 and np.std(pred_bins) > 1e-9:
            self.iso_ = IsotonicRegression(out_of_bounds="clip").fit(pred_bins, obs_bins)
        else:
            self.iso_ = None
        tau_tilde = self._apply_iso(tau_point)

        # (2) choose (c*, kappa*) maximising VALIDATION AUUC of the r-ranking.
        #     c grid includes the validation ATE (a practically-meaningful, interpretable
        #     "beats-the-average-user" threshold). If kappa*->0 recovers CATE ranking,
        #     that is an HONEST finding (reliability ~ CATE), not a failure.
        val_ate = float(Yval[Tval == 1].mean() - Yval[Tval == 0].mean())
        c_candidates = sorted(set([round(x, 6) for x in self.cfg.c_grid] + [round(val_ate, 6)]))
        base_auuc = M.auuc(Yval, Tval, tau_tilde)          # CATE-ranking reference
        grid = []
        best = {"c": c_candidates[0], "kappa": 1e-6, "auuc": -1e9}
        for c in c_candidates:
            for k in self.kappa_grid:
                r = self.reliability(tau_tilde, k * s_raw, c)
                a = M.auuc(Yval, Tval, r)
                grid.append({"c": float(c), "kappa": float(k), "val_auuc": float(a)})
                if a > best["auuc"]:
                    best = {"c": float(c), "kappa": float(k), "auuc": float(a)}
        self.c_kappa_grid_ = grid
        self.c_star_ = best["c"]; self.kappa_ = best["kappa"]
        self.val_auuc_cate_ = float(base_auuc)
        self.val_auuc_reliability_ = float(best["auuc"])

        # (3) choose gamma* maximising validation net value at (c*, kappa*)
        s = self.kappa_ * s_raw
        self.selection_grid_ = self._select_gamma(Yval, Tval, e_val, tau_tilde, s, self.c_star_)
        best_g = max(self.selection_grid_, key=lambda d: d["val_net_incremental"])
        self.gamma_star_ = best_g["gamma"]

        # (4) group-level coverage diagnostic (honest, not used for selection)
        self.group_coverage_ = self._group_coverage(Xval, Tval, Yval, tau_tilde, s)
        return self

    def _select_gamma(self, Y, T, e, tau_tilde, s, c) -> List[Dict]:
        Y = np.asarray(Y, np.float64); T = np.asarray(T, np.float64)
        r = self.reliability(tau_tilde, s, c)
        out = []
        for g in self.cfg.gamma_grid:
            pi = (r >= g).astype(np.float64)
            inc = M.policy_incremental_outcome(Y, T, pi, e)
            n_tar = float(pi.sum())
            out.append({"gamma": float(g), "val_target_frac": float(pi.mean()),
                        "val_incremental": float(inc),
                        "val_net_incremental": float(inc - c * n_tar)})
        return out

    def _group_coverage(self, X, T, Y, tau_tilde, s) -> float:
        order = np.argsort(tau_tilde); n = len(tau_tilde); nb = self.cfg.n_gates_bins
        edges = np.linspace(0, n, nb + 1).astype(int)
        tt, yy, cc, ss = T[order], Y[order], tau_tilde[order], s[order]
        cov = []
        for b in range(nb):
            sl = slice(edges[b], edges[b + 1]); tb, yb = tt[sl], yy[sl]
            if (tb == 1).sum() == 0 or (tb == 0).sum() == 0:
                continue
            obs = yb[tb == 1].mean() - yb[tb == 0].mean()
            lo = cc[sl].mean() - self.z * ss[sl].mean()
            hi = cc[sl].mean() + self.z * ss[sl].mean()
            cov.append(float(lo <= obs <= hi))
        return float(np.mean(cov)) if cov else float("nan")

    # ------------------------------------------------------------------ #
    def predict_cate(self, X):
        tau_point, _ = self.predict_raw(X)
        return self._apply_iso(tau_point)

    def predict_spread(self, X):
        return self.kappa_ * self._s_raw(X)

    def predict_reliability(self, X, c: Optional[float] = None):
        c = self.c_star_ if c is None else c
        return self.reliability(self.predict_cate(X), self.predict_spread(X), c)

    def predict_lcb(self, X):
        return (self.predict_cate(X) - self.z * self.predict_spread(X)).astype(np.float64)

    def predict_policy(self, X, c: Optional[float] = None, gamma: Optional[float] = None):
        c = self.c_star_ if c is None else c
        gamma = self.gamma_star_ if gamma is None else gamma
        return (self.predict_reliability(X, c=c) >= gamma).astype(np.int8)

    def predict_outcome(self, X, T):
        X = np.asarray(X, np.float64); T = np.asarray(T)
        p1 = self.nuis_["m1"].predict_proba(X)[:, 1]
        p0 = self.nuis_["m0"].predict_proba(X)[:, 1]
        return np.where(T == 1, p1, p0).astype(np.float64)

    def decision_frame(self, X, c: Optional[float] = None) -> Dict[str, np.ndarray]:
        c = self.c_star_ if c is None else c
        tau_tilde = self.predict_cate(X); s = self.predict_spread(X)
        r = self.reliability(tau_tilde, s, c)
        return {"cate": tau_tilde, "spread": s, "reliability": r, "lcb": tau_tilde - self.z * s}
