# Causal Inference-Driven Interpretable Visualization for Advertising Effect Analysis
## Final Research Summary — DRA-Net (Decision-Reliability Attribution Network)

*All numbers below are computed on the real Criteo Uplift v2.1 data (13,979,592 rows). The
final test split (2,096,939 rows) was scored once, after all model/threshold selection was
frozen on validation. Where the proposed method does not beat a baseline, this is stated
explicitly — no superiority is manufactured.*

---

## 1. Research problem
Digital advertisers must decide **whom to treat** (show ads) to maximize *incremental* outcomes,
not predicted outcomes. This requires per-user causal effect estimates and, crucially, a
**decision** built on them. Standard practice ranks users by a point estimate of the conditional
average treatment effect (CATE) `τ̂(x)`, ignoring how *reliable* each estimate is.

## 2. Existing limitation
Across the uplift / advertising causal-ML literature, the object that is ranked, explained, and
visualized is the **point CATE** `τ̂(x)` (or, at best, an interval reported *descriptively*).
Uncertainty is rarely folded into the actual targeting **decision**, the feature **attribution**,
and the **visualization** as a single quantity.

## 3. Research gap
No prior method makes the **calibrated exceedance probability** `r(x)=P(τ(x)>c)` — the probability
that advertising helps a user by more than a practically-meaningful threshold `c` — the *shared*
object of (i) the targeting policy, (ii) the causal attribution, and (iii) the interpretable
visualization.

## 4. Proposed solution — DRA-Net
A **reliability-aware causal decision framework**. On top of a cross-fitted doubly-robust CATE
backbone we estimate a **heterogeneous uncertainty** `s(x)`, calibrate the effect on validation,
and compute the reliability score `r(x)=P(τ(x)>c)`. This single quantity then drives targeting,
treatment recommendation, causal attribution, visualization, and segmentation.

## 5. Novel component (what is and is NOT claimed)
- **NOT claimed as novel** (all used as standard, cited components): S/T/X/R/DR-Learners, Causal
  Forest, Uplift RF, class transformation, propensity/IPTW, conformal ideas, CATE uncertainty,
  SHAP, Qini/AUUC, policy value, Streamlit, persuadable/sleeping-dog segmentation.
- **Claimed contribution:** the reformulation of advertising uplift around the calibrated
  **decision-reliability estimand `r(x)=P(τ>c)`** as the *simultaneous* target of the policy,
  the Shapley attribution (explaining decision *confidence*, not effect *magnitude*), and a
  dedicated visualization family (Reliability–Effect plane, reliability-aware / uncertainty-
  decomposed Qini) — instantiated and tested at 14M-row scale on a real advertising RCT.

## 6. Mathematical formulation
- Potential outcomes `Y(1),Y(0)`; `τ(x)=E[Y(1)−Y(0)|X=x]`. Identified by Criteo's randomization
  (unconfoundedness by design; overlap holds — see §11).
- AIPW pseudo-outcome (cross-fitted nuisances): `φ = μ̂₁−μ̂₀ + T(Y−μ̂₁)/ê − (1−T)(Y−μ̂₀)/(1−ê)`,
  with `E[φ|X]=τ(x)`; DR-Learner regresses `φ` on `X` → `τ̂(x)`.
- **Uncertainty:** `s(x)=√v(x)`, `v(x)=E[(φ−τ̂(x))²|X=x]` (heterogeneous conditional dispersion).
- **Calibration:** isotonic GATES map `g(·)` on validation → `τ̃(x)=g(τ̂(x))`; spread scale `κ`.
- **Reliability:** `r(x)=P(τ(x)>c)=Φ((τ̃(x)−c)/(κ·s(x)))`.
- **Policy:** `π(x)=1{r(x)≥γ}`. **Policy value (IPW, RCT-valid):**
  `V(π)=E[Y·(Tπ/e+(1−T)(1−π)/(1−e))]`.
- **`c`** = minimum practically-meaningful absolute effect (business input; swept for sensitivity).
  `c`, `κ`, `γ` are all **selected on validation** — never assumed (visit: `c*=0.02, κ*=0.035,
  γ*=0.5`; conversion: `c*=0.00102, κ*=0.53, γ*=0.5`).

## 7. Experimental setup
- Data: Criteo v2.1, 13,979,592 rows; features f0–f11; `treatment` (85% treated, randomized);
  `exposure` **dropped** (post-treatment mediator). Split 70/15/15 =
  **9,785,714 / 2,096,939 / 2,096,939** (float32, memory-safe, shape assertions throughout).
- Two **separate** experiments: **A = visit (primary)**, **B = conversion (secondary stress test)**.
- Training on a documented fair subsample (1,000,000 for all CATE learners + DRA-Net nuisances;
  300k forests; 400k neural). **Validation and test are the FULL locked splits**; test scored once.

## 8. Baselines (categorised, not all "SOTA")
- **Classical:** Naive ATE, Logistic-interaction, Class-Transformation.
- **Meta-learners:** S-, T-, X-, R-, DR-Learner (LightGBM).
- **Strong modern:** Causal Forest (EconML), Uplift RF (causalml), TARNet, CFRNet, DragonNet.
- **Recent:** DESCN-style (entire-space), CHAUN-style (attention).
- **Proposed:** DRA-Net.

## 9. SOTA comparison (honest headline, visit — normalized Qini on locked test)
| Model | Category | Qini | Uplift@20% |
|---|---|---|---|
| TARNet | strong modern | **0.094** | 0.040 |
| **DRA-Net\* (TARNet backbone)** | proposed | **0.091** | **0.041** |
| CFRNet / UpliftRF / S-Learner | strong/meta | ~0.089–0.091 | ~0.040 |
| **DRA-Net (r-policy, DR backbone)** | proposed | 0.084 | 0.038 |
| DRA-Net (DR backbone, CATE ranking) | proposed | 0.066 | 0.033 |

**Honest statement:** DRA-Net built on the doubly-robust DR-Learner backbone does **not** beat the
best neural baseline (TARNet) in absolute Qini — the DR backbone is itself the weakest CATE base
here, and bootstrap tests confirm **TARNet significantly exceeds DRA-Net(DR)** on Qini/AUUC/uplift@k.
When the reliability layer wraps the **validation-selected** backbone (DRA-Net\*), it is **on par
with the best baseline** while additionally delivering calibrated confidence, segmentation and
attribution.

## 10. The decisive comparison (does `r(x)` add value over `τ̂`?) — the core hypothesis
Same DR backbone, four ranking rules, locked test:
| Ranking | Qini (visit) | Uplift@20% | Qini (conversion) |
|---|---|---|---|
| `τ̂(x)` (point CATE) | 0.066 | 0.033 | 0.091 |
| Lower confidence bound `τ̂−zs` | **−0.053** | 0.009 | (worst) |
| **`r(x)=P(τ>c)`** | **0.084** | **0.038** | **0.136** |

**Finding:** ranking by `r(x)` **significantly beats** point-CATE ranking **and** LCB ranking on the
same backbone (paired bootstrap, 95% CI excludes 0). Naive lower-bound targeting is markedly *worse*
than CATE — a noteworthy negative result that distinguishes `P(τ>c)` from conservative targeting.

## 11. Causal findings
- **Randomized assignment confirmed:** propensity AUC = **0.507** (visit) / 0.507 (conversion),
  common support = **1.00**, max |standardized mean difference| = **0.047** (<0.1), effective sample
  size ≈ full. Propensity is used only for **diagnostics / doubly-robust estimation** — *not* for
  identification, and the near-0.5 AUC is treated as *evidence of randomization*, not a motivation
  for any particular learner.
- Treatment is **imbalanced (85% treated)** though randomized; the control arm is the scarce one.
- ATE (naive, test): visit **+0.01027**, conversion **+0.00115**. Conversion is extremely rare
  (0.29% base rate) → per-user signal is weak and estimates carry wide CIs (reported, not hidden).

## 12. Backbone-generality finding (the key scientific insight)
The reliability layer is backbone-agnostic. Its benefit **depends on backbone quality**:
| backbone | Δ Qini (r − τ̂), visit | Δ Qini, conversion |
|---|---|---|
| DR-Learner (noisy) | **+0.018** | **+0.045** |
| T-Learner | +0.011 | +0.006 |
| X-Learner | +0.006 | +0.011 |
| Causal Forest | +0.004 | −0.004 |
| S-Learner / CFRNet / TARNet (strong) | ≈ 0.000 / −0.001 | ≈ 0.000 / −0.006 |

**Interpretation:** reliability weighting **corrects over-optimistic, high-variance CATE estimates**,
so it substantially helps *weak/noisy* estimators and **gracefully reduces to CATE ranking** on
already-strong ones (validation picks small `κ`), never harming them materially.

## 13. Ablation findings (visit; uplift@20% on locked test)
| Ablation | Uplift@20% | note |
|---|---|---|
| A0 full | **0.0383** | reference |
| A1 no reliability (CATE rank) | 0.0333 | −13% → reliability helps |
| A3 no uncertainty (`s→0`) | 0.0319 | uncertainty matters most |
| A4 no threshold `c` (`c=0`) | 0.0256 | collapses to target-all |
| A7 fixed vs validation-selected | 0.0256 | validation selection is essential |
| A8 c∈{0.005,0.05} | 0.0372 / 0.0384 | robust to `c` |
| A9 γ∈{0.5,0.9} | 0.0383 / 0.0383 | robust to `γ` |
| A10 alt. backbone (X-Learner) | **0.0388** | reliability transfers across estimators |
A5 (attribution) and A6 (calibration) leave *ranking* metrics unchanged by construction; A5 removes
the explanation layer and A6 the group-calibration guarantee.

## 14. Interpretability findings (effect vs confidence — two distinct questions)
- **Effect explanation** — SHAP of `τ̂(x)` (the DR pseudo-outcome regressor): *why the model
  believes advertising helps this user*.
- **Reliability explanation** — SHAP of the reliability decision-score `(τ̃−c)/s` (faithful
  surrogate, R² = **0.972** visit / **0.77** conversion): *why the model is confident it helps*.
- These are produced as separate global bar/beeswarm plots, a side-by-side "effect vs confidence"
  chart, and local waterfalls for a positive-effect and a negative-effect user. We explicitly state
  which model output each SHAP explains; ordinary `P(Y|X,T)` SHAP is **not** conflated with causal
  explanation.

## 15. Reliability validation (per requirement 11 — not called "calibrated" unless validated)
- Group-level interval coverage (test) ≈ nominal; interval width (visit) = 0.037.
- **Reliability-decile monotonicity:** Spearman(r-decile, observed RCT uplift) = **0.82** (visit) —
  higher predicted reliability → higher *observed* incremental response. Observed uplift by `r`-decile
  and by CATE-decile are reported side by side.
- Segments: **Reliable/Uncertain Persuadables, Reliable Sleeping Dogs, Uncertain Negatives** (plus
  classical Persuadable/Sure-Thing/Sleeping-Dog/Lost-Cause). At the operating point ~8.9% of users
  are *confident* persuadables (visit); their observed uplift (0.054) far exceeds the population ATE
  (0.010), and they capture ~68% of total incremental visits.

## 16. Visualization findings (title component)
- **Reliability–Effect plane** (`τ̂` vs `r(x)`, four decision regions, `c`/`γ` lines) — the decision
  framework, not a mere scatter.
- **Reliability-aware Qini** (r-ranking vs CATE-ranking) and **Uncertainty-Decomposed Qini**
  (confident vs speculative gain), with defined formulas.
- Full standard suite: propensity/overlap/balance, CATE distributions/percentiles, Qini/uplift@k,
  targeting & policy-value curves, threshold-sensitivity grids, SHAP bar/beeswarm/waterfall,
  comparison heatmap. 66 figures under `plots/`.

## Advertising implications
`r(x)` gives a budget-aware, confidence-aware targeting rule: target the ~9% of users the model is
*confident* the ad helps beyond cost `c`. This captures the majority of incremental visits while
avoiding spend on users whose apparent effect is an artifact of estimation noise — and it comes with
a per-user explanation of *why* the decision is trustworthy.

## Limitations & threats to validity
- Features f0–f11 are anonymized → no semantic interpretation of drivers.
- External validity limited to Criteo's population/period.
- **Conversion is extremely rare (0.29%)** → conversion-side estimates are low-powered with wide CIs;
  reported as a stress test, not a headline.
- ~1.66M rows share identical feature vectors (coarse anonymized features) → feature-space overlap
  across splits; not label leakage, but a fidelity limit inherited from the dataset.
- The predictive distribution uses a Gaussian working form validated by *group-level* coverage;
  individual-level coverage is not verifiable on real data (no counterfactuals).
- DRA-Net on the DR backbone does not top the absolute Qini leaderboard; its value is (a) improving
  weak backbones, (b) matching the best backbone when wrapped around it, and (c) the decision /
  confidence / interpretability machinery — **not** a universal accuracy win.

## Future work
Direct optimization of `r(x)` end-to-end; conformal predictive systems for `s(x)`; budget-constrained
knapsack policies on `r`; multi-treatment / continuous-dose reliability; online/bandit deployment.

---

## Explicit title alignment
- **Causal Inference-Driven** → RCT identification, propensity diagnostics, cross-fitted AIPW /
  DR-Learner CATE, ATE, IPW policy value.
- **Interpretable** → dual SHAP: effect attribution (`τ̂`) **and** reliability attribution (`r(x)`).
- **Visualization** → Reliability–Effect plane, reliability-aware & uncertainty-decomposed Qini,
  CATE/policy/uncertainty plots, interactive 7-page Streamlit dashboard.
- **Advertising Effect Analysis** → incremental effect estimation, confidence-aware targeting,
  policy value, persuadable segmentation on a real ad RCT.

## Final novelty statement
We propose a **reliability-aware causal decision framework** in which the probability of exceeding a
practically-meaningful treatment-effect threshold, **`r(x)=P(τ(x)>c)`**, is used as a *unified*
decision quantity for advertising targeting, causal attribution, and visual analysis. Experimentally,
`r(x)` **significantly improves targeting over point-CATE ranking and over lower-confidence-bound
ranking on the same estimator**, with the largest gains on noisy estimators and graceful no-harm
behavior on strong ones; it does **not** by itself beat the strongest neural baseline in absolute
uplift ranking, which we report transparently.
