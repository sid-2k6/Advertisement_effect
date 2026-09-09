# CAPEX / Derived-Column Integrity Audit

All checks below were run directly against `train.csv` (2557 rows,
2016-01-01..2022-12-31) using pandas/numpy. Formulas were solved for
numerically (least-squares / direct arithmetic), not assumed.

## 1. `TOTAL_OPERATING_EXP`

Hypothesis: `TOTAL_OPERATING_EXP = LABOR_EXP + SUPPLY_EXP + OVERHEAD_EXP + CAPITAL_EXP + OTHER_OPERATING_EXP`

**Result: REJECTED.** Actual mean discrepancy = 14,255.63, max = 17,880.80
(i.e. `TOTAL_OPERATING_EXP` is consistently and substantially larger than the
sum of these five components; the gap correlates 0.97 with an implied
"PHARMACY_EXP" derived from the budget-variance identity below).

**Verified conclusion:** `TOTAL_OPERATING_EXP` includes an additional actual
expense category (most plausibly a PHARMACY expense, paralleling the existing
`BUDGETED_PHARMACY_EXP` budget line) that has **no corresponding raw actual
column in the supplied CSVs.** Its magnitude was triangulated two independent
ways (via the 5-component gap, and via the `BUDGET_VARIANCE` identity below);
the two independent estimates correlate at 0.97 with each other and both sit
around a mean of ~14,250-14,255, a ~1,600 std - consistent with an unobserved
actual pharmacy-expense series with a similar mean/spread as the existing
`BUDGETED_PHARMACY_EXP` column (mean ≈ its budgeted counterpart, ratio
actual/budgeted mean ≈ 0.99, std of ratio ≈ 0.09).

**I cannot verify the exact generating formula for `TOTAL_OPERATING_EXP` from
the supplied files** - the raw actual-expense column set in train/val/test.csv
is incomplete relative to what `TOTAL_OPERATING_EXP` and `BUDGET_VARIANCE`
were generated from. No preprocessing notebook, generator script, or data
dictionary was supplied in this repository checkout (branch `Suren-1`) that
documents this. This is stated explicitly rather than invented.

## 2. `BUDGET_VARIANCE`

Hypothesis: `BUDGET_VARIANCE = TOTAL_OPERATING_EXP - (BUDGETED_LABOR_EXP + BUDGETED_SUPPLY_EXP + BUDGETED_PHARMACY_EXP + BUDGETED_OVERHEAD_EXP)`

**Result: REJECTED as stated** (mean discrepancy = 3,608.39, max = 6,028.06),
**but this discrepancy correlates 0.997 with `CAPITAL_EXP`**, and fitting
`discrepancy ~ a*CAPITAL_EXP + b` by least squares gives slope ≈ 1.0005,
intercept ≈ 2,992.5, with max residual after this fit ≈ 172.8 (down from
6,028). This means:

`BUDGET_VARIANCE ≈ TOTAL_OPERATING_EXP - CAPITAL_EXP - (BUDGETED_LABOR_EXP + BUDGETED_SUPPLY_EXP + BUDGETED_PHARMACY_EXP + BUDGETED_OVERHEAD_EXP) - 2992.5`

i.e. capital expenditure is excluded from the budget-variance comparison (a
defensible accounting choice - CAPEX is often tracked separately from
operating-budget variance), but there remains a residual ~2,820-3,144
(mean 2,992.8, std only 48) unexplained additive offset that does not match
any single supplied column exactly. **I cannot fully verify the exact
`BUDGET_VARIANCE` formula from the supplied files** - it is very likely a
budgeted-vs-actual PHARMACY_EXP term (paralleling finding #1 above, since the
residual's order of magnitude and near-constant behavior across time is
consistent with a smoothly-budgeted line item), but I do not have the actual
PHARMACY_EXP column needed to confirm this exactly, and I explicitly decline
to invent one.

## 3. `OPERATING_MARGIN_PCT`

Hypothesis: `OPERATING_MARGIN_PCT = (NET_PATIENT_REVENUE - TOTAL_OPERATING_EXP) / NET_PATIENT_REVENUE * 100`

**Result: VERIFIED.** Max discrepancy = 2.1e-14 (floating-point noise only).
This identity holds exactly across all 2557 train rows.

## 4. What this means for the model / pipeline

- `TOTAL_OPERATING_EXP` and `BUDGET_VARIANCE` are used as-is from the CSVs
  (as a forecasting target and as an EXPENSE-stream feature respectively).
  Since their exact generating formula cannot be reconstructed from the
  supplied files, **no attempt was made to "correct" or recompute them** -
  doing so would require inventing an unobserved PHARMACY_EXP series, which
  the strict non-hallucination rule explicitly forbids. They are used
  unmodified, exactly as delivered in train/val/test.csv.
- `OPERATING_MARGIN_PCT` is verified fully consistent and is also used as-is
  (it is an EXPENSE-stream feature, not a target).
- No CAPEX-related "correction" was applied anywhere in this project, because
  no inconsistency was found that could be corrected without inventing data.
  The task's instruction "If the current preprocessing already performs the
  required CAPEX treatment correctly: DO NOT unnecessarily rewrite it" is
  followed by leaving all these columns untouched.

## 5. Explicit non-verifiable items (per the non-hallucination rule)

- The exact formula generating `TOTAL_OPERATING_EXP` from raw inputs: **cannot
  verify** (missing an actual-pharmacy-expense-equivalent raw column).
- The exact formula generating `BUDGET_VARIANCE`: **cannot verify** (same
  missing column, plus an unexplained ~2,993 constant offset).
- Whether any "CAPEX correction" was already applied upstream, and to what
  columns exactly: **cannot verify** - no preprocessing notebook, generator
  script, or data dictionary was found in this repository checkout (branch
  `Suren-1`) describing the CAPEX treatment. Only `Prop_model_v1 (1).ipynb`
  and the five CSVs were supplied; the preprocessing pipeline that produced
  those CSVs (referenced as `DeepBudgetVis_Preprocessing_v2.ipynb` in the
  request) was **not found in this repository/branch** and could not be
  inspected.
