# Distribution shift across splits (train / val / test)

Measured directly on the raw (unscaled) target columns:

| split | NET_PATIENT_REVENUE mean | NET_PATIENT_REVENUE std | TOTAL_OPERATING_EXP mean | TOTAL_OPERATING_EXP std |
|---|---|---|---|---|
| train (2016-2022) | 142,976 | 20,795 | 100,604 | 7,782 |
| val (2023-2024 H1) | 172,770 | 17,820 | 110,349 | 5,894 |
| test (2024 H2-2025) | 183,290 | 17,319 | 113,586 | 6,122 |

Both targets show a clear upward level shift from train -> val -> test (this
is a synthetic 10-year hospital dataset with an apparent growth trend), while
the standard deviation shrinks slightly. This is a genuine, verified property
of the supplied data (not an assumption).

**Implication for the results in this report:** validation performance
(measured on 2023 - mid-2024) is not a perfect proxy for test performance
(measured on mid-2024 - 2025) precisely because the two periods have
different target-level statistics. Since the target scaler (log1p + z-score)
is fit ONLY on train (2016-2022) and never refit, both val and test are
scored in a regime the scaler never saw calibrated data for - which is the
correct, leakage-free way to evaluate, but it also means absolute R2/MAE
naturally degrade from val to test simply because both models are
extrapolating into a higher, still-shifting revenue regime. This affects
BOTH the baseline and the upgraded model equally (both lose ~0.13-0.14 R2
points on NET_PATIENT_REVENUE going from val to test), and is the most
likely explanation for why the val-set ranking (upgraded > baseline) did not
fully hold on the test set, where the ranking reversed to a small degree.
This is offered as an interpretation grounded in a measured, verified data
property - not as a certainty, since isolating the exact causal reason for a
model-comparison reversal on 549 test rows is not possible from this data
alone.
