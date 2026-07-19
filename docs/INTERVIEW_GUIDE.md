# Interview guide

## Thirty-second summary

“I built an admission-time hospital cost estimator from a 248-patient case-study
dataset. I first reframed the problem to remove future-data leakage, then put all
imputation and encoding inside scikit-learn pipelines. I compared a median baseline,
Ridge, and random forest using cross-validation. Ridge achieved a 27.4% lower held-out
MAE than the baseline, but it underpredicts rare high-cost cases, so I surfaced that
limitation and an empirical error band in the app.”

## Decisions to be ready to defend

### Why exclude length of stay and implant cost?

The product goal is an estimate at admission. Final length of stay and realized
implant cost are not known then. Including them would produce impressive metrics for
a model that could not operate at the stated decision point.

### Why log-transform the target?

Costs are strongly right-skewed. A log transform reduces the influence of extreme
values while ensuring positive inverse-transformed predictions. Evaluation remains in
the original currency so the error is understandable.

### Why did Ridge beat random forest?

With only 198 training rows, a regularized linear model can generalize better than a
more flexible ensemble. The cross-validation difference is small, so the conclusion
is “Ridge won this evaluation,” not “Ridge is universally superior.”

### What would you do with more time and data?

Use a temporal holdout, obtain multiple hospitals, bootstrap metric confidence
intervals, tune models with nested cross-validation, test calibrated conformal
intervals, analyze adequately sized subgroups, and monitor drift after deployment.

## Honest weaknesses to volunteer

- Test R² is only 0.235 and the highest-cost cases are underpredicted.
- The 50-row test split makes subgroup metrics unstable.
- Planned implant use may be uncertain at admission and needs domain confirmation.
- The dataset does not support causal or clinical claims.
