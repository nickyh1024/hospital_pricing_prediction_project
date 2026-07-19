# Model card: admission-time hospital cost estimator

## Intended use

This model demonstrates how admission-time clinical and administrative information
can support an early hospital cost estimate. It is a portfolio case study for learning
and discussion. It must not be used for clinical decisions, patient eligibility,
billing, or a binding price quote.

## Data

The dataset contains 248 historical admissions from one hospital. The 18 predictors
cover demographics, admission details, vitals, laboratory values, complaint/history
codes, and whether an implant is planned. The target is total hospital cost.

Realized length of stay and realized implant cost are excluded because they occur
after admission and would leak future information into an admission-time prediction.

## Evaluation design

- Fixed 80/20 train-test split with random seed 42
- Five-fold shuffled cross-validation within the training set for model selection
- Median prediction as a required baseline
- MAE, RMSE, and R² measured in original currency units
- Permutation importance calculated only after selection on the held-out split
- Empirical 90% error band estimated from out-of-fold training residuals

## Current performance

Ridge regression was selected with a mean cross-validation MAE of ₹59,046. On 50
held-out admissions, MAE was ₹60,734, RMSE was ₹120,577, and R² was 0.235. The median
baseline MAE was ₹83,656, so the selected model improved test MAE by 27.4%.

## Performance slices

`outputs/subgroup_metrics.csv` reports held-out MAE and signed bias by gender, age
band, admission type, and planned implant use. Groups with fewer than five test rows
are suppressed. These are debugging slices, not a fairness certification: the sample
is too small to support reliable conclusions about equitable performance.

## Known limitations

- The sample is small, historical, and from a single institution.
- The test set contains only 50 rows, so all metrics have high uncertainty.
- Rare high-cost admissions are often underpredicted.
- No dates are available, so temporal drift cannot be measured.
- No external hospital data are available for transportability testing.
- The empirical uncertainty band is not a guaranteed conformal interval.

## Requirements before real-world use

A real deployment would require multi-site data, clinical and legal review, temporal
validation, calibrated intervals, subgroup analysis with adequate power, monitoring,
human oversight, security controls, and a clearly documented intervention policy.
