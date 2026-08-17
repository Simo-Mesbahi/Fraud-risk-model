# Methodology

## 1. Objective

This document describes the methodology used to build the **Health Insurance Fraud Risk Model**.

The project is designed as an end-to-end machine-learning system for prioritizing health-insurance claims for human fraud investigation.

The objective is not simply to classify claims as fraudulent or legitimate.

The operational objective is:

> Rank claims by estimated fraud risk so that a limited investigation team can concentrate its capacity on the highest-risk observations.

The primary operating assumption is an investigation capacity of approximately:

```text
3% of submitted claims
```

---

# 2. End-to-End Methodology

The project follows the pipeline:

```text
Problem Definition
        ↓
Synthetic Data Generation
        ↓
Data Validation
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Temporal Dataset Splitting
        ↓
Baseline Modelling
        ↓
Model Comparison
        ↓
Champion Selection
        ↓
Frozen Final Training
        ↓
Out-of-Time Evaluation
        ↓
Error Analysis
        ↓
SHAP Explainability
        ↓
Model Governance
        ↓
Reusable Inference
        ↓
FastAPI
        ↓
Docker
        ↓
Automated Tests
        ↓
Continuous Integration
```

---

# 3. Problem Formulation

Fraud detection is formulated as a binary supervised-learning problem:

```text
y = is_fraud
```

where:

```text
0 = legitimate synthetic claim
1 = fraudulent synthetic claim
```

However, because fraud prevalence is low and investigation capacity is constrained, the primary use of model output is **ranking**, not hard classification.

Each claim receives:

```text
P(fraud | available claim information)
```

The claims are then ordered from highest to lowest estimated risk.

---

# 4. Why Accuracy Is Not the Primary Metric

Fraud is rare.

On the final out-of-time test population, fraud prevalence is approximately:

```text
2.885%
```

A classifier predicting every claim as legitimate would therefore achieve high conventional accuracy while providing essentially no fraud-investigation value.

The project instead focuses on:

- Average Precision;
- ROC-AUC;
- Precision@K;
- Recall@K;
- Lift@K;
- fraud amount captured;
- calibration metrics.

Operational metrics are evaluated at investigation-capacity levels rather than only at a fixed probability threshold.

---

# 5. Synthetic Data Generation

The project uses synthetic data so that the entire workflow can be published without exposing sensitive insurance or healthcare information.

The generator creates:

- customers;
- policies;
- healthcare providers;
- healthcare claims;
- service information;
- reimbursement information;
- behavioural history;
- fraud mechanisms;
- legitimate anomalies;
- controlled missingness;
- invalid observations;
- temporal behaviour.

The generator is designed to create a non-trivial modelling problem rather than a perfectly separable fraud dataset.

---

# 6. Fraud Mechanisms

The synthetic environment contains several fraud mechanisms.

These include:

```text
customer_provider_pattern
repeated_service
mixed_pattern
frequency_abuse
provider_abnormality
amount_inflation
```

Different mechanisms produce different observable signals.

This enables evaluation not only of overall model performance but also of mechanism-specific weaknesses.

---

# 7. Fraud Difficulty

Fraud observations are also assigned synthetic difficulty levels:

```text
easy
medium
hard
```

These labels are diagnostic only.

They are excluded from model inputs and used after prediction to assess whether performance degrades as fraud patterns become more subtle.

---

# 8. Legitimate Anomalies

The generator intentionally produces unusual but legitimate claims.

Examples include:

- repeated legitimate services;
- unusual legitimate providers;
- legitimate high-frequency activity;
- legitimate high-amount claims.

This prevents the problem from becoming equivalent to simple anomaly detection.

A realistic fraud-risk system must distinguish:

```text
unusual
```

from:

```text
fraudulent
```

These synthetic anomaly labels are excluded from predictive features and used for false-positive analysis.

---

# 9. Data Validation

Generated data is validated before modelling.

Validation covers:

- schema expectations;
- numerical constraints;
- date consistency;
- missingness;
- referential integrity.

Examples of blocking validation rules include:

```text
claim_amount > 0
service_units >= 1
requested_reimbursement <= claim_amount
service_date <= claim_submission_date
```

Controlled missingness generates warnings rather than blocking failures.

---

# 10. Data Cleaning

The initial synthetic claims population contains:

```text
100,000 claims
```

The cleaning pipeline rejects observations violating blocking data-quality rules.

After cleaning:

```text
99,911 claims
```

remain.

Rejected observations:

```text
89 claims
```

are preserved separately for auditability.

After cleaning, validation reports:

```text
0 blocking errors
```

Controlled missing values remain intentionally preserved.

---

# 11. Exploratory Data Analysis

Exploratory analysis is performed before model development.

The EDA covers:

- missingness;
- fraud prevalence;
- claim amount distributions;
- reimbursement behaviour;
- service-category risk;
- service-code risk;
- customer-provider interactions;
- fraud mechanisms;
- numerical effect sizes;
- univariate discrimination;
- correlations;
- temporal claim volume;
- temporal fraud prevalence.

The EDA is implemented primarily in:

```text
notebooks/02_exploratory_data_analysis.ipynb
```

Generated figures are stored under:

```text
artifacts/metadata/eda_figures/
```

---

# 12. Feature Analysis

Feature analysis focuses on whether behavioural and contextual variables provide useful fraud discrimination.

The analysis includes:

- customer activity;
- provider activity;
- repeated services;
- customer-provider interactions;
- claim amount relative to historical baselines;
- temporal recency;
- behavioural ratios.

This work is documented in:

```text
notebooks/03_feature_analysis.ipynb
```

---

# 13. Feature Engineering Principles

Feature engineering follows several principles.

## Historical context

Fraud risk should depend on behaviour preceding the current claim.

Historical features therefore summarize past customer, provider and interaction activity.

## Relative behaviour

Absolute amounts alone may not identify suspicious activity.

The project therefore constructs relative variables such as:

```text
claim_to_service_median_ratio
claim_to_customer_avg_ratio
claim_to_provider_avg_ratio
```

## Behavioural intensity

Recent activity is compared with longer-term activity using features such as:

```text
recent_claim_share_30d_365d
recent_amount_share_30d_365d
provider_recent_activity_ratio
customer_provider_intensity
same_service_intensity
```

## Missingness information

Missingness is explicitly represented through indicator features rather than being silently discarded.

---

# 14. Temporal Leakage Prevention

Fraud modelling is especially vulnerable to temporal leakage.

A claim scored at time:

```text
t
```

must not use information that becomes available after:

```text
t
```

Historical features are therefore constructed from prior observations.

The project also avoids random train/test splitting because random splitting can mix future behaviour into the training population.

---

# 15. Target Leakage Prevention

Synthetic fraud generation uses variables that contain direct or indirect knowledge of the generated fraud outcome.

These diagnostic variables are excluded from predictive modelling.

Examples include:

```text
latent_fraud_score
synthetic_fraud_probability
fraud_difficulty
fraud_mechanism
legitimate_anomaly
legitimate_anomaly_type
```

The target:

```text
is_fraud
```

is also excluded from model inputs.

---

# 16. Temporal Dataset Split

The modelling population is divided chronologically.

| Split | Period |
|---|---|
| Training | 2023-01-01 → 2025-06-30 |
| Validation | 2025-07-01 → 2025-12-31 |
| Final test | 2026-01-01 → 2026-06-30 |

This reproduces the deployment direction:

```text
Past
  ↓
Model
  ↓
Future
```

The final 2026 period is never used for model selection.

---

# 17. Preprocessing

Numerical and categorical features are handled through a reusable preprocessing pipeline.

The preprocessor is fitted only on the appropriate training population.

It is then frozen and reused for:

- validation;
- final test evaluation;
- batch scoring;
- API inference.

This prevents training-serving preprocessing inconsistencies.

The frozen preprocessor is stored in:

```text
artifacts/preprocessors/health_fraud_preprocessor.joblib
```

---

# 18. Candidate Models

Several models are evaluated.

## Dummy Classifier

Purpose:

- establish a non-informative baseline;
- verify that trained models provide meaningful lift.

## Logistic Regression

Purpose:

- provide a strong linear baseline;
- provide interpretable probabilistic predictions.

## Balanced Logistic Regression

Purpose:

- evaluate class-weighting effects on recall and ranking.

## Random Forest

Purpose:

- model nonlinear relationships and interactions;
- provide a tree-based ensemble benchmark.

## XGBoost

Purpose:

- model nonlinear fraud patterns;
- capture complex interactions;
- provide strong ranking performance on structured tabular data.

---

# 19. Model Tuning Strategy

Hyperparameter tuning is intentionally constrained.

The objective is not exhaustive search.

Instead, tuning explores a compact set of technically meaningful configurations to balance:

```text
model quality
+
computational efficiency
+
methodological credibility
```

This is particularly important for Random Forest and XGBoost, where unnecessarily large grids can dramatically increase runtime without proportional improvement.

---

# 20. Model Selection

Candidate models are compared on the temporal validation period.

Important criteria include:

- Average Precision;
- ROC-AUC;
- Precision at investigation capacity;
- Recall at investigation capacity;
- Lift;
- fraud amount captured;
- calibration;
- computational practicality.

The final champion is:

```text
XGBoost
```

---

# 21. Frozen Final Training

After champion selection, the model configuration is frozen.

The final model is fitted on:

```text
training + validation
```

The final training matrix contains:

```text
85,735 observations
```

The final held-out test population contains:

```text
14,176 observations
```

After preprocessing, both matrices contain:

```text
107 transformed features
```

The test population remains untouched until this final evaluation.

---

# 22. Final Out-of-Time Evaluation

The frozen model is evaluated once on:

```text
2026-01-01 → 2026-06-30
```

Final test results:

| Metric | Result |
|---|---:|
| Test claims | 14,176 |
| Fraud cases | 409 |
| Fraud prevalence | 2.885% |
| Average Precision | **0.5520** |
| ROC-AUC | **0.8518** |
| Brier Score | **0.0174** |
| Log Loss | **0.0797** |

---

# 23. Capacity-Based Evaluation

The principal operational policy reviews the top:

```text
3%
```

of claims ranked by predicted fraud risk.

This corresponds to:

```text
426 claims
```

on the final test population.

Results:

| Metric | Result |
|---|---:|
| Precision @ 3% | **51.64%** |
| Recall @ 3% | **53.79%** |
| Lift @ 3% | **17.90×** |
| Fraud amount captured @ 3% | **55.15%** |

Confusion counts at this operating point:

```text
True positives:   220
False positives:  206
False negatives:  189
True negatives:   13,561
```

---

# 24. Why Top-K Evaluation Is Used

The operational question is not:

> Is the predicted probability above 0.50?

The operational question is:

> Given limited investigator capacity, which claims should be reviewed first?

Claims are therefore ranked by:

```text
fraud_risk_score
```

and the highest-risk fraction is selected.

This directly connects model evaluation with operational capacity.

---

# 25. Capacity Curve

The model is evaluated across multiple review fractions:

```text
0.5%
1%
2%
3%
5%
7.5%
10%
15%
```

As review capacity increases:

- recall increases;
- fraud amount capture increases;
- precision decreases;
- lift decreases.

This allows the operating policy to be selected according to available investigation resources.

---

# 26. Calibration

Fraud probabilities are evaluated using:

```text
Brier Score
Log Loss
Calibration curves
```

Calibration matters because the API exposes a fraud-risk probability rather than only a ranking.

However, ranking performance remains the primary business objective.

---

# 27. Error Analysis

Model evaluation does not stop at global metrics.

Dedicated error analysis examines:

- false positives;
- false negatives;
- fraud mechanisms;
- fraud difficulty;
- legitimate anomalies.

This helps determine **where the model fails**, not only how often.

---

# 28. Fraud-Mechanism Analysis

Final recall at the 3% investigation capacity varies substantially by fraud mechanism.

| Mechanism | Recall @ 3% |
|---|---:|
| Customer-provider pattern | 80.30% |
| Repeated service | 68.97% |
| Mixed pattern | 61.76% |
| Frequency abuse | 57.75% |
| Provider abnormality | 53.52% |
| Amount inflation | **8.00%** |

The main identified weakness is:

```text
amount_inflation
```

This mechanism accounts for:

```text
69 missed fraudulent claims
```

in the final test population.

---

# 29. Difficulty Analysis

Recall decreases as synthetic fraud becomes more difficult.

| Difficulty | Recall @ 3% |
|---|---:|
| Easy | 67.26% |
| Medium | 55.45% |
| Hard | 28.95% |

This provides evidence that the synthetic difficulty mechanism behaves as intended.

---

# 30. False Positive Analysis

At the 3% operating point:

```text
206 legitimate claims
```

are selected for investigation.

Among these:

```text
57.77%
```

correspond to simulated legitimate anomalies.

This is an important operational finding.

Many false positives are not arbitrary model errors; they represent unusual legitimate behaviour that resembles fraud.

This reinforces the requirement for human investigation.

---

# 31. False Negative Analysis

At the same operating point:

```text
189 fraudulent claims
```

are missed.

The largest missed category is:

```text
amount_inflation
```

followed by other more difficult fraud mechanisms.

False-negative analysis is therefore used to identify priorities for future model improvement.

---

# 32. Statistical Uncertainty

Bootstrap confidence intervals are estimated for major ranking metrics.

Average Precision:

```text
Estimate: 0.5520
95% CI: [0.5036, 0.6029]
```

ROC-AUC:

```text
Estimate: 0.8518
95% CI: [0.8235, 0.8782]
```

These intervals quantify sampling uncertainty within the synthetic test population.

They do not quantify the synthetic-to-real-world generalization gap.

---

# 33. Explainability

The final frozen XGBoost model is analysed using SHAP.

The explainability pipeline includes:

- global SHAP importance;
- SHAP beeswarm analysis;
- business-feature aggregation;
- local claim explanations;
- false-positive explanations;
- false-negative explanations;
- hard-fraud cases;
- legitimate anomalies.

SHAP is computed on a representative sample of:

```text
5,000 held-out test claims
```

---

# 34. Main Model Drivers

The strongest business-level drivers are:

```text
1. claim_to_service_median_ratio
2. days_since_customer_previous_claim
3. reimbursement_ratio
4. provider_claims_30d
5. submission_month
```

The strongest driver is:

```text
claim_to_service_median_ratio
```

This supports the importance of contextual amount comparisons in fraud-risk ranking.

SHAP values describe model behaviour and must not be interpreted as causal effects.

---

# 35. Model Artifacts

The frozen final model is stored in:

```text
artifacts/models/health_fraud_xgboost.joblib
```

The preprocessing pipeline is stored in:

```text
artifacts/preprocessors/health_fraud_preprocessor.joblib
```

Model metadata is stored in:

```text
artifacts/metadata/health_fraud_model_metadata.json
```

These artifacts form the frozen inference contract.

---

# 36. Production-Style Scoring

Reusable inference is implemented through:

```text
src/health_fraud/models/predict.py
```

The scoring layer:

1. loads frozen artifacts;
2. validates metadata;
3. validates input data;
4. constructs required features;
5. applies the frozen preprocessor;
6. predicts fraud probabilities;
7. ranks claims by risk;
8. selects a configurable top investigation fraction.

This prevents notebooks from becoming the production inference interface.

---

# 37. Batch Scoring

Batch scoring is available through:

```bash
python scripts/score_claims.py
```

The scoring workflow generates:

```text
artifacts/predictions/claim_scores.parquet
artifacts/predictions/top_review_claims.parquet
```

This provides a non-API inference pathway for batch investigation workflows.

---

# 38. REST API

The frozen model is exposed through FastAPI.

Endpoints include:

```text
GET  /
GET  /health
GET  /model-info
POST /score
POST /score-batch
POST /top-review
GET  /docs
```

The API uses the same feature-building, preprocessing and scoring pipeline as batch inference.

This minimizes training-serving skew.

---

# 39. Dockerization

The API is containerized using:

```text
Dockerfile
```

and orchestrated locally using:

```text
docker-compose.yml
```

The container includes an automated healthcheck against:

```text
/health
```

The validated Docker Compose service reports:

```text
healthy
```

when the model and API are available.

---

# 40. Automated Testing

The automated test suite validates:

- API health;
- model metadata;
- single scoring;
- batch scoring;
- investigation ranking;
- invalid review fractions;
- missing model features.

Validated test result:

```text
7 passed
```

The test suite can be executed with:

```bash
PYTHONPATH=.:src python -m pytest api/tests tests -v
```

---

# 41. Continuous Integration

GitHub Actions automatically runs the project test workflow on:

```text
push → main
pull request → main
```

The CI workflow performs:

```text
Repository Checkout
        ↓
Python 3.12 Setup
        ↓
Dependency Installation
        ↓
Compilation Checks
        ↓
Automated Tests
```

This verifies that changes do not break the inference or API workflow.

---

# 42. Human-in-the-Loop Design

The intended operational workflow is:

```text
Model
  ↓
Risk Ranking
  ↓
Investigation Queue
  ↓
Human Investigator
  ↓
Decision
```

The model is a prioritization system.

It is not a fraud adjudication system.

A high model score means:

> The claim exhibits patterns associated with fraudulent synthetic claims.

It does not mean:

> Fraud has been proven.

---

# 43. Monitoring Methodology

A real deployment should monitor four dimensions.

## Data Quality

Monitor:

- missingness;
- invalid values;
- schema changes;
- categorical-domain changes.

## Data Drift

Monitor changes in important variables including:

- claim amount;
- reimbursement ratio;
- service mix;
- provider activity;
- customer activity;
- relative amount ratios.

## Model Performance

Once labels become available:

- Average Precision;
- ROC-AUC;
- Precision@K;
- Recall@K;
- Lift@K;
- fraud amount captured;
- calibration.

## Operational Performance

Monitor:

- investigation volume;
- confirmed-fraud yield;
- false-positive burden;
- investigator capacity;
- recovered or prevented fraud amount.

---

# 44. Retraining Strategy

Retraining should be considered when:

- important features drift materially;
- Precision@K deteriorates;
- Recall@K deteriorates;
- fraud amount capture falls;
- calibration degrades;
- fraud mechanisms change;
- operational investigation capacity changes.

Retraining should reproduce the same temporal validation discipline used in initial development.

---

# 45. Methodological Limitations

The most important limitation is that the complete environment is synthetic.

Consequently:

- fraud mechanisms are simulated;
- behavioural relationships are designed rather than naturally observed;
- real healthcare coding complexity is absent;
- real fraud networks may behave differently;
- real missingness mechanisms may differ;
- operational feedback loops are not reproduced.

The reported results therefore validate the **methodology and engineering implementation**, not expected real-world insurance performance.

---

# 46. Reproducibility

The main workflow can be reproduced using:

```bash
python scripts/generate_data.py --overwrite
python scripts/validate_data.py --data-dir data/synthetic
python scripts/clean_data.py
python scripts/validate_data.py --data-dir data/interim
python scripts/score_claims.py
PYTHONPATH=.:src python -m pytest api/tests tests -v
```

The API can be launched using:

```bash
docker compose up --build -d
```

---

# 47. Related Documentation

Additional project documentation:

```text
docs/problem_definition.md
docs/data_dictionary.md
docs/model_card.md
README.md
```

The responsibilities are intentionally separated:

| Document | Purpose |
|---|---|
| `README.md` | High-level project presentation and usage |
| `problem_definition.md` | Business problem and modelling objective |
| `data_dictionary.md` | Data and feature definitions |
| `methodology.md` | End-to-end modelling methodology |
| `model_card.md` | Final model performance, governance and limitations |

---

# 48. Conclusion

The methodology deliberately connects machine-learning development with the operational fraud-investigation problem.

The project does not stop at training a classifier.

It implements:

```text
Data Quality
+ Temporal Validation
+ Leakage Control
+ Feature Engineering
+ Model Comparison
+ Capacity-Based Evaluation
+ Error Analysis
+ Explainability
+ Governance
+ Reusable Inference
+ REST API
+ Docker
+ Automated Testing
+ Continuous Integration
```

The resulting system demonstrates an end-to-end approach to fraud-risk prioritization while preserving the central principle that model predictions support, rather than replace, human investigation.