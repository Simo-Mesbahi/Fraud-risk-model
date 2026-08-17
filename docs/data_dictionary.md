# Data Dictionary

## 1. Purpose

This document describes the datasets used by the **Health Insurance Fraud Risk Model**.

The project uses fully synthetic health-insurance data designed to reproduce realistic relationships between:

- customers;
- insurance policies;
- healthcare providers;
- healthcare services;
- claims;
- reimbursements;
- historical behaviour;
- legitimate anomalies;
- fraud mechanisms.

The synthetic data supports the complete machine-learning workflow:

```text
Generation
    ↓
Validation
    ↓
Cleaning
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Temporal Evaluation
    ↓
Fraud Risk Scoring
```

The dataset must not be interpreted as real insurance data.

---

# 2. Core Datasets

Four primary datasets are generated:

| Dataset | Description | Approximate Rows |
|---|---|---:|
| `customers.parquet` | Customer-level information | 20,000 |
| `providers.parquet` | Healthcare-provider information | 1,200 |
| `policies.parquet` | Insurance-policy information | 20,000 |
| `claims.parquet` | Claim-level transactional data | 100,000 before cleaning |

After data-quality cleaning, the modelling claim population contains:

```text
99,911 claims
```

The rejected observations are preserved separately in:

```text
data/interim/claims_rejected.parquet
```

---

# 3. Storage Layers

The project separates data according to processing stage.

## `data/raw/`

Reserved for raw source data.

The current project uses synthetic generation, therefore no external production insurance data is stored here.

## `data/synthetic/`

Contains the directly generated synthetic datasets:

```text
customers.parquet
providers.parquet
policies.parquet
claims.parquet
```

These files can intentionally contain controlled data-quality problems.

## `data/interim/`

Contains validated and cleaned datasets used downstream.

```text
customers.parquet
providers.parquet
policies.parquet
claims.parquet
claims_rejected.parquet
```

## `data/processed/`

Reserved for additional processed datasets when required by downstream workflows.

---

# 4. Customers Dataset

File:

```text
customers.parquet
```

Primary key:

```text
customer_id
```

The customer table represents insured individuals.

It contains customer-level demographic, contractual and behavioural attributes used to provide context for individual claims.

Representative variables include:

| Variable / Family | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| age-related variables | Synthetic customer demographic information |
| tenure-related variables | Duration of the customer relationship |
| coverage-related variables | Synthetic insurance coverage characteristics |
| behavioural segment | Synthetic customer behaviour profile |

Customer attributes are joined to claims through:

```text
customer_id
```

---

# 5. Providers Dataset

File:

```text
providers.parquet
```

Primary key:

```text
provider_id
```

The provider table represents healthcare providers associated with submitted claims.

Representative variables include:

| Variable / Family | Description |
|---|---|
| `provider_id` | Unique healthcare-provider identifier |
| provider type | Synthetic provider category |
| provider region | Synthetic geographic category |
| provider tenure | Duration of provider activity |
| behavioural segment | Synthetic provider behaviour profile |

Provider information is joined to claims through:

```text
provider_id
```

A controlled subset of synthetic claims can have a missing provider identifier.

This missingness is intentionally preserved and represented explicitly during feature engineering.

---

# 6. Policies Dataset

File:

```text
policies.parquet
```

Primary policy identifiers connect insurance-contract information to individual claims.

Representative variables include:

| Variable / Family | Description |
|---|---|
| policy identifier | Unique synthetic policy identifier |
| customer identifier | Customer associated with the policy |
| coverage characteristics | Synthetic insurance coverage information |
| policy tenure | Age of the insurance contract |
| policy-change information | Information about recent contractual changes |

Policy variables provide contractual context for fraud-risk estimation.

---

# 7. Claims Dataset

File:

```text
claims.parquet
```

Primary key:

```text
claim_id
```

The claim table is the central modelling dataset.

Each row represents one synthetic health-insurance claim.

Claims are linked to customer, policy and provider information.

---

# 8. Claim Identifiers

## `claim_id`

Unique claim identifier.

Example:

```text
CLM_00075763
```

Used for:

- scoring outputs;
- investigation queues;
- API responses;
- model diagnostics;
- explainability cases.

It is an identifier and is not intended to provide predictive information.

---

## `customer_id`

Identifier of the customer associated with the claim.

Used to join customer information and construct historical customer behaviour.

---

## `provider_id`

Identifier of the healthcare provider associated with the claim.

Used to join provider information and construct provider-level behavioural history.

Missing values are intentionally simulated for a controlled subset of claims.

---

# 9. Claim Financial Variables

## `claim_amount`

Total amount associated with the healthcare claim.

Data-quality requirement:

```text
claim_amount > 0
```

Invalid observations are rejected during cleaning.

---

## `requested_reimbursement`

Amount requested for reimbursement.

Data-quality requirement:

```text
requested_reimbursement <= claim_amount
```

This variable contributes to reimbursement-related features.

---

## `reimbursement_ratio`

Engineered ratio measuring the proportion of the claim requested for reimbursement.

Conceptually:

```text
requested_reimbursement / claim_amount
```

This variable is an important fraud-risk signal in the final model.

---

## `requested_to_limit_ratio`

Engineered feature comparing requested reimbursement with the applicable coverage or reimbursement limit.

It captures whether a request is unusually large relative to contractual limits.

---

# 10. Service Variables

Claims contain information describing the healthcare service.

Representative variables include:

| Variable / Family | Description |
|---|---|
| service category | Broad healthcare-service category |
| service code | Synthetic service identifier |
| `service_units` | Number of service units associated with the claim |
| `service_date` | Date on which the healthcare service occurred |

Data-quality requirement:

```text
service_units >= 1
```

---

# 11. Claim Submission Variables

## `claim_submission_date`

Date on which the claim was submitted.

Data-quality requirement:

```text
service_date <= claim_submission_date
```

The submission date is also the primary temporal reference used for historical feature construction and temporal dataset splitting.

---

## `submission_hour`

Hour at which the claim was submitted.

Engineered from claim submission information.

---

## `submission_dayofweek`

Day of the week of claim submission.

---

## `submission_month`

Calendar month of claim submission.

This feature appears among the strongest business-level model drivers.

---

## `submission_is_weekend`

Indicator identifying claims submitted during a weekend.

---

# 12. Service-Date Features

Temporal service features include:

```text
service_dayofweek
service_month
service_is_weekend
```

These variables represent calendar characteristics of the underlying healthcare service.

---

# 13. Documentation Variables

## `document_count`

Number of documents associated with a claim.

Controlled missing values are intentionally generated.

The missingness itself is represented by:

```text
document_count_missing
```

---

## `has_prescription`

Indicates whether a prescription is associated with the claim when applicable.

Controlled missing values are intentionally generated.

Missingness is represented by:

```text
has_prescription_missing
```

---

# 14. Missingness Indicators

The production feature pipeline explicitly models several forms of missing information.

Examples include:

```text
has_prescription_missing
document_count_missing
provider_missing
days_since_policy_change_missing
```

These indicators allow the model to distinguish between:

```text
observed value
```

and:

```text
information unavailable
```

rather than silently treating both situations as equivalent.

---

# 15. Customer Historical Features

Historical customer behaviour is computed using information available before the claim being scored.

Representative features include:

```text
customer_claims_7d
customer_claims_30d
customer_claims_90d
customer_claims_365d
customer_amount_30d
customer_amount_365d
customer_avg_claim_amount_365d
days_since_customer_previous_claim
```

These variables measure recent claim frequency, amount and recency.

---

# 16. Provider Historical Features

Provider-level behavioural features describe recent activity associated with a healthcare provider.

Representative features include:

```text
provider_claims_30d
provider_claims_90d
provider_avg_claim_amount_90d
provider_tenure_months
provider_recent_activity_ratio
```

These variables help identify abnormal provider activity.

---

# 17. Customer-Provider Interaction Features

Fraud risk can depend on the interaction between a specific customer and provider rather than either entity independently.

Representative features include:

```text
customer_provider_claims_30d
days_since_same_provider_claim
customer_provider_intensity
```

These variables capture repeated or unusually concentrated customer-provider activity.

---

# 18. Repeated-Service Features

Repeated healthcare services can represent either legitimate recurring treatment or suspicious activity.

Representative features include:

```text
same_service_claims_30d
same_service_intensity
```

The model must therefore distinguish suspicious repetition from legitimate repeated care.

---

# 19. Relative Amount Features

Relative amount features compare the current claim with historical reference populations.

Important examples include:

```text
claim_to_service_median_ratio
claim_to_customer_avg_ratio
claim_to_provider_avg_ratio
amount_above_service_typical
```

These variables are often more informative than absolute claim amount because fraud risk depends on context.

For example:

```text
claim_to_service_median_ratio
```

compares the current claim amount with the historical median amount associated with the corresponding service.

This is the strongest business-level driver observed in the final SHAP analysis.

---

# 20. Behavioural Ratio Features

Additional engineered behavioural ratios include:

```text
recent_claim_share_30d_365d
recent_amount_share_30d_365d
provider_recent_activity_ratio
customer_provider_intensity
same_service_intensity
```

These variables measure whether recent activity is unusually concentrated relative to longer-term historical behaviour.

---

# 21. Target Variable

## `is_fraud`

Binary supervised-learning target.

Conceptually:

```text
0 = legitimate synthetic claim
1 = fraudulent synthetic claim
```

The target is used only for:

- model training;
- validation;
- final evaluation;
- error analysis.

It is explicitly excluded from the predictive feature matrix.

Final out-of-time test prevalence:

```text
2.885%
```

---

# 22. Synthetic Fraud Diagnostic Variables

The synthetic generator contains diagnostic variables used to control or describe fraud generation.

Examples include:

```text
latent_fraud_score
synthetic_fraud_probability
fraud_difficulty
fraud_mechanism
legitimate_anomaly
legitimate_anomaly_type
```

These variables are useful for:

- synthetic data generation;
- model diagnostics;
- fraud-mechanism analysis;
- false-positive analysis;
- false-negative analysis;
- difficulty analysis.

They are **not legitimate predictive inputs** because they contain information derived from the synthetic fraud-generation process.

They are therefore excluded from the model feature matrix.

---

# 23. Fraud Mechanism

## `fraud_mechanism`

Synthetic diagnostic variable describing the mechanism used to generate a fraudulent observation.

Mechanisms evaluated in the final model include:

```text
customer_provider_pattern
repeated_service
mixed_pattern
frequency_abuse
provider_abnormality
amount_inflation
```

This variable is used for post-model error analysis only.

It must never be used as a predictive feature.

---

# 24. Fraud Difficulty

## `fraud_difficulty`

Synthetic diagnostic label describing how strongly the fraud pattern is expressed.

Values include:

```text
easy
medium
hard
```

This variable supports evaluation of model performance across different fraud difficulty levels.

It is excluded from model training.

---

# 25. Legitimate Anomalies

## `legitimate_anomaly`

Synthetic diagnostic variable identifying unusual but legitimate claims.

## `legitimate_anomaly_type`

Examples include:

```text
repeated_service_legitimate
unusual_provider_legitimate
high_frequency_legitimate
high_amount_legitimate
```

These variables are particularly useful for false-positive analysis.

They are excluded from predictive modelling.

---

# 26. Referential Integrity

The validation pipeline checks relationships between:

```text
claims
customers
providers
policies
```

After cleaning, the modelling population passes blocking referential-integrity validation.

Missing provider identifiers belonging to the controlled synthetic missingness mechanism are handled separately from invalid foreign-key references.

---

# 27. Data-Quality Rules

Important blocking rules include:

| Rule | Requirement |
|---|---|
| Claim amount | `claim_amount > 0` |
| Requested reimbursement | `requested_reimbursement <= claim_amount` |
| Service units | `service_units >= 1` |
| Service/submission chronology | `service_date <= claim_submission_date` |
| Referential integrity | Valid relationships between datasets |

Controlled missingness generates warnings rather than blocking errors.

---

# 28. Controlled Missingness

The cleaned dataset intentionally preserves selected missing values.

Observed after cleaning:

| Variable | Missing Values |
|---|---:|
| `provider_id` | 1,024 |
| `has_prescription` | 2,987 |
| `document_count` | 464 |

These values are preserved because missingness itself can represent realistic information availability.

The model feature pipeline handles this explicitly.

---

# 29. Feature Contract

The frozen model uses a fixed feature contract stored with the model metadata.

Model metadata:

```text
artifacts/metadata/health_fraud_model_metadata.json
```

The production scoring layer validates incoming data against this contract before prediction.

The final business-level feature set contains:

```text
57 features
```

After preprocessing and categorical encoding, the final transformed model matrix contains:

```text
107 features
```

---

# 30. Model Artifacts

The frozen inference artifacts are stored in:

```text
artifacts/models/health_fraud_xgboost.joblib
artifacts/preprocessors/health_fraud_preprocessor.joblib
artifacts/metadata/health_fraud_model_metadata.json
```

The model and preprocessor must be treated as a matched artifact set.

---

# 31. Prediction Outputs

Batch scoring produces:

```text
artifacts/predictions/claim_scores.parquet
```

containing fraud-risk scores for scored claims.

The investigation-prioritization output is stored in:

```text
artifacts/predictions/top_review_claims.parquet
```

The primary operational policy selects approximately the top:

```text
3%
```

of claims ranked by fraud risk.

---

# 32. Important Modelling Restrictions

The following information must not enter the predictive feature matrix:

```text
is_fraud
fraud_mechanism
fraud_difficulty
latent_fraud_score
synthetic_fraud_probability
legitimate_anomaly
legitimate_anomaly_type
```

More generally:

> Any field generated from knowledge of the synthetic fraud label or future information must be excluded from model inputs.

This restriction is fundamental to preventing target and temporal leakage.

---

# 33. Data Disclaimer

All datasets in this repository are synthetic.

They are designed to support:

- machine-learning experimentation;
- fraud-risk ranking;
- data-engineering demonstrations;
- explainability analysis;
- API development;
- deployment demonstrations.

They do not represent real customers, healthcare providers, insurance contracts or insurance claims.

Real-world deployment would require a new data dictionary based on the actual production schema, data lineage, regulatory requirements and governance controls.