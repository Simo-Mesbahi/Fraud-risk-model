# Model Card — Health Insurance Fraud Risk Model

## 1. Model Overview

| Field | Value |
|---|---|
| Model | XGBoost binary classifier |
| Task | Health insurance claim fraud risk scoring |
| Output | Fraud probability / risk score |
| Operational strategy | Rank claims by predicted fraud risk |
| Primary review policy | Top 3% highest-risk claims |
| Evaluation design | Out-of-time temporal validation |
| Final test period | 2026-01-01 → 2026-06-30 |
| Final test claims | 14,176 |
| Test fraud prevalence | 2.885% |
| Dataset | Synthetic health insurance claims |
| Status | Portfolio / prototype — not production approved |

---

## 2. Intended Use

The model estimates the probability that a health insurance claim is fraudulent.

Its primary purpose is **fraud-investigation prioritization** rather than automatic claim rejection.

The expected operational workflow is:

1. receive an insurance claim;
2. construct historical and contextual features using information available at scoring time;
3. generate a fraud-risk probability;
4. rank claims by predicted risk;
5. send the highest-risk claims to fraud investigators;
6. use human investigation to determine the appropriate action.

The model therefore acts as a **decision-support and triage system**.

It should not independently determine whether a customer, provider, or claim is fraudulent.

---

## 3. Business Objective

Fraud detection is a highly imbalanced classification problem.

Only approximately 2.5–3% of claims in the simulated environment are fraudulent. Reviewing every claim would therefore be operationally inefficient.

The objective is consequently not simply to maximize classification accuracy.

The primary business objective is:

> Concentrate as much fraud as possible inside a small investigation capacity.

The main operating point evaluated in this project assumes that investigators can review approximately:

**3% of submitted claims.**

Performance is therefore evaluated using ranking-oriented metrics such as:

- Average Precision;
- Precision@3%;
- Recall@3%;
- Lift@3%;
- fraud amount capture@3%.

---

## 4. Dataset

The project uses a synthetic health-insurance claims environment designed specifically for fraud-model development.

The generation process contains:

- customers;
- insurance policies;
- healthcare providers;
- healthcare services;
- claim amounts;
- reimbursement information;
- submission information;
- historical customer behaviour;
- historical provider behaviour;
- customer-provider interactions;
- legitimate anomalous behaviour;
- multiple fraud mechanisms;
- multiple fraud difficulty levels;
- missing values;
- intentionally invalid records;
- temporal drift.

The generated claims table initially contains approximately:

**100,000 claims and 55 columns.**

After data-quality cleaning:

**99,911 valid claims remain.**

Invalid synthetic records are preserved separately rather than silently corrected.

---

## 5. Data Quality

The raw synthetic dataset intentionally contains a small number of invalid records to reproduce realistic data-engineering conditions.

Examples include:

- non-positive claim amounts;
- invalid service-unit counts;
- service dates occurring after claim submission.

The cleaning pipeline rejected:

**89 claims.**

After cleaning, validation reported:

**0 blocking data-quality errors.**

Controlled missingness remains intentionally present in fields such as:

- `provider_id`;
- `has_prescription`;
- `document_count`.

These values are treated as expected missingness rather than data corruption.

---

## 6. Temporal Evaluation Strategy

Random train/test splitting was deliberately avoided.

Insurance fraud models operate on future claims, so evaluation should reproduce that deployment condition.

The dataset is split chronologically.

### Training period

2023-01-01 → 2025-06-30

### Validation period

2025-07-01 → 2025-12-31

### Final out-of-time test period

2026-01-01 → 2026-06-30

The final test set contains:

**14,176 claims**

including:

**409 fraudulent claims.**

This temporal design provides a more realistic estimate of future generalization than a random split.

---

## 7. Leakage Control

Several synthetic variables are generated for simulation, diagnostics, or evaluation purposes only.

They must never be provided to the predictive model.

Examples include:

- `latent_fraud_score`;
- `synthetic_fraud_probability`;
- `fraud_difficulty`;
- `fraud_mechanism`;
- `legitimate_anomaly`;
- `legitimate_anomaly_type`.

The target:

- `is_fraud`

is also excluded from the feature matrix.

Identifiers are not treated as predictive business features.

Historical features are constructed using information preceding the current claim in order to reduce temporal leakage.

---

## 8. Feature Families

The model uses information available or derivable at claim-scoring time.

Important feature families include:

### Claim characteristics

Examples:

- claim amount;
- requested reimbursement;
- reimbursement ratio;
- service category;
- service units;
- submission channel;
- documentation information.

### Customer characteristics

Examples:

- customer age;
- customer tenure;
- coverage level;
- customer behaviour segment.

### Policy characteristics

Examples:

- policy tenure;
- recent policy changes;
- time since policy change.

### Provider characteristics

Examples:

- provider type;
- provider region;
- provider tenure;
- provider behaviour segment.

### Historical behavioural features

Examples:

- customer claims over 7 / 30 / 90 / 365 days;
- customer historical claim amounts;
- time since previous customer claim;
- time since previous claim with the same provider;
- repeated service activity;
- customer-provider interaction frequency;
- provider claims over recent windows;
- provider historical claim amounts.

### Relative anomaly features

Examples:

- claim-to-service median ratio;
- claim-to-customer historical average ratio;
- claim-to-provider historical average ratio.

These relative features are particularly useful because fraud often depends on deviation from expected behaviour rather than absolute amount alone.

---

## 9. Candidate Models

Several model families were evaluated.

The baseline experiments included:

- Dummy classifier;
- Logistic Regression;
- class-balanced Logistic Regression;
- Random Forest;
- XGBoost.

The Dummy model establishes the performance expected without useful predictive information.

Logistic Regression provides an interpretable linear baseline.

Random Forest and XGBoost evaluate nonlinear relationships and feature interactions.

Model selection was performed on the validation period only.

The final test period was kept separate until the champion model had been frozen.

---

## 10. Champion Model

The selected final model is:

**XGBoost**

The complete preprocessing pipeline is fitted on the combined training and validation periods after model selection.

The frozen champion is then evaluated once on the untouched 2026 test period.

Final transformed matrix dimensions:

- training/validation matrix: **85,735 × 107**
- test matrix: **14,176 × 107**

---

## 11. Final Out-of-Time Performance

### Global discrimination and probability metrics

| Metric | Test result |
|---|---:|
| Average Precision | **0.5520** |
| ROC-AUC | **0.8518** |
| Brier Score | **0.0174** |
| Log Loss | **0.0797** |

Because fraud prevalence is low, Average Precision is particularly informative.

The model's AP of **0.5520** is substantially above the approximately **2.9% fraud prevalence** of the test population.

---

## 12. Operational Performance — Top 3% Review Policy

The primary business operating point reviews the highest-risk 3% of claims.

| Metric | Result |
|---|---:|
| Claims reviewed | **426** |
| Precision@3% | **51.64%** |
| Recall@3% | **53.79%** |
| Lift@3% | **17.90×** |
| Fraud amount captured@3% | **55.15%** |

This means that reviewing only approximately **3% of claims** identifies approximately:

- **53.8% of fraudulent claims**;
- **55.1% of fraudulent claim amount**.

Approximately one out of every two claims investigated at this operating point is fraudulent in the synthetic test environment.

This represents a substantial concentration of fraud compared with untargeted review.

---

## 13. Review-Capacity Trade-off

Performance was evaluated across several investigation capacities.

| Review rate | Precision | Recall | Lift | Fraud amount captured |
|---:|---:|---:|---:|---:|
| 0.5% | 98.59% | 17.11% | 34.17× | 16.93% |
| 1% | 94.37% | 32.76% | 32.71× | 32.83% |
| 2% | 68.31% | 47.43% | 23.68× | 48.66% |
| **3%** | **51.64%** | **53.79%** | **17.90×** | **55.15%** |
| 5% | 34.56% | 59.90% | 11.98× | 60.80% |
| 7.5% | 24.81% | 64.55% | 8.60× | 68.74% |
| 10% | 19.46% | 67.48% | 6.75× | 72.44% |
| 15% | 13.82% | 71.88% | 4.79× | 76.69% |

This illustrates the operational trade-off between investigation workload and fraud capture.

The 3% threshold is therefore a **business operating point**, not an intrinsic statistical threshold.

---

## 14. Confusion Profile at the 3% Operating Point

At the top-3% review threshold:

| Outcome | Claims |
|---|---:|
| True positives | **220** |
| False positives | **206** |
| False negatives | **189** |
| True negatives | **13,561** |

The model should therefore be interpreted primarily as a ranking system.

A fixed probability threshold such as 0.50 is not necessarily the appropriate production decision rule for a capacity-constrained fraud-investigation workflow.

---

## 15. Validation-to-Test Stability

| Metric | Validation | Test | Difference |
|---|---:|---:|---:|
| Average Precision | 0.4846 | 0.5520 | +0.0674 |
| ROC-AUC | 0.8451 | 0.8518 | +0.0067 |
| Brier Score | 0.0183 | 0.0174 | -0.0009 |
| Precision@3% | 46.53% | 51.64% | +5.12 pp |
| Recall@3% | 51.28% | 53.79% | +2.51 pp |
| Lift@3% | 17.08× | 17.90× | +0.82× |
| Fraud amount capture@3% | 49.07% | 55.15% | +6.08 pp |

No major degradation is observed between validation and the final out-of-time test period.

The results nevertheless originate from the same synthetic simulation framework and must not be interpreted as evidence of real-world production stability.

---

## 16. Explainability

SHAP is used to analyse the frozen XGBoost model.

The explainability analysis covers:

- global feature importance;
- business-level feature aggregation;
- local claim explanations;
- true positives;
- false positives;
- false negatives;
- legitimate anomalies;
- difficult fraud cases;
- mechanism-specific failures.

SHAP explanations were calculated on a representative sample of:

**5,000 test claims.**

The transformed model contains:

**107 features.**

---

## 17. Main Business Drivers

The five strongest business-level model drivers observed in the final explainability analysis are:

1. `claim_to_service_median_ratio`
2. `days_since_customer_previous_claim`
3. `reimbursement_ratio`
4. `provider_claims_30d`
5. `submission_month`

The strongest driver is:

**`claim_to_service_median_ratio`**

This indicates that deviation of a claim amount from the historical amount expected for that service is an important risk signal.

SHAP importance describes model behaviour and must not be interpreted as causal evidence.

---

## 18. Fraud Mechanism Performance

Detection quality varies materially across synthetic fraud mechanisms.

At the 3% review operating point:

| Fraud mechanism | Recall@3% |
|---|---:|
| customer-provider pattern | **80.30%** |
| repeated service | **68.97%** |
| mixed pattern | **61.76%** |
| frequency abuse | **57.75%** |
| provider abnormality | **53.52%** |
| amount inflation | **8.00%** |

The main model weakness is therefore:

**`amount_inflation`.**

Among missed fraudulent claims, `amount_inflation` accounts for:

**69 false negatives.**

This limitation should be explicitly monitored rather than hidden behind aggregate model metrics.

---

## 19. Fraud Difficulty

Performance also varies according to the synthetic fraud-difficulty label.

| Difficulty | Median risk score | Recall@3% |
|---|---:|---:|
| Easy | 0.6707 | **67.26%** |
| Medium | 0.2050 | **55.45%** |
| Hard | 0.0261 | **28.95%** |

The expected ordering is preserved:

**easy > medium > hard**

in terms of detectability.

Hard fraud therefore remains a major residual risk.

---

## 20. False Positive Analysis

At the 3% review operating point:

**206 legitimate claims are flagged for investigation.**

Among these false positives:

**57.77% correspond to simulated legitimate anomalies.**

Important categories include:

- repeated legitimate services;
- unusual but legitimate provider behaviour;
- legitimate high-frequency activity;
- legitimate high-amount claims.

This is operationally meaningful.

Many model false positives are not arbitrary normal claims; they are legitimate behaviours that resemble fraud patterns.

Human investigation therefore remains essential.

---

## 21. False Negative Analysis

At the same operating point:

**189 fraudulent claims are not selected for review.**

The largest missed-fraud category is:

**amount inflation — 69 claims.**

Other missed mechanisms include:

- provider abnormality;
- frequency abuse;
- mixed patterns;
- repeated services;
- customer-provider patterns.

False negatives should be monitored continuously because aggregate discrimination metrics can conceal mechanism-specific weaknesses.

---

## 22. Uncertainty

Bootstrap confidence intervals were estimated on the final test set.

### Average Precision

Estimate:

**0.5520**

95% bootstrap interval:

**[0.5036, 0.6029]**

### ROC-AUC

Estimate:

**0.8518**

95% bootstrap interval:

**[0.8235, 0.8782]**

These intervals quantify sampling uncertainty within the synthetic test population.

They do not account for uncertainty caused by differences between synthetic and real insurance data.

---

## 23. Known Limitations

### Synthetic data

The most important limitation is that the complete modelling environment is synthetic.

Real-world healthcare insurance data may contain:

- different fraud mechanisms;
- stronger behavioural heterogeneity;
- coding inconsistencies;
- missing data mechanisms not reproduced here;
- regulatory constraints;
- provider networks;
- geographic effects;
- coordinated fraud rings;
- investigation feedback loops;
- changing fraud strategies.

Reported performance must therefore not be interpreted as expected real-world production performance.

### Amount-inflation weakness

The model has limited sensitivity to the synthetic `amount_inflation` mechanism.

This is the primary observed model weakness.

### Hard fraud

Hard fraud receives substantially lower model scores than easy and medium fraud.

### Concept drift

Fraud behaviour changes over time.

The synthetic environment intentionally contains drift, but real-world drift can be substantially more complex.

### False positives

Legitimate unusual behaviour can resemble fraudulent activity.

Automatic rejection based solely on the model score would therefore create unacceptable operational risk.

### Explainability

SHAP explains the behaviour of the predictive model.

It does not prove why fraud occurred and should not be interpreted causally.

---

## 24. Prohibited Uses

The model should **not** be used to:

- automatically accuse a customer or provider of fraud;
- automatically reject reimbursement solely because of the model score;
- make legal conclusions;
- replace fraud investigators;
- infer criminal intent;
- operate on real customers without independent validation and governance review.

A high risk score means:

> the claim exhibits patterns that the model associates with fraudulent claims.

It does not establish that fraud occurred.

---

## 25. Human Oversight

The recommended operational architecture is:

**Model → risk ranking → investigation queue → human review → decision**

Investigators should have access to:

- the claim;
- relevant historical information;
- key risk drivers;
- contextual information;
- model explanation;
- applicable business rules.

The final decision should remain outside the predictive model.

---

## 26. Monitoring Requirements

A production implementation should monitor at least four dimensions.

### Data quality

Monitor:

- missing-value rates;
- invalid values;
- categorical-domain changes;
- schema changes;
- unexpected distributions.

### Data drift

Monitor changes in important features such as:

- claim amount;
- reimbursement ratio;
- service mix;
- provider activity;
- customer activity;
- claim-to-service ratio.

### Model performance

Once labels become available, monitor:

- Average Precision;
- ROC-AUC;
- Precision@K;
- Recall@K;
- Lift@K;
- fraud amount captured;
- calibration.

### Operational performance

Monitor:

- review volume;
- investigation capacity;
- confirmed-fraud yield;
- false-positive burden;
- fraud amount recovered or prevented;
- performance by fraud mechanism when available.

---

## 27. Drift and Retraining

Retraining should not be triggered solely by a fixed calendar schedule.

A retraining review should be initiated when there is evidence of:

- material feature drift;
- fraud-prevalence changes;
- deterioration in Precision@K or Recall@K;
- declining lift;
- calibration deterioration;
- emerging fraud patterns;
- provider-network changes;
- changes in policy or reimbursement rules.

Every retrained model should undergo the same temporal validation and governance process before replacing the deployed champion.

---

## 28. Model Artifacts

The final modelling pipeline exports:

```text
artifacts/models/health_fraud_xgboost.joblib
artifacts/preprocessors/health_fraud_preprocessor.joblib
artifacts/metadata/health_fraud_model_metadata.json