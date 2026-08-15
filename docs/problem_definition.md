# Health Insurance Fraud Risk Detection
## Problem Definition

## 1. Business Context

Health insurers process large volumes of reimbursement claims across different healthcare services such as consultations, dental care, optical care, pharmacy, physiotherapy, medical devices, and other covered treatments.

Manual review of every claim is neither operationally feasible nor economically efficient. Fraud investigation teams therefore need a way to identify and prioritize claims presenting unusual or potentially fraudulent patterns.

The objective of this project is not to automatically determine whether fraud has occurred.

Instead, the system is designed as a **decision-support tool** that estimates fraud risk, explains the main risk signals, and helps investigators prioritize claims for manual review.

---

## 2. Business Question

The central business question is:

> Among newly submitted health insurance reimbursement claims, which claims should be investigated first, and why?

The project therefore addresses both:

1. **Risk estimation** — How likely is a newly submitted claim to be fraudulent?
2. **Investigation prioritization** — Given limited investigation capacity, which claims should be reviewed first?

---

## 3. Project Scope

### In scope

The first version focuses exclusively on **supplementary health insurance reimbursement claims**.

Potential healthcare service categories include:

- medical consultations;
- dental care;
- optical care;
- physiotherapy;
- pharmacy expenses;
- medical devices;
- diagnostic services;
- other eligible healthcare services.

### Out of scope

The project does not cover:

- motor insurance;
- home insurance;
- automatic claim rejection;
- automatic accusations of fraud;
- legal determination of fraud;
- automatic sanctions against customers or providers;
- real customer data;
- real healthcare provider data.

All data used in this project are synthetic.

---

## 4. Unit of Analysis

The statistical unit is a **health insurance reimbursement claim**.

Each observation corresponds to one submitted claim.

The main entities are:

- `claim_id` — reimbursement claim;
- `customer_id` — insured person;
- `policy_id` — insurance contract;
- `provider_id` — healthcare provider.

These relationships are important because:

- one customer may submit multiple claims;
- one policy may generate multiple claims;
- one provider may appear across many customers and claims.

Historical and relational behavior can therefore contain useful fraud-risk information.

---

## 5. Prediction Point

A fundamental requirement is to define exactly **when the prediction is made**.

For this project:

> Fraud risk is estimated immediately after the initial reimbursement claim has been submitted and validated technically, but before any dedicated fraud investigation.

We define:

`prediction_timestamp = claim_submission_timestamp`

Only information available at or before this timestamp may be used by the model.

For every candidate feature:

`feature_timestamp <= prediction_timestamp`

This prevents the model from learning from information that would not exist when a real prediction is required.

---

## 6. Target Definition

The supervised-learning target is:

`is_fraud`

with:

- `is_fraud = 1`: the claim is eventually confirmed as fraudulent;
- `is_fraud = 0`: the claim is considered legitimate within the observation process.

The model estimates:

`fraud_probability = P(is_fraud = 1 | information available at prediction time)`

The output represents **estimated risk**, not proof that fraud occurred.

---

## 7. Target and Label Limitations

Fraud labels are inherently imperfect in real insurance systems.

A claim recorded as legitimate does not necessarily prove that no fraud occurred. Some fraudulent behavior may remain undetected.

Historical labels may also suffer from **selection bias** because claims previously investigated were generally not selected randomly.

Important real-world limitations therefore include:

- undetected fraud;
- label noise;
- delayed fraud confirmation;
- selective investigation bias;
- partially observed outcomes.

Because this project uses synthetic data, the underlying target is known by construction. Nevertheless, these real-world limitations remain part of the methodological analysis.

---

## 8. Information Available for Prediction

Potential information can be divided into several families.

### Claim information

Examples:

- healthcare service category;
- claim amount;
- reimbursement amount requested;
- service date;
- submission date;
- number of units or sessions;
- submission channel.

### Customer information

Examples:

- policy tenure;
- historical claim frequency;
- historical reimbursement amounts;
- number of providers previously visited;
- time since previous claim;
- recent changes in claiming behavior.

### Policy information

Examples:

- coverage type;
- reimbursement limits;
- contract tenure;
- recent contract modifications.

### Provider information

Examples:

- historical claim volume;
- typical amount by healthcare service;
- number of customers associated with the provider;
- changes in claim activity;
- historical risk indicators available before the current claim.

### Temporal and behavioral information

Examples:

- claims during the previous 7, 30, 90 and 365 days;
- repeated services over short periods;
- deviation from customer historical behavior;
- deviation from provider historical behavior;
- repeated customer-provider interactions.

All historical features must be calculated using **past information only**.

---

## 9. Leakage Prevention

Variables created after fraud investigation must never be included as model predictors.

Examples of prohibited variables include:

- investigation result;
- investigator fraud score;
- fraud confirmation date;
- final investigation status;
- recovery amount;
- legal action;
- sanction outcome;
- post-investigation expert conclusions.

These variables would reveal information unavailable at prediction time and create **target leakage**.

Leakage prevention will be treated as a first-class requirement throughout the project.

---

## 10. Fraud Mechanisms

Synthetic fraud will not be generated using a single deterministic rule.

Instead, fraud probability will depend on combinations of weak and noisy signals.

Potential mechanisms include:

### Abnormal claim frequency

An unusually large number of claims over a short period may increase risk.

High frequency alone must not imply fraud.

### Amount anomalies

Claims substantially above the typical amount for the same healthcare service may present increased risk.

### Repeated services

Repeated reimbursement requests for similar treatments within unusually short intervals may constitute a risk signal.

### Provider anomalies

Potential signals include:

- unusually rapid increase in claim volume;
- atypically high amounts;
- unusual concentration of claims;
- abnormal activity relative to provider history.

### Customer-provider interaction anomalies

Risk may increase when several weak signals occur simultaneously.

For example:

`high customer claim frequency`

+

`unusual provider activity`

+

`abnormal claim amount`

+

`repeated customer-provider interactions`

No single variable should perfectly determine the target.

---

## 11. Class Imbalance

Fraud is expected to represent only a small proportion of all reimbursement claims.

The synthetic dataset will therefore reproduce a strongly imbalanced classification problem.

A preliminary fraud prevalence of approximately **2–3%** will be considered during data-generation design.

The final prevalence will be configurable and validated after generation.

Because of this imbalance, **accuracy will not be used as the primary model-selection metric**.

---

## 12. Temporal Validation

The model will ultimately be used to score future claims using historical information.

The primary validation strategy will therefore preserve temporal order:

`TRAIN → VALIDATION → TEST`

where:

- training contains the oldest observations;
- validation contains a later period;
- test contains the most recent held-out period.

A random train/test split will not be the primary validation strategy.

The test period will remain untouched during model development and model selection.

---

## 13. Model Output

For each claim, the model produces:

`fraud_probability ∈ [0, 1]`

Example:

`fraud_probability = 0.82`

This probability represents estimated fraud risk.

Three separate properties will eventually be evaluated:

1. discrimination;
2. ranking quality;
3. probability calibration.

---

## 14. Explainability

Investigators must understand why a claim has been prioritized.

Technical explainability methods such as SHAP may be used internally.

However, the final application should translate technical explanations into understandable business signals.

Example:

**Fraud risk: 82%**

Main risk indicators:

- unusually high amount for this service;
- repeated similar claims over a short period;
- provider activity above historical baseline.

Explainability supports investigation. It does not constitute evidence of fraud.

---

## 15. Business Prioritization

Fraud probability and investigation priority are different concepts.

The model produces:

`fraud_probability`

The business layer produces:

`investigation_priority`

Prioritization may eventually incorporate:

- fraud probability;
- financial exposure;
- investigation capacity;
- investigation cost;
- potentially recoverable amount.

A simple candidate quantity is:

`expected_fraud_exposure = fraud_probability × claim_amount`

This is only a starting hypothesis and will be evaluated rather than assumed to be the optimal business rule.

---

## 16. Investigation Capacity

Fraud investigators have limited review capacity.

The operational problem is therefore not simply:

> Is fraud probability greater than 0.5?

A more realistic question is:

> If only K claims can be investigated, which K claims should be selected?

The value of `K` will remain configurable.

This motivates ranking-oriented evaluation.

---

## 17. Evaluation Strategy

### Machine-learning metrics

The project will evaluate:

- PR-AUC;
- precision;
- recall;
- F1 score;
- ROC-AUC;
- confusion matrix;
- probability calibration.

Given class imbalance, PR-AUC will receive particular attention.

### Operational metrics

The project will additionally evaluate:

- Precision@K;
- Recall@K;
- fraud amount captured @K;
- expected financial value captured under fixed investigation capacity.

---

## 18. Error Analysis

Model performance will not be reduced to a single metric.

False positives and false negatives will be explicitly investigated.

Questions include:

- Which fraudulent claims are missed?
- Which legitimate claims are incorrectly prioritized?
- Which fraud mechanisms are hardest to detect?
- Does performance vary by healthcare service?
- Does performance deteriorate over time?
- Are certain providers or customer segments disproportionately flagged?
- Is calibration reliable in the high-risk region?

---

## 19. Human-in-the-Loop

The model is a **decision-support system**.

The final workflow is:

`Risk estimation → Prioritization → Human investigation → Final decision`

A high fraud probability must never be interpreted as proof that a customer or healthcare provider committed fraud.

The final operational decision remains with the investigator.

---

## 20. Synthetic Data

No proprietary insurance dataset is available for this project.

Synthetic data will therefore be generated programmatically.

The generator will explicitly model:

- customers;
- policies;
- healthcare providers;
- healthcare services;
- reimbursement claims;
- temporal behavior;
- claim amounts;
- fraud mechanisms;
- class imbalance;
- noise and overlap between fraudulent and legitimate behavior.

Generation will be reproducible through a fixed random seed and configuration files.

Synthetic data are used to demonstrate the methodology and architecture of the solution.

They must **not** be used to claim real-world production performance.

---

## 21. Success Criteria

The project should demonstrate the ability to:

1. define the fraud problem from a business perspective;
2. generate coherent and reproducible synthetic health-claim data;
3. prevent temporal and target leakage;
4. engineer meaningful behavioral and temporal features;
5. handle severe class imbalance appropriately;
6. compare interpretable baselines with stronger ML models;
7. evaluate discrimination, ranking and calibration separately;
8. analyze model errors;
9. explain individual predictions;
10. convert fraud scores into investigation priorities;
11. expose the model through a clean API;
12. present results through a business-facing application;
13. document limitations and scaling constraints.

---

## 22. Final Problem Statement

> **Build a decision-support system for supplementary health insurance fraud detection that estimates the probability of fraud for newly submitted reimbursement claims using only information available at claim submission time, explains the main risk signals, and prioritizes claims for human investigation under limited review capacity.**