# Fraud Risk Decision Support Platform

End-to-end machine learning platform for **health insurance fraud investigation prioritization**, combining temporal feature engineering, fraud-risk scoring, operational ranking, TreeSHAP explainability, REST inference, an investigation-oriented web application, Docker deployment and automated testing.

> **Project scope:** technical prototype built on synthetic health-insurance data generated programmatically through the project's configurable data-generation pipeline.  
> The system is designed to support human fraud-investigation prioritization by scoring, ranking and explaining claims.
---

## Executive Summary

Fraud investigation teams operate under limited review capacity. The objective of this project is therefore not simply to classify claims as fraudulent or legitimate, but to **rank incoming claims by fraud risk and concentrate investigations where they are expected to create the most value**.

The platform implements the complete decision-support workflow:

```text
Claims
  ↓
Data Validation
  ↓
Temporal & Behavioural Features
  ↓
XGBoost Fraud Risk Model
  ↓
Risk Ranking
  ↓
Top-Risk Investigation Queue
  ↓
TreeSHAP Explanation
  ↓
Human Investigation
```

The final model is evaluated on a fully held-out **out-of-time test period from January to June 2026**.

At the selected **3% investigation capacity**, the system captures approximately **53.8% of fraudulent claims** and **55.1% of fraudulent claim amount**, with a **17.9× lift** over untargeted review.

---

## Live Demo

**Application:** `LIVE_DEMO_URL`

The application exposes the complete decision-support workflow, including portfolio monitoring, individual claim analysis, investigation prioritization, batch scoring, model insights and runtime status.

---

## 1. Business Problem

Health-insurance fraud is a highly imbalanced detection problem.

When only a small proportion of submitted claims can be investigated manually, reviewing every claim is operationally unrealistic and conventional classification accuracy becomes a poor optimization objective.

This project therefore formulates fraud detection as a:

> **risk-ranking and investigation-prioritization problem**

Rather than applying an arbitrary probability threshold such as `0.50`, claims are ranked by predicted fraud risk and investigators focus on the highest-risk portion of the portfolio.

The operating policy used for final evaluation assumes capacity to review **3% of submitted claims**.

---

## 2. Solution Overview

The platform covers the ML lifecycle from data generation to investigator-facing inference:

- configurable synthetic health-insurance data generation;
- data-quality validation and invalid-record isolation;
- leakage-controlled temporal feature engineering;
- temporal train / validation / test splitting;
- comparison of multiple model families;
- frozen XGBoost champion model;
- out-of-time model evaluation;
- operational capacity analysis;
- TreeSHAP global and local explainability;
- reusable inference layer;
- FastAPI REST service;
- Streamlit decision-support interface;
- Docker / Docker Compose deployment;
- automated unit and integration testing;
- GitHub Actions continuous integration.

---

## 3. Key Results

### Final out-of-time test performance

| Metric | Result |
|---|---:|
| Test claims | **14,176** |
| Fraud cases | **409** |
| Fraud prevalence | **2.885%** |
| Average Precision | **0.5520** |
| ROC-AUC | **0.8518** |
| Brier Score | **0.0174** |
| Log Loss | **0.0797** |
| Precision @ 3% | **51.64%** |
| Recall @ 3% | **53.79%** |
| Lift @ 3% | **17.90×** |
| Fraud amount captured @ 3% | **55.15%** |

Because fraud prevalence is approximately **2.9%**, Average Precision is treated as a primary discrimination metric alongside operational capacity metrics.

The results are obtained on synthetic data and should therefore be interpreted as evidence of the **methodology and system design**, not as estimates of real-world insurance performance.

---

## 4. Operational Decision Policy — Top 3% Review Capacity

The model produces a continuous fraud-risk score.

Claims are sorted from highest to lowest risk and the top fraction is forwarded for investigation.

At a **3% review capacity**:

| Operational measure | Result |
|---|---:|
| Claims reviewed | **426** |
| True positives | **220** |
| False positives | **206** |
| Fraud recall | **53.79%** |
| Investigation precision | **51.64%** |
| Lift | **17.90×** |
| Fraud amount captured | **55.15%** |

The 3% operating point is a **business capacity assumption**, not an intrinsic statistical threshold.

It can be adjusted according to available investigation resources and the desired trade-off between workload and fraud capture.

---

## 5. Application Capabilities

The Streamlit application provides six operational views:

### Overview
Executive monitoring of portfolio risk, model performance and investigation capacity.

### Claim Analysis
Individual claim scoring with risk assessment and TreeSHAP-based explanation.

### Investigation Queue
Prioritized access to the highest-risk claims for manual investigation.

### Portfolio Scoring
Batch scoring and portfolio-level risk ranking.

### Model Insights
Model performance, feature importance and explainability artifacts.

### System Status
Runtime model identity, API availability and inference/explainability status.

The frontend consumes the same inference API used for programmatic scoring, avoiding a separate UI-only prediction implementation.

---

## 6. System Architecture

```text
                    ┌─────────────────────────┐
                    │ Synthetic Claims Data   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ Data Quality Validation │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ Temporal Feature        │
                    │ Engineering             │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ Model Experimentation   │
                    │ & Champion Selection    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ Frozen XGBoost Model    │
                    │ + Preprocessor          │
                    └────────────┬────────────┘
                                 │
             ┌───────────────────┴───────────────────┐
             │                                       │
   ┌─────────▼─────────┐                   ┌─────────▼─────────┐
   │ TreeSHAP          │                   │ FraudScorer       │
   │ Explainability    │                   │ Inference Layer   │
   └───────────────────┘                   └─────────┬─────────┘
                                                    │
                                          ┌─────────▼─────────┐
                                          │ FastAPI REST API  │
                                          └─────────┬─────────┘
                                                    │
                                          ┌─────────▼─────────┐
                                          │ Streamlit UI      │
                                          └───────────────────┘
```

The frozen preprocessing and feature contracts used during final evaluation are reused at inference time.

---

## 7. Machine Learning Methodology

### Temporal validation

Random train/test splitting is deliberately avoided.

A deployed fraud model learns from historical claims and scores future claims, so the evaluation reproduces this temporal direction.

| Split | Period |
|---|---|
| Training | 2023-01-01 → 2025-06-30 |
| Validation | 2025-07-01 → 2025-12-31 |
| Final out-of-time test | 2026-01-01 → 2026-06-30 |

The final test period remains untouched during model selection.

### Candidate models

The experimentation stage compares:

- Dummy classifier;
- Logistic Regression;
- class-balanced Logistic Regression;
- Random Forest;
- XGBoost.

The final champion is:

**XGBoost — version 1.0.0**

The production feature contract contains **57 source features**, transformed into **107 model features** by the frozen preprocessing pipeline.

---

## 8. Feature Engineering

The model combines claim-level information with historical behavioural context.

Major feature families include:

- claim and reimbursement characteristics;
- customer history;
- policy history;
- provider behaviour;
- recent claim frequency;
- recent monetary activity;
- customer-provider interactions;
- repeated-service activity;
- temporal submission patterns;
- relative anomaly indicators.

Examples of engineered signals include:

```text
claim_to_service_median_ratio
claim_to_customer_avg_ratio
claim_to_provider_avg_ratio
requested_to_limit_ratio
recent_claim_share_30d_365d
provider_recent_activity_ratio
customer_provider_intensity
same_service_intensity
```

Special care is given to **temporal leakage control**: historical behavioural features use information available before the claim being scored.

Synthetic generation variables such as fraud mechanism, fraud difficulty and latent fraud probability are explicitly excluded from predictive modelling.

---

## 9. Explainability with SHAP

The XGBoost model is explained using **TreeSHAP**.

Explainability is available at two levels.

### Global

- transformed feature importance;
- business-level feature aggregation;
- SHAP distribution analysis;
- model-driver analysis.

### Local

For an individual claim, the application provides:

- fraud-risk probability;
- baseline model output;
- feature-level SHAP contributions;
- strongest positive risk drivers;
- strongest negative risk drivers;
- consistency checks between model probability and SHAP reconstruction.

The deployed explanation contract covers all **107 transformed features**.

SHAP is used to explain **model behaviour**, not to establish causal relationships.

![Global SHAP Feature Importance](artifacts/explainability/figures/01_shap_global_bar.png)

![SHAP Distribution](artifacts/explainability/figures/02_shap_beeswarm.png)

---

## 10. REST API

Inference is exposed through **FastAPI**.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Service information |
| `GET` | `/health` | Runtime and model health |
| `GET` | `/model-info` | Deployed model contract |
| `POST` | `/score` | Score one claim |
| `POST` | `/score-batch` | Score multiple claims |
| `POST` | `/top-review` | Generate prioritized review queue |
| `POST` | `/explain` | Explain an individual prediction |
| `GET` | `/docs` | Interactive OpenAPI documentation |

Example health contract:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "XGBoost",
  "model_version": "1.0.0"
}
```

---

## 11. Streamlit Frontend

The investigation interface is implemented with **Streamlit**.

The frontend communicates with FastAPI through a dedicated API client rather than loading the model independently.

```text
Browser
   ↓
Streamlit
   ↓
FraudAPIClient
   ↓
FastAPI
   ↓
FraudScorer
   ↓
Preprocessor + XGBoost + TreeSHAP
```

This separation keeps model inference centralized and creates a clearer boundary between presentation and ML serving.

---

## 12. Repository Structure

```text
Fraud-risk-model/
│
├── api/                    # FastAPI inference service
├── artifacts/
│   ├── explainability/     # SHAP and error-analysis artifacts
│   ├── metadata/           # Model metadata and evaluation figures
│   ├── models/             # Frozen model
│   ├── predictions/        # Scoring outputs
│   └── preprocessors/      # Frozen preprocessing pipeline
│
├── configs/                # Project configuration
├── data/                   # Synthetic data pipeline outputs
├── docs/                   # Project documentation
├── frontend/               # Streamlit application
├── notebooks/              # Analysis and experimentation
├── scripts/                # Operational scripts
├── src/health_fraud/       # Core ML package
├── tests/
│   ├── unit/
│   └── integration/
│
├── .github/workflows/      # Continuous integration
├── Dockerfile              # API image
├── docker-compose.yml      # Local application stack
├── requirements.txt
└── README.md
```

---

## 13. Local Installation

### Clone

```bash
git clone https://github.com/Simo-Mesbahi/Fraud-risk-model.git
cd Fraud-risk-model
```

### Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the API

```bash
PYTHONPATH=.:src python -m uvicorn api.app.main:app \
  --host 0.0.0.0 \
  --port 8000
```

### Run the frontend

```bash
PYTHONPATH=.:src streamlit run frontend/app.py \
  --server.port 8501
```

API documentation:

```text
http://localhost:8000/docs
```

Frontend:

```text
http://localhost:8501
```

---

## 14. Docker Deployment

The complete application can be started with Docker Compose:

```bash
docker compose build
docker compose up -d
```

Check runtime status:

```bash
docker compose ps
```

Local services:

```text
API       http://localhost:8000
Frontend  http://localhost:8501
```

The API and frontend containers expose independent health checks.

For hosted deployment, the services can be deployed independently while the frontend receives the API base URL through:

```text
FRAUD_API_URL
```

---

## 15. Testing & Quality Assurance

The project contains dedicated **unit and integration tests** covering the critical ML and application contracts.

Current validated suite:

```text
228 tests passed
Critical-path coverage: 78.33%
Required coverage gate: 75%
```

Coverage includes:

- feature validation;
- inference preparation;
- deterministic scoring;
- ranking behaviour;
- top-fraction selection;
- model artifact contracts;
- API contracts;
- batch scoring;
- TreeSHAP additivity;
- probability / explanation consistency;
- frontend ↔ API integration;
- formatting and validation utilities.

GitHub Actions automatically validates the project on pushes and pull requests to `main`.

The CI pipeline includes:

```text
Dependency validation
        ↓
Artifact validation
        ↓
Python compilation
        ↓
Import contracts
        ↓
Full test suite
        ↓
Coverage gate
        ↓
Docker build validation
```

---

## 16. Model Artifacts & Metadata

The deployed inference stack is versioned through explicit artifacts:

```text
artifacts/models/health_fraud_xgboost.joblib
artifacts/preprocessors/health_fraud_preprocessor.joblib
artifacts/metadata/health_fraud_model_metadata.json
```

The metadata contract records information including:

- model identity and version;
- target;
- source feature contract;
- review policy;
- training period;
- test period;
- hyperparameters;
- final evaluation metrics.

Explainability and evaluation artifacts are stored separately from the runtime model.

---

## 17. Reproducibility

Reproducibility is supported through:

- deterministic random seeds;
- explicit temporal splits;
- frozen feature ordering;
- frozen preprocessing artifacts;
- frozen model artifacts;
- model metadata;
- deterministic inference tests;
- Dockerized runtime environments;
- automated CI validation.

The runtime API exposes the deployed model identity so the frontend can verify that it is communicating with the expected inference contract.

---

## 18. Limitations

This project deliberately documents its limitations.

### Synthetic data

All modelling results originate from a configurable synthetic insurance environment.

They demonstrate the modelling methodology and engineering architecture, but **cannot be interpreted as expected performance on real Foyer or other insurer data**.

### Simulation-specific patterns

Synthetic data may contain patterns that are easier or structurally different from fraud behaviour observed in production.

### Residual fraud risk

At the selected 3% operating point, a significant proportion of fraudulent claims remains outside the investigation queue.

### Human decision required

A high model score is not evidence that fraud occurred.

The platform is designed to support investigation prioritization, not autonomous claim rejection.

### Current model lifecycle

The deployed `XGBoost — v1.0.0` model and its analytical artifacts are currently frozen.

New data does not automatically trigger retraining or champion replacement.

---

## 19. Production Considerations

A real insurance deployment would require additional controls around:

- access control and authentication;
- personal-data protection;
- audit logging;
- model and feature monitoring;
- data-drift detection;
- concept-drift monitoring;
- model calibration monitoring;
- investigation feedback capture;
- model registry and version governance;
- automated retraining controls;
- champion / challenger evaluation;
- controlled model promotion;
- rollback capability;
- security and infrastructure hardening.

Model promotion should depend on multiple operational and statistical gates rather than a single performance metric.

---

## 20. Roadmap

Planned extensions include:

```text
New labelled data
       ↓
Data-quality gates
       ↓
Candidate retraining
       ↓
Champion / Challenger evaluation
       ↓
Performance + stability gates
       ↓
Controlled model promotion
       ↓
Artifact regeneration
       ↓
CI validation
       ↓
Deployment
```

Priority improvements:

- automated training and evaluation pipeline;
- model registry;
- champion / challenger lifecycle;
- automatic regeneration of evaluation artifacts;
- drift monitoring;
- production observability;
- investigator feedback loop;
- authentication and role-based access;
- persistent investigation state;
- automated deployment pipeline.

---

## 21. Author

**Mohammed El Mesbahi**

Data Science & Artificial Intelligence

Project focus:

** Data sciences · Machine Learning · Fraud Analytics · Explainable AI · Decision Support · ML Engineering**
