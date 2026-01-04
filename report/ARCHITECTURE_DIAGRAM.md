# Heart Disease Prediction MLOps - System Architecture

## Complete End-to-End Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  USER INTERACTION LAYER                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                     ┌────────────────────┼────────────────────┐
                     │                    │                    │
                ┌────▼────┐          ┌────▼────┐          ┌───▼────┐
                │  Web UI │          │ Mobile  │          │  cURL  │
                │ (Future)│          │ (Future)│          │  CLI   │
                └─────────┘          └─────────┘          └────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   API GATEWAY / INGRESS                                  │
│  - SSL/TLS Termination                                                                   │
│  - Rate Limiting (100 RPS)                                                               │
│  - CORS Headers                                                                          │
│  - Load Balancing                                                                        │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               KUBERNETES SERVICE LAYER                                   │
│                                                                                           │
│  ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐    │
│  │  LoadBalancer SVC   │      │   ClusterIP SVC     │      │   Headless SVC      │    │
│  │  (External Access)  │      │  (Internal Comms)   │      │  (StatefulSet)      │    │
│  │   Port 80 → 8000    │      │     Port 8000       │      │    Port 8000        │    │
│  └──────────┬──────────┘      └──────────┬──────────┘      └──────────┬──────────┘    │
│             └────────────────────┬────────┴──────────────────────────┬─────────────────┤
│                                  │                                    │                  │
│                                  ▼                                    ▼                  │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
│  │                            POD REPLICAS (3 instances)                              │ │
│  │                                                                                     │ │
│  │  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐               │ │
│  │  │   Pod 1     │          │   Pod 2     │          │   Pod 3     │               │ │
│  │  │ ┌─────────┐ │          │ ┌─────────┐ │          │ ┌─────────┐ │               │ │
│  │  │ │ FastAPI │ │          │ │ FastAPI │ │          │ │ FastAPI │ │               │ │
│  │  │ │Container│ │          │ │Container│ │          │ │Container│ │               │ │
│  │  │ └─────────┘ │          │ └─────────┘ │          │ └─────────┘ │               │ │
│  │  │ CPU: 250m   │          │ CPU: 250m   │          │ CPU: 250m   │               │ │
│  │  │ RAM: 512Mi  │          │ RAM: 512Mi  │          │ RAM: 512Mi  │               │ │
│  │  └─────────────┘          └─────────────┘          └─────────────┘               │ │
│  │                                                                                     │ │
│  │  Health: Liveness, Readiness, Startup Probes                                      │ │
│  │  Anti-affinity: Distributed across nodes                                          │ │
│  └───────────────────────────────────────────────────────────────────────────────────┘ │
│                                  ▲                                                       │
│                                  │                                                       │
│  ┌──────────────────────────────┼───────────────────────────────────────────────────┐ │
│  │                    HORIZONTAL POD AUTOSCALER (HPA)                                 │ │
│  │  - Min: 2 replicas | Max: 10 replicas                                             │ │
│  │  - CPU Target: 70% | Memory Target: 80%                                           │ │
│  │  - Scale Up: Immediate | Scale Down: 5min stabilization                           │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                    │                          │                          │
                    ▼                          ▼                          ▼
         ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
         │  ConfigMap       │      │  PVC (1Gi)       │      │  Secrets         │
         │  - MODEL_NAME    │      │  - Model Storage │      │  - API Keys      │
         │  - LOG_LEVEL     │      │  - ReadOnlyMany  │      │  (Future)        │
         │  - WORKERS       │      └──────────────────┘      └──────────────────┘
         └──────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  MONITORING & LOGGING LAYER                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
        ┌──────────────────────────────────┼──────────────────────────────────┐
        │                                  │                                   │
        ▼                                  ▼                                   ▼
┌───────────────┐                 ┌───────────────┐                 ┌────────────────┐
│  Prometheus   │◄────scrape─────│   API Pods    │────logs────────►│   File Logs    │
│  (port 9090)  │                 │  /metrics     │                 │  logs/api.log  │
└───────┬───────┘                 └───────────────┘                 └────────────────┘
        │                                 │
        │ query                           │ metrics
        ▼                                 ▼
┌───────────────┐                 ┌───────────────────────────────────────────────────┐
│   Grafana     │                 │  Prometheus Metrics (9 custom):                   │
│  (port 3000)  │                 │  - api_requests_total                             │
│               │                 │  - api_request_duration_seconds                   │
│  Dashboard:   │                 │  - predictions_total                              │
│  - 10 Panels  │                 │  - prediction_confidence                          │
│  - Real-time  │                 │  - batch_prediction_size                          │
│  - Alerts     │                 │  - model_load_time_seconds                        │
└───────┬───────┘                 │  - active_requests                                │
        │                         │  - api_errors_total                               │
        │ alerts                  │  - api_health_status                              │
        ▼                         └───────────────────────────────────────────────────┘
┌───────────────┐
│ AlertManager  │
│  (port 9093)  │
│               │
│  8 Alert Rules:
│  - APIDown
│  - ModelNotLoaded
│  - HighErrorRate
│  - HighLatency
│  - HighCPU/Memory
│  - etc.
└───────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                ML MODEL & DATA LAYER                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                   TRAINED MODELS                                      │
│                                                                                        │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐             │
│  │   XGBoost       │      │  Random Forest  │      │    Logistic     │             │
│  │  (Production)   │      │    (Backup)     │      │   Regression    │             │
│  │                 │      │                 │      │                 │             │
│  │  Accuracy: 88.5%│      │  Accuracy: 83.6%│      │  Accuracy: 85.2%│             │
│  │  ROC-AUC: 0.94  │      │  ROC-AUC: 0.91  │      │  ROC-AUC: 0.92  │             │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘             │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                    PREPROCESSING PIPELINE                                     │   │
│  │  1. FeatureEngineering (7 new features)                                      │   │
│  │  2. OutlierHandler (IQR-based)                                               │   │
│  │  3. StandardScaler (normalization)                                           │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                   DATA STORAGE                                        │
│                                                                                        │
│  ┌────────────────────┐                    ┌────────────────────┐                   │
│  │  Raw Dataset       │                    │  MLflow Artifacts  │                   │
│  │  heart_disease.csv │                    │  - Models          │                   │
│  │  303 samples       │                    │  - Metrics         │                   │
│  │  13 features       │                    │  - Visualizations  │                   │
│  └────────────────────┘                    └────────────────────┘                   │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CI/CD PIPELINE LAYER                                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              GITHUB ACTIONS WORKFLOW                                  │
│                                                                                        │
│  ┌──────┐    ┌──────┐    ┌───────┐    ┌────────┐    ┌──────┐    ┌─────┐    ┌────┐ │
│  │ Lint │───►│ Test │───►│ Train │───►│ MLflow │───►│ Intg │───►│ Sec │───►│Ntfy│ │
│  └──────┘    └──────┘    └───────┘    └────────┘    └──────┘    └─────┘    └────┘ │
│                                                                                        │
│  Stage 1: Code Quality (Black, isort, Flake8, Pylint)                                │
│  Stage 2: Unit Tests (Python 3.9, 3.10, 3.11 matrix) - 54 tests                      │
│  Stage 3: Model Training & Validation                                                │
│  Stage 4: MLflow Experiment Logging                                                  │
│  Stage 5: Integration Testing                                                        │
│  Stage 6: Security Scanning                                                          │
│  Stage 7: Notifications                                                              │
│                                                                                        │
│  Triggers: Push to main, Pull Requests                                               │
│  Artifacts: Test reports, coverage, trained models                                   │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                                  ┌─────────────────┐
                                  │ Docker Registry │
                                  │                 │
                                  │ Images:         │
                                  │ - latest        │
                                  │ - v1.0.0        │
                                  │ - dev           │
                                  └─────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SECURITY LAYER                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│  RBAC                     NetworkPolicy               Security Context                │
│  - ServiceAccount         - Ingress Rules             - Non-root User                 │
│  - Roles                  - Egress Rules              - Read-only FS                  │
│  - RoleBindings           - Pod Isolation             - No Privileges                 │
│                                                                                        │
│  Input Validation         Secrets Management          TLS/SSL                         │
│  - Pydantic Models        - K8s Secrets              - Cert-Manager                   │
│  - Type Checking          - Env Variables            - HTTPS Ingress                  │
│  - Range Validation       - Encrypted Storage        - mTLS (Future)                  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE DATA FLOW                             │
└─────────────────────────────────────────────────────────────────────────────────┘

[1] Data Acquisition
    │
    ▼
┌───────────────────┐
│ UCI Repository    │
│ Heart Disease     │
│ Dataset (303 rows)│
└─────────┬─────────┘
          │ download
          ▼
[2] Data Cleaning & EDA
    │
    ▼
┌───────────────────┐         ┌──────────────────┐
│ data/             │────────►│ notebooks/       │
│ - raw.csv         │         │ 01_eda.ipynb     │
│ - clean.csv       │         │                  │
└─────────┬─────────┘         │ Visualizations:  │
          │                   │ - 12 plots       │
          │                   └──────────────────┘
          ▼
[3] Feature Engineering
    │
    ▼
┌─────────────────────────────────────┐
│ FeatureEngineering Transformer      │
│ - age_group                          │
│ - chol_category                      │
│ - bp_category                        │
│ - heart_rate_reserve                 │
│ - cp_severity                        │
│ - risk_score                         │
│ - metabolic_health                   │
└──────────────┬──────────────────────┘
               │
               ▼
[4] Preprocessing Pipeline
    │
    ▼
┌────────────────────────────────────┐
│ Pipeline Steps:                    │
│ 1. OutlierHandler (IQR)           │
│ 2. StandardScaler                 │
│ 3. Feature Engineering            │
└────────────┬───────────────────────┘
             │
             │ 80/20 split
             ▼
        ┌─────────┴─────────┐
        │                   │
    ┌───▼────┐          ┌───▼────┐
    │ Train  │          │  Test  │
    │ 242    │          │   61   │
    └───┬────┘          └───┬────┘
        │                   │
        ▼                   │
[5] Model Training          │
    │                       │
    ├─────────────┬─────────┼─────────┐
    │             │         │         │
┌───▼───┐    ┌───▼───┐ ┌──▼───┐ ┌───▼────┐
│LogReg │    │RandFor│ │XGBoost│ │5-Fold  │
│       │    │       │ │       │ │Cross   │
│85.2%  │    │83.6%  │ │88.5% ││Valid   │
└───┬───┘    └───┬───┘ └──┬───┘ └───┬────┘
    │            │        │         │
    └────────────┴────────┴─────────┘
                 │
                 ▼
[6] Model Evaluation
    │
    ▼
┌────────────────────────────────────┐
│ Metrics:                           │
│ - Accuracy, Precision, Recall      │
│ - F1-Score, ROC-AUC               │
│ - Confusion Matrix                 │
│ - Feature Importance               │
└────────────┬───────────────────────┘
             │
             ▼
[7] MLflow Logging
    │
    ▼
┌────────────────────────────────────┐
│ mlruns/                            │
│ - Parameters                       │
│ - Metrics                          │
│ - Artifacts (models, plots)        │
│ - Tags & Metadata                  │
└────────────┬───────────────────────┘
             │
             ▼
[8] Model Selection & Registry
    │
    ▼
┌────────────────────────────────────┐
│ Best Model: XGBoost                │
│ - Staged to Production             │
│ - Version: 1.0.0                   │
└────────────┬───────────────────────┘
             │
             ▼
[9] Model Packaging
    │
    ▼
┌─────────────────────────────────────┐
│ models/                             │
│ - xgboost_model.pkl                 │
│ - preprocessing_pipeline.pkl        │
│ - xgboost_metadata.json             │
│ - xgboost_model_card.md             │
└─────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE DATA FLOW                            │
└─────────────────────────────────────────────────────────────────────────────────┘

[1] Client Request
    │
    ▼
┌────────────────────┐
│ POST /predict      │
│ {                  │
│   "age": 63,       │
│   "sex": 1,        │
│   "cp": 1,         │
│   ...              │
│ }                  │
└─────────┬──────────┘
          │
          ▼
[2] Input Validation
    │
    ▼
┌────────────────────────────────┐
│ Pydantic Model                 │
│ - Type checking                │
│ - Range validation             │
│ - Required fields              │
└────────────┬───────────────────┘
             │ VALID
             ▼
[3] Load Models
    │
    ▼
┌─────────────────────────────────┐
│ HeartDiseasePredictor          │
│ - preprocessing_pipeline.pkl    │
│ - xgboost_model.pkl            │
└────────────┬────────────────────┘
             │
             ▼
[4] Preprocessing
    │
    ▼
┌─────────────────────────────────┐
│ Transform Input:                │
│ 1. Feature engineering          │
│ 2. Outlier handling             │
│ 3. Scaling                      │
└────────────┬────────────────────┘
             │
             ▼
[5] Prediction
    │
    ▼
┌─────────────────────────────────┐
│ XGBoost.predict()               │
│ XGBoost.predict_proba()         │
└────────────┬────────────────────┘
             │
             ▼
[6] Post-processing
    │
    ▼
┌──────────────────────────────────┐
│ Result Dictionary:               │
│ {                                │
│   "prediction": 1,               │
│   "diagnosis": "Heart Disease",  │
│   "confidence": 0.87,            │
│   "probabilities": {...},        │
│   "model_used": "xgboost",       │
│   "timestamp": "..."             │
│ }                                │
└────────────┬─────────────────────┘
             │
             ▼
[7] Logging & Metrics
    │
    ├─────────────┬──────────────┐
    │             │              │
┌───▼────┐   ┌───▼────┐    ┌───▼─────────┐
│ File   │   │Prometheus│    │  MLflow     │
│ Logs   │   │ Metrics  │    │ (Optional)  │
└────────┘   └──────────┘    └─────────────┘
             │
             ▼
[8] Response to Client
    │
    ▼
┌────────────────────────────┐
│ HTTP 200 OK                │
│ Content-Type: application  │
│             /json          │
│ Body: {...}                │
└────────────────────────────┘
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DEPLOYMENT ENVIRONMENTS                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

DEVELOPMENT                  STAGING                     PRODUCTION
    │                           │                             │
    ▼                           ▼                             ▼
┌──────────┐              ┌──────────┐                 ┌──────────┐
│ Docker   │              │ Minikube │                 │   GKE    │
│ Compose  │              │          │                 │   EKS    │
│          │              │ 1 Node   │                 │   AKS    │
│ 1 Pod    │              │ 2 Pods   │                 │          │
└──────────┘              └──────────┘                 │ 3+ Nodes │
                                                        │ 3+ Pods  │
                                                        │ HA Setup │
                                                        └──────────┘

Local Testing ────► Integration Testing ────► Production Deployment
     │                       │                          │
     │                       │                          │
     ▼                       ▼                          ▼
Unit Tests              API Tests                 Monitoring
Linting                 E2E Tests                 Auto-scaling
Coverage                Load Tests                Alerting
```

---

**Architecture Version**: 1.0
**Last Updated**: December 22, 2025
**Created By**: Saif Afzal
