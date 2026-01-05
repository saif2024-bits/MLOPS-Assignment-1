# MLOPS Assignment 1 - Deliverables Checklist

## Repository Information
- **GitHub Repository**: https://github.com/saif2024-bits/MLOPS-Assignment-1
- **Main Branch**: https://github.com/saif2024-bits/MLOPS-Assignment-1/tree/main
- **CI/CD Status**: ✅ All Pipelines Passing

---

## A) GitHub Repository Deliverables

### ✅ 1. Code
**Location**: `src/`, `app/`

**Files**:
- `src/train.py` - Model training script with dynamic paths
- `src/train_mlflow.py` - MLflow experiment tracking
- `src/preprocessing.py` - Data preprocessing and feature engineering
- `src/model_pipeline.py` - Prediction pipeline and model management
- `app/main.py` - FastAPI application
- `app/monitoring.py` - Prometheus metrics
- `app/test_api.py` - API tests

**Status**: ✅ Complete

---

### ✅ 2. Dockerfile(s)
**Location**: Root directory

**Files**:
- `Dockerfile` - Multi-stage production Docker image
- `docker-compose.yml` - Basic API deployment
- `docker-compose.monitoring.yml` - Full monitoring stack (Prometheus + Grafana)
- `.dockerignore` - Docker build optimization

**Key Features**:
- Multi-stage build (builder + runtime)
- Non-root user (security)
- Health checks
- Optimized image size (~400MB)

**Status**: ✅ Complete

---

### ✅ 3. Requirements Files
**Location**: Root directory

**Files**:
- `requirements.txt` - Main dependencies (39 lines)
- `requirements_production.txt` - Production-specific dependencies
- `environment.yml` - Conda environment specification

**Status**: ✅ Complete

---

### ✅ 4. Cleaned Dataset and Download Script
**Location**: `data/`

**Files**:
- `data/heart_disease_clean.csv` - Cleaned dataset (303 records, no missing values)
- `data/heart_disease.csv` - Raw dataset (with missing values)
- `data/download_data.py` - Automated download script from UCI repository

**Script Features**:
- Automatic download from UCI ML Repository
- Missing value handling (median imputation)
- Creates both raw and clean versions
- Validation and info display

**Command**:
```bash
python data/download_data.py
```

**Status**: ✅ Complete

---

### ✅ 5. Jupyter Notebooks
**Location**: `notebooks/`

**Files**:
- `01_eda.ipynb` - Exploratory Data Analysis
- `01_eda_executed.ipynb` - EDA with outputs
- `02_model_training.ipynb` - Model training experiments
- `03_mlflow_experiments.ipynb` - MLflow tracking experiments

**Covers**:
- **EDA**: Missing values, distributions, correlations, outliers, visualizations
- **Training**: Multiple models (Logistic Regression, Random Forest, XGBoost)
- **Inference**: Predictions with confidence scores

**Status**: ✅ Complete

---

### ✅ 6. Tests Folder
**Location**: `tests/`

**Files**:
- `tests/test_preprocessing.py` - Data loading and preprocessing tests (17 tests)
- `tests/test_model.py` - Model training and evaluation tests (18 tests)
- `tests/test_model_pipeline.py` - Prediction pipeline tests (20 tests)
- `tests/conftest.py` - Pytest configuration
- `tests/__init__.py` - Package marker

**Coverage**: 93% (55 unit tests total)

**Command**:
```bash
pytest tests/ -v --cov=src --cov=app
```

**Status**: ✅ Complete

---

### ✅ 7. GitHub Actions Workflow
**Location**: `.github/workflows/`

**File**: `ci-cd.yml` (360 lines)

**Pipeline Stages**:
1. **Lint** - Code quality (Flake8, Black, isort, Pylint)
2. **Test** - Unit tests with coverage (93%)
3. **Train Model** - Model training and validation
4. **MLflow Tracking** - Experiment tracking
5. **Integration Test** - Model loading and prediction
6. **Security Scan** - Dependency vulnerability check

**Triggers**:
- Push to main/develop
- Pull requests
- Manual workflow dispatch

**Status**: ✅ Complete - All stages passing

---

### ✅ 8. Deployment Manifests
**Location**: `k8s/`

**Kubernetes Manifests**:
- `deployment.yaml` - Application deployment (3 replicas)
- `service.yaml` - LoadBalancer service
- `ingress.yaml` - Ingress configuration
- `hpa.yaml` - Horizontal Pod Autoscaler
- `rbac.yaml` - Role-based access control
- `deploy.sh` - Deployment script
- `cleanup.sh` - Cleanup script
- `README.md` - Deployment documentation

**Features**:
- Auto-scaling (2-10 pods based on CPU)
- Health checks (liveness + readiness)
- Resource limits
- Rolling updates
- RBAC enabled

**Helm Charts**: Not included (Kubernetes manifests provided instead)

**Status**: ✅ Complete

---

### ✅ 9. Screenshot Folder
**Location**: `screenshots/`

**Files** (14 screenshots):
- `01_missing_values.png` - Missing data visualization
- `02_class_balance.png` - Target distribution
- `03_numerical_distributions.png` - Feature distributions
- `04_categorical_distributions.png` - Categorical features
- `05_correlation_heatmap.png` - Correlation matrix
- `06_target_correlations.png` - Feature-target correlations
- `07_numerical_vs_target.png` - Numerical features vs target
- `08_categorical_vs_target.png` - Categorical features vs target
- `09_outliers.png` - Outlier detection
- `10_pairplot.png` - Feature relationships
- `model_comparison.png` - Model performance comparison
- `roc_curves.png` - ROC curves for all models
- Plus 2 more MLflow visualizations

**Status**: ✅ Complete

---

### ⚠️ 10. Final Written Report (10 pages)
**Location**: `report/`

**Current File**: `MLOps_Assignment_Report.md` (1,125 lines, ~30KB)

**Status**: ⚠️ **PARTIAL** - Report exists in Markdown format (.md)

**Action Required**: Convert to .doc/.docx format

**Content Sections** (Complete):
1. Executive Summary
2. Introduction
3. Dataset Description
4. Exploratory Data Analysis
5. Data Preprocessing Pipeline
6. Model Development & Training
7. MLflow Experiment Tracking
8. Model Deployment & Serving
9. Containerization & Orchestration
10. Monitoring & Observability
11. CI/CD Pipeline
12. Conclusion & Future Work

**Conversion Options**:
```bash
# Option 1: Using Pandoc
pandoc report/MLOps_Assignment_Report.md -o report/MLOps_Assignment_Report.docx

# Option 2: Manual conversion
# Open .md file → Copy content → Paste in Word → Format → Save as .docx
```

---

## B) Short Video (End-to-End Pipeline)

### 📹 Video Requirements
**Status**: ⚠️ **TODO** - Not yet created

**Required Content**:
1. **Introduction** (30s)
   - Project overview
   - Show GitHub repository

2. **Data Pipeline** (1 min)
   - Download data script
   - Show EDA notebook

3. **Model Training** (1.5 min)
   - Run training script
   - Show MLflow UI with experiments

4. **Testing** (1 min)
   - Run unit tests with coverage
   - Show CI/CD pipeline

5. **Docker Deployment** (1.5 min)
   - Build Docker image
   - Run container
   - Test API endpoints

6. **Kubernetes** (1 min)
   - Deploy to K8s
   - Show scaling
   - Show monitoring

7. **Conclusion** (30s)
   - Summary
   - GitHub link

**Total Duration**: 7-8 minutes

**Guide Available**: `VIDEO_DEMO_GUIDE.md` (if created earlier)

---

## C) Deployed API URL / Access Instructions

### 🚀 Option 1: Local Testing (Recommended)

#### Docker Deployment:
```bash
# Clone repository
git clone https://github.com/saif2024-bits/MLOPS-Assignment-1.git
cd MLOPS-Assignment-1

# Build Docker image
docker build -t heart-disease-api:latest .

# Run container
docker run -d -p 8000:8000 --name heart-api heart-disease-api:latest

# Access API
# - Health: http://localhost:8000/health
# - Docs: http://localhost:8000/docs
# - Metrics: http://localhost:8000/metrics

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'

# Cleanup
docker stop heart-api
docker rm heart-api
```

#### Local Python Deployment:
```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download data
python data/download_data.py

# Train models
python src/train.py

# Run API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Access at http://localhost:8000/docs
```

#### Kubernetes Deployment:
```bash
# Deploy to Kubernetes
./k8s/deploy.sh

# Check deployment
kubectl get pods
kubectl get services

# Access API (once exposed)
# kubectl port-forward service/heart-disease-api 8000:8000

# Cleanup
./k8s/cleanup.sh
```

### 🌐 Option 2: Public Deployment
**Status**: ⚠️ **Not Deployed** - Local testing only

**To Deploy Publicly**:
- AWS ECS/EKS
- Google Cloud Run
- Azure Container Instances
- Heroku
- Railway.app

---

## Production-Readiness Requirements

### ✅ 1. Scripts Execute from Clean Setup

**Verification**:
```bash
# Create fresh environment
python -m venv clean_env
source clean_env/bin/activate
pip install -r requirements.txt

# All scripts work
python data/download_data.py  # ✅ Works
python src/train.py            # ✅ Works (dynamic paths)
python src/train_mlflow.py     # ✅ Works (dynamic paths)
pytest tests/                  # ✅ Works (93% coverage)
```

**Evidence**:
- GitHub Actions runs from clean Ubuntu environment
- All dynamic paths use `PROJECT_ROOT`
- No hardcoded absolute paths

**Status**: ✅ **VERIFIED**

---

### ✅ 2. Model Serves in Isolated Docker Environment

**Verification**:
```bash
# Build from scratch
docker build -t heart-disease-api:test .

# Run isolated
docker run -d -p 8000:8000 heart-disease-api:test

# Test health
curl http://localhost:8000/health
# Response: {"status": "healthy", ...}

# Test prediction
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
# Response: {"prediction": 0, "probabilities": {...}, ...}
```

**Evidence**:
- Multi-stage Dockerfile
- Health checks passing
- API endpoints working
- Models loaded successfully

**Status**: ✅ **VERIFIED**

---

### ✅ 3. Pipeline Fails on Errors with Clear Logs

**Verification**:
GitHub Actions pipeline has 6 stages, each fails independently:

**Example Error Messages**:
```
# Syntax Error
src/train.py:45:1: E999 SyntaxError: invalid syntax
Error: Process completed with exit code 1

# Test Failure
FAILED tests/test_model.py::test_accuracy
AssertionError: Model accuracy 0.55 below threshold 0.60
Error: Process completed with exit code 1

# Import Error
ModuleNotFoundError: No module named 'xgboost'
Error: Process completed with exit code 1
```

**Pipeline Stages**:
1. Lint - ✅ Fails on syntax/style errors
2. Test - ✅ Fails on test failures
3. Train - ✅ Fails on training errors
4. MLflow - ✅ Fails on tracking errors
5. Integration - ✅ Fails on prediction errors
6. Security - ⚠️ Warnings only (continue-on-error)

**Status**: ✅ **VERIFIED**

---

## Summary Status

| Deliverable | Status | Notes |
|-------------|--------|-------|
| **A1. Code** | ✅ Complete | All source files present |
| **A2. Dockerfiles** | ✅ Complete | Multi-stage + docker-compose |
| **A3. Requirements** | ✅ Complete | 3 files (txt + yml) |
| **A4. Dataset + Script** | ✅ Complete | Auto-download from UCI |
| **A5. Notebooks** | ✅ Complete | EDA + Training + Inference |
| **A6. Tests** | ✅ Complete | 55 tests, 93% coverage |
| **A7. CI/CD Workflow** | ✅ Complete | All stages passing |
| **A8. K8s Manifests** | ✅ Complete | Full deployment setup |
| **A9. Screenshots** | ✅ Complete | 14 visualizations |
| **A10. Report (doc/docx)** | ⚠️ **Partial** | **Need .docx conversion** |
| **B. Video** | ⚠️ **TODO** | **Need to record** |
| **C. Deployment** | ✅ Complete | Local Docker instructions |
| **Prod Req 1** | ✅ Verified | Clean setup works |
| **Prod Req 2** | ✅ Verified | Docker isolation works |
| **Prod Req 3** | ✅ Verified | Pipeline fails properly |

---

## Action Items

### Must Complete:
1. ⚠️ **Convert report to .docx format**
   ```bash
   pandoc report/MLOps_Assignment_Report.md -o report/MLOps_Assignment_Report.docx
   ```

2. ⚠️ **Record demonstration video** (7-8 minutes)
   - Follow VIDEO_DEMO_GUIDE.md structure
   - Cover end-to-end pipeline
   - Upload to YouTube/Drive

### Optional:
3. 📦 Deploy to public cloud (AWS/GCP/Azure)
4. 🎨 Add Helm charts (currently using raw K8s manifests)

---

## Repository Structure

```
MLOPS-Assignment-1/
├── .github/workflows/
│   └── ci-cd.yml              ✅ CI/CD pipeline
├── app/
│   ├── main.py                ✅ FastAPI application
│   ├── monitoring.py          ✅ Prometheus metrics
│   └── test_api.py            ✅ API tests
├── data/
│   ├── download_data.py       ✅ Auto-download script
│   ├── heart_disease.csv      ✅ Raw data
│   └── heart_disease_clean.csv ✅ Cleaned data
├── k8s/
│   ├── deployment.yaml        ✅ K8s deployment
│   ├── service.yaml           ✅ K8s service
│   ├── ingress.yaml           ✅ Ingress
│   ├── hpa.yaml               ✅ Auto-scaling
│   └── ...                    ✅ More manifests
├── models/
│   ├── *.pkl                  ✅ Trained models
│   └── *.json                 ✅ Metadata
├── monitoring/
│   ├── prometheus/            ✅ Prometheus config
│   └── grafana/               ✅ Grafana dashboards
├── notebooks/
│   ├── 01_eda.ipynb           ✅ EDA
│   ├── 02_model_training.ipynb ✅ Training
│   └── 03_mlflow_experiments.ipynb ✅ MLflow
├── report/
│   ├── MLOps_Assignment_Report.md ✅ Report (Markdown)
│   └── MLOps_Assignment_Report.docx ⚠️ TODO
├── screenshots/               ✅ 14 visualizations
├── src/
│   ├── train.py               ✅ Training script
│   ├── train_mlflow.py        ✅ MLflow tracking
│   ├── preprocessing.py       ✅ Data preprocessing
│   └── model_pipeline.py      ✅ Prediction pipeline
├── tests/
│   ├── test_preprocessing.py  ✅ 17 tests
│   ├── test_model.py          ✅ 18 tests
│   └── test_model_pipeline.py ✅ 20 tests
├── Dockerfile                 ✅ Multi-stage Docker
├── docker-compose.yml         ✅ Docker compose
├── requirements.txt           ✅ Dependencies
├── environment.yml            ✅ Conda env
├── README.md                  ✅ Main documentation
├── QUICK_START.md             ✅ Quick start guide
└── PRODUCTION_READINESS_VERIFICATION.md ✅ Verification doc
```

---

## Submission Checklist

- [x] Code committed and pushed
- [x] CI/CD pipeline passing
- [x] All tests passing (93% coverage)
- [x] Docker builds successfully
- [x] Documentation complete
- [ ] **Report converted to .docx**
- [ ] **Video recorded and uploaded**
- [x] GitHub repository public/accessible

---

**Last Updated**: 2026-01-05
**Repository**: https://github.com/saif2024-bits/MLOPS-Assignment-1
**Status**: 95% Complete (Report + Video pending)
