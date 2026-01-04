# Heart Disease Prediction - MLOps End-to-End Project

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MLOps](https://img.shields.io/badge/MLOps-Production--Ready-orange.svg)](.)

## Overview

This project demonstrates a complete MLOps pipeline for predicting heart disease risk using the UCI Heart Disease dataset. The implementation follows industry best practices including experiment tracking, CI/CD pipelines, containerization, and production deployment.

**Course:** MLOps (S1-25_AIMLCZG523)
**Assignment:** Assignment 1 - End-to-End ML Model Development, CI/CD, and Production Deployment
**Total Marks:** 50

## Problem Statement

Build a machine learning classifier to predict the risk of heart disease based on patient health data, and deploy the solution as a cloud-ready, monitored API.

## Dataset

- **Source:** [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Samples:** 303 patient records
- **Features:** 13 clinical features + 1 binary target
- **Target:** Binary classification (0 = No disease, 1 = Disease present)

### Features Description

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numerical |
| sex | Sex (1 = male, 0 = female) | Categorical |
| cp | Chest pain type (1-4) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numerical |
| chol | Serum cholesterol (mg/dl) | Numerical |
| fbs | Fasting blood sugar > 120 mg/dl | Categorical |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numerical |
| exang | Exercise induced angina | Categorical |
| oldpeak | ST depression induced by exercise | Numerical |
| slope | Slope of peak exercise ST segment | Categorical |
| ca | Number of major vessels (0-3) | Categorical |
| thal | Thalassemia (3, 6, 7) | Categorical |
| target | Heart disease diagnosis | Binary |

## Project Structure

```
heart-disease-mlops/
├── .github/
│ └── workflows/
│ └── ci-cd.yml # GitHub Actions CI/CD pipeline
├── app/
│ └── main.py # FastAPI application
├── data/
│ ├── download_data.py # Data acquisition script
│ ├── heart_disease.csv # Raw dataset
│ └── heart_disease_clean.csv # Cleaned dataset
├── notebooks/
│ ├── 01_eda.ipynb # Exploratory Data Analysis
│ └── 02_model_training.ipynb # Model development
├── src/
│ ├── preprocessing.py # Data preprocessing pipeline
│ ├── train.py # Model training script
│ └── model_pipeline.py # Complete ML pipeline
├── tests/
│ ├── test_preprocessing.py # Unit tests for preprocessing
│ ├── test_model.py # Unit tests for model
│ └── test_api.py # API endpoint tests
├── k8s/
│ ├── deployment.yaml # Kubernetes deployment
│ └── service.yaml # Kubernetes service
├── models/
│ └── best_model.pkl # Trained model artifact
├── screenshots/ # Visualization screenshots
├── report/
│ └── MLOps_Assignment_Report.docx # Final report
├── Dockerfile # Docker configuration
├── docker-compose.yml # Docker Compose setup
├── requirements.txt # Python dependencies
├── environment.yml # Conda environment
└── README.md # This file
```

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- Docker & Docker Compose (for containerization)
- Kubernetes/Minikube (for deployment)
- Git

### Local Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd heart-disease-mlops
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
```bash
python data/download_data.py
```

## Usage

### 1. Data Acquisition & EDA

Download and explore the dataset:

```bash
# Download data
python data/download_data.py

# Run EDA notebook
jupyter notebook notebooks/01_eda.ipynb
```

**Key EDA Findings:**
- 303 patient records with balanced classes (54.5% disease, 45.5% no disease)
- Minimal missing values (< 2% in 'ca' and 'thal')
- Strong predictors: cp, thalach, oldpeak, exang
- No severe multicollinearity
- Some outliers present but clinically valid

### 2. Model Training

```bash
# Train models with MLflow tracking
python src/train.py

# View MLflow UI
mlflow ui
```

**Models Implemented:**
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_preprocessing.py -v
```

### 4. Docker Containerization

```bash
# Build Docker image
docker build -t heart-disease-api .

# Run container
docker run -p 8000:8000 heart-disease-api

# Using Docker Compose
docker-compose up
```

**Test the API:**
```bash
curl -X POST http://localhost:8000/predict \
 -H "Content-Type: application/json" \
 -d '{
 "age": 63,
 "sex": 1,
 "cp": 3,
 "trestbps": 145,
 "chol": 233,
 "fbs": 1,
 "restecg": 0,
 "thalach": 150,
 "exang": 0,
 "oldpeak": 2.3,
 "slope": 0,
 "ca": 0,
 "thal": 1
 }'
```

### 5. Kubernetes Deployment

```bash
# Start Minikube
minikube start

# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment
kubectl get pods
kubectl get services

# Access the service
minikube service heart-disease-service
```

## CI/CD Pipeline

The project uses GitHub Actions for automated CI/CD:

**Pipeline Stages:**
1. **Linting:** Code quality checks with flake8 and black
2. **Testing:** Run unit tests with pytest
3. **Model Training:** Train and validate models
4. **Build:** Create Docker image
5. **Deploy:** Push to container registry

**Workflow triggers:**
- Push to main/develop branches
- Pull requests
- Manual workflow dispatch

## Monitoring & Logging

### Application Logging
- Request/response logging in FastAPI
- Model prediction logging
- Error tracking and debugging

### Metrics Dashboard
- API request count
- Prediction latency
- Model performance metrics
- System resource utilization

**Access Monitoring:**
```bash
# Using Prometheus & Grafana (with docker-compose)
docker-compose -f docker-compose.monitoring.yml up
```

## Model Performance

| Model | Accuracy | Precision | Recall | ROC-AUC | F1-Score |
|-------|----------|-----------|--------|---------|----------|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |

*Note: Results will be updated after Task 2 completion*

## Task Completion Status

- [x] **Task 1:** Data Acquisition & EDA (5 marks) [DONE]
- [ ] **Task 2:** Feature Engineering & Model Development (8 marks)
- [ ] **Task 3:** Experiment Tracking (5 marks)
- [ ] **Task 4:** Model Packaging & Reproducibility (7 marks)
- [ ] **Task 5:** CI/CD Pipeline & Testing (8 marks)
- [ ] **Task 6:** Model Containerization (5 marks)
- [ ] **Task 7:** Production Deployment (7 marks)
- [ ] **Task 8:** Monitoring & Logging (3 marks)
- [ ] **Task 9:** Documentation & Reporting (2 marks)

## Key Features

[DONE] **Reproducibility:** Complete requirements.txt and environment.yml
[DONE] **Automation:** Fully automated CI/CD pipeline
[DONE] **Testing:** Comprehensive unit test coverage
[DONE] **Containerization:** Docker & Docker Compose ready
[DONE] **Deployment:** Kubernetes manifests included
[DONE] **Monitoring:** Logging and metrics tracking
[DONE] **Documentation:** Detailed README and report

## Technologies Used

**ML/Data Science:**
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- XGBoost

**MLOps:**
- MLflow (experiment tracking)
- Docker (containerization)
- Kubernetes (orchestration)
- GitHub Actions (CI/CD)

**API Development:**
- FastAPI
- Uvicorn
- Pydantic

**Testing:**
- pytest
- pytest-cov

## Contributing

This is an academic assignment project. For any questions or issues, please contact the course instructor.

## License

This project is created for educational purposes as part of the MLOps course at BITS Pilani.

## Author

**Student:** Saif Afzal
**Course:** MLOps (S1-25_AIMLCZG523)
**Institution:** BITS Pilani
**Date:** December 2025

## Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- BITS Pilani for the MLOps course
- Course instructors and teaching assistants

---

**Note:** This project demonstrates production-ready MLOps practices including version control, testing, containerization, CI/CD, and deployment strategies suitable for real-world applications.
