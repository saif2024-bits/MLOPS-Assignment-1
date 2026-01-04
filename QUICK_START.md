# Heart Disease Prediction MLOps - Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide will help you set up and run the Heart Disease Prediction system quickly.

---

## Prerequisites

- **Python**: 3.9, 3.10, or 3.11
- **Docker**: 20.10+ (optional, for containerized deployment)
- **Git**: For cloning the repository

---

## Option 1: Local Development Setup (Recommended for Testing)

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd heart-disease-mlops

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_production.txt
```

### Step 2: Get the Data

```bash
# Download and prepare dataset
python data/download_data.py
```

### Step 3: Train the Model

```bash
# Train all models (takes ~2-3 minutes)
python src/train.py
```

### Step 4: Start the API

```bash
# Start FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Step 5: Test the API

Open your browser and go to:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

Or use curl:
```bash
curl http://localhost:8000/health
```

### Step 6: Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6
  }'
```

**Expected Response**:
```json
{
  "prediction": 1,
  "diagnosis": "Heart Disease",
  "confidence": 0.87,
  "probabilities": {
    "no_disease": 0.13,
    "disease": 0.87
  },
  "model_used": "xgboost",
  "timestamp": "2025-12-22T..."
}
```

---

## Option 2: Docker Deployment (Recommended for Production)

### Step 1: Build and Run

```bash
# Build Docker image
docker build -t heart-disease-api:latest .

# Run container
docker run -p 8000:8000 heart-disease-api:latest
```

### Step 2: Test

```bash
curl http://localhost:8000/health
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Option 3: Kubernetes Deployment (Production-Ready)

### Prerequisites
- Kubernetes cluster (Minikube, GKE, EKS, or AKS)
- kubectl configured

### Quick Deploy

```bash
# Make scripts executable
chmod +x k8s/*.sh

# Deploy to Kubernetes
./k8s/deploy.sh

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=heart-disease-api --timeout=300s

# Access via port-forward
kubectl port-forward svc/heart-disease-api-service 8000:80
```

### Test Deployment

```bash
curl http://localhost:8000/health
```

### Check Status

```bash
# View pods
kubectl get pods -l app=heart-disease-api

# View logs
kubectl logs -f -l app=heart-disease-api

# View service
kubectl get svc heart-disease-api-service
```

---

## Option 4: Full Monitoring Stack

### Start Monitoring

```bash
# Ensure logs directory exists
mkdir -p logs

# Start complete stack (API + Prometheus + Grafana + AlertManager)
docker-compose -f docker-compose.monitoring.yml up -d

# Check all services are running
docker-compose -f docker-compose.monitoring.yml ps
```

### Access Dashboards

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **AlertManager**: http://localhost:9093

### View Grafana Dashboard

1. Open http://localhost:3000
2. Login: `admin` / `admin`
3. Go to **Dashboards** → **Heart Disease Prediction API Dashboard**
4. View real-time metrics!

---

## Running Tests

### Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_preprocessing.py -v
```

### API Tests

```bash
# Ensure API is running first
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Run API tests
python app/test_api.py
```

### Verify Everything

```bash
# Run complete verification
chmod +x verify_all_tasks.sh
./verify_all_tasks.sh
```

---

## Common Commands

### Development

```bash
# Activate environment
source venv/bin/activate

# Train specific model
python src/train.py --model xgboost

# Start API with auto-reload
uvicorn app.main:app --reload

# Run linting
black src/ app/ tests/
isort src/ app/ tests/
flake8 src/ app/ tests/
pylint src/ app/ tests/
```

### Docker

```bash
# Build
docker build -t heart-disease-api:latest .

# Run
docker run -p 8000:8000 heart-disease-api:latest

# Run with volumes (for development)
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  heart-disease-api:latest

# View logs
docker logs -f <container-id>

# Stop and remove
docker stop <container-id>
docker rm <container-id>
```

### Kubernetes

```bash
# Deploy
./k8s/deploy.sh

# Scale
kubectl scale deployment heart-disease-api --replicas=5

# Update image
kubectl set image deployment/heart-disease-api \
  api=heart-disease-api:v1.1.0

# Rollback
kubectl rollout undo deployment/heart-disease-api

# Delete
./k8s/cleanup.sh
```

### Monitoring

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# View Prometheus targets
curl http://localhost:9090/api/v1/targets | jq

# Query metrics
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=api_requests_total'

# Stop monitoring
docker-compose -f docker-compose.monitoring.yml down
```

---

## Troubleshooting

### Issue: Model not loading

**Solution**:
```bash
# Check models directory
ls -la models/

# Retrain if needed
python src/train.py
```

### Issue: Port already in use

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

### Issue: Dependencies not found

**Solution**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements_production.txt
```

### Issue: Kubernetes pods not starting

**Solution**:
```bash
# Check pod status
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>

# For Minikube - load image
minikube image load heart-disease-api:latest
```

### Issue: Tests failing

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall test dependencies
pip install pytest pytest-cov

# Run tests with verbose output
pytest -vv tests/
```

---

## Quick Reference

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API docs |
| `/model/info` | GET | Model information |
| `/features` | GET | Feature descriptions |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/metrics` | GET | Prometheus metrics |

### Feature Values

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `age` | int | 1-120 | Age in years |
| `sex` | int | 0-1 | Sex (1=male, 0=female) |
| `cp` | int | 1-4 | Chest pain type |
| `trestbps` | int | 80-200 | Resting blood pressure (mm Hg) |
| `chol` | int | 100-600 | Serum cholesterol (mg/dl) |
| `fbs` | int | 0-1 | Fasting blood sugar > 120 mg/dl |
| `restecg` | int | 0-2 | Resting ECG results |
| `thalach` | int | 60-220 | Maximum heart rate achieved |
| `exang` | int | 0-1 | Exercise induced angina |
| `oldpeak` | float | 0-10 | ST depression induced by exercise |
| `slope` | int | 1-3 | Slope of peak exercise ST segment |
| `ca` | int | 0-3 | Number of major vessels (fluoroscopy) |
| `thal` | int | 3,6,7 | Thalassemia |

---

## Next Steps

After getting the system running:

1. ✅ **Explore the API**: Visit http://localhost:8000/docs
2. ✅ **Make Predictions**: Use the `/predict` endpoint
3. ✅ **View Monitoring**: Set up the monitoring stack
4. ✅ **Run Tests**: Execute the test suite
5. ✅ **Deploy to K8s**: Try Kubernetes deployment
6. ✅ **Read Documentation**: Check individual README files

---

## Getting Help

- **API Documentation**: http://localhost:8000/docs
- **Kubernetes Guide**: `k8s/README.md`
- **Monitoring Guide**: `monitoring/README.md`
- **API Guide**: `app/README.md`
- **Task Summaries**: `TASK*_SUMMARY.md` files
- **Full Report**: `report/MLOps_Assignment_Report.md`

---

## Useful Links

- UCI Heart Disease Dataset: https://archive.ics.uci.edu/ml/datasets/heart+disease
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Kubernetes Documentation: https://kubernetes.io/docs/
- Prometheus Documentation: https://prometheus.io/docs/
- MLflow Documentation: https://mlflow.org/docs/

---

**Quick Start Guide Version**: 1.0
**Last Updated**: December 22, 2025
**For**: Heart Disease Prediction MLOps Project

---

*Happy Predicting! 🏥❤️*
