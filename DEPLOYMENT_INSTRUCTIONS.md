# Heart Disease MLOps - Deployment Instructions

## Quick Access URLs

- **GitHub Repository**: https://github.com/saif2024-bits/MLOPS-Assignment-1
- **CI/CD Pipeline**: https://github.com/saif2024-bits/MLOPS-Assignment-1/actions

---

## Deployment Options

### Option 1: Docker (Recommended for Testing)

#### Prerequisites:
- Docker installed (https://docs.docker.com/get-docker/)
- Git installed

#### Steps:

```bash
# 1. Clone the repository
git clone https://github.com/saif2024-bits/MLOPS-Assignment-1.git
cd MLOPS-Assignment-1

# 2. Build Docker image
docker build -t heart-disease-api:latest .

# 3. Run container
docker run -d \
  -p 8000:8000 \
  --name heart-api \
  heart-disease-api:latest

# 4. Verify deployment
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "model_name": "xgboost",
#   ...
# }
```

#### Access Points:
- **API Health**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs
- **Redoc**: http://localhost:8000/redoc
- **Metrics**: http://localhost:8000/metrics

#### Test Prediction:
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

Expected Response:
```json
{
  "prediction": 0,
  "diagnosis": "No Heart Disease",
  "confidence": 0.937,
  "probabilities": {
    "no_disease": 0.937,
    "disease": 0.063
  },
  "model_used": "xgboost",
  "timestamp": "2026-01-05T...",
  "model_performance": {
    "accuracy": 0.869,
    "roc_auc": 0.961
  }
}
```

#### Cleanup:
```bash
docker stop heart-api
docker rm heart-api
docker rmi heart-disease-api:latest
```

---

### Option 2: Docker Compose (API + Monitoring)

#### Prerequisites:
- Docker Compose installed

#### Steps:

```bash
# 1. Clone repository
git clone https://github.com/saif2024-bits/MLOPS-Assignment-1.git
cd MLOPS-Assignment-1

# 2. Start all services
docker-compose -f docker-compose.monitoring.yml up -d

# 3. Verify services
docker-compose -f docker-compose.monitoring.yml ps
```

#### Access Points:
- **API**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **AlertManager**: http://localhost:9093

#### Cleanup:
```bash
docker-compose -f docker-compose.monitoring.yml down
```

---

### Option 3: Local Python Environment

#### Prerequisites:
- Python 3.11+
- pip

#### Steps:

```bash
# 1. Clone repository
git clone https://github.com/saif2024-bits/MLOPS-Assignment-1.git
cd MLOPS-Assignment-1

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data
python data/download_data.py

# 5. Train models
python src/train.py

# 6. Run API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Access at http://localhost:8000/docs
```

#### Alternative - MLflow Training:
```bash
# Train with MLflow tracking
python src/train_mlflow.py

# View experiments
mlflow ui
# Open http://localhost:5000
```

---

### Option 4: Kubernetes Deployment

#### Prerequisites:
- Kubernetes cluster (Minikube, Docker Desktop, or cloud)
- kubectl installed

#### Steps:

```bash
# 1. Clone repository
git clone https://github.com/saif2024-bits/MLOPS-Assignment-1.git
cd MLOPS-Assignment-1

# 2. Deploy to Kubernetes
./k8s/deploy.sh

# 3. Verify deployment
kubectl get pods
kubectl get services
kubectl get hpa

# 4. Access API
# If using Minikube:
minikube service heart-disease-api

# If using port-forward:
kubectl port-forward service/heart-disease-api 8000:8000
# Access at http://localhost:8000/docs

# 5. Check scaling
kubectl get hpa heart-disease-api-hpa --watch
```

#### Cleanup:
```bash
./k8s/cleanup.sh
```

---

## Testing the Deployment

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. API Documentation
Open browser: http://localhost:8000/docs

### 3. Make Predictions

#### High Risk Patient:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

#### Low Risk Patient:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45, "sex": 0, "cp": 1, "trestbps": 120,
    "chol": 200, "fbs": 0, "restecg": 0, "thalach": 170,
    "exang": 0, "oldpeak": 0.5, "slope": 2, "ca": 0, "thal": 3
  }'
```

### 4. Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"age": 63, "sex": 1, "cp": 3, ...},
    {"age": 45, "sex": 0, "cp": 1, ...}
  ]'
```

### 5. Model Information
```bash
curl http://localhost:8000/model/info
```

### 6. Metrics (Prometheus format)
```bash
curl http://localhost:8000/metrics
```

---

## Running Tests

### Unit Tests:
```bash
pytest tests/ -v --cov=src --cov=app
```

### API Tests:
```bash
python app/test_api.py
```

### Integration Tests (requires running API):
```bash
# Terminal 1: Start API
uvicorn app.main:app

# Terminal 2: Run tests
pytest tests/test_model_pipeline.py -v
```

---

## CI/CD Pipeline

The project includes a complete GitHub Actions CI/CD pipeline:

### Pipeline Stages:
1. **Lint** - Code quality checks
2. **Test** - Unit tests with 93% coverage
3. **Train Model** - Automated training
4. **MLflow Tracking** - Experiment tracking
5. **Integration Test** - End-to-end testing
6. **Security Scan** - Dependency vulnerabilities

### View Pipeline:
https://github.com/saif2024-bits/MLOPS-Assignment-1/actions

### Trigger Pipeline:
```bash
git commit -m "Your changes"
git push origin main
```

---

## Monitoring

### Prometheus Metrics:
```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access Prometheus
open http://localhost:9090
```

### Grafana Dashboard:
```bash
# Access Grafana
open http://localhost:3000
# Login: admin/admin

# Import dashboard: monitoring/grafana/dashboards/heart-disease-api-dashboard.json
```

### Available Metrics:
- `api_requests_total` - Total API requests
- `api_request_duration_seconds` - Request latency
- `api_predictions_total` - Total predictions
- `model_prediction_score` - Prediction confidence
- `model_performance_accuracy` - Model accuracy
- `model_performance_roc_auc` - Model ROC-AUC

---

## Troubleshooting

### Issue: Docker build fails
```bash
# Clear Docker cache
docker system prune -a

# Rebuild
docker build --no-cache -t heart-disease-api:latest .
```

### Issue: Port already in use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
docker run -p 8001:8000 heart-disease-api:latest
```

### Issue: Models not loading
```bash
# Ensure models exist
ls -la models/*.pkl

# If missing, train models
python src/train.py
```

### Issue: Import errors
```bash
# Ensure correct directory
cd MLOPS-Assignment-1

# Verify Python path
python -c "import sys; print(sys.path)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## Production Deployment Considerations

### Cloud Platforms:

#### AWS:
- **ECS/Fargate**: Deploy Docker container
- **EKS**: Kubernetes deployment
- **Lambda**: Serverless (requires refactoring)

#### Google Cloud:
- **Cloud Run**: Serverless containers
- **GKE**: Kubernetes deployment
- **App Engine**: Platform-as-a-Service

#### Azure:
- **Container Instances**: Simple container deployment
- **AKS**: Kubernetes deployment
- **App Service**: Web app deployment

### Configuration:
Update environment variables for production:
```bash
# Environment variables
MODEL_DIR=/app/models
MODEL_NAME=xgboost
LOG_LEVEL=INFO
WORKERS=4
```

### Security:
- Enable HTTPS/TLS
- Add authentication (API keys, OAuth)
- Set up rate limiting
- Configure CORS properly
- Use secrets management (AWS Secrets Manager, etc.)

---

## Support

For issues or questions:
- **GitHub Issues**: https://github.com/saif2024-bits/MLOPS-Assignment-1/issues
- **Documentation**: See README.md and QUICK_START.md
- **Email**: 2024aa05546@wilp.bits-pilani.ac.in

---

**Last Updated**: 2026-01-05
**Version**: 1.0
**Status**: Production Ready ✅
