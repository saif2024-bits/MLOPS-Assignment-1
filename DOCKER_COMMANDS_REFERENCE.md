# 🚀 Docker & Deployment Commands Reference

## Quick Start

### Start All Services
```bash
cd /Users/nadiaashfaq/saif-mlops/MLOPS-Assignment-1
docker compose -f docker-compose.monitoring.yml up -d
```

### Stop All Services
```bash
docker compose -f docker-compose.monitoring.yml down
```

### View Logs
```bash
# API logs
docker logs heart-disease-api -f

# Prometheus logs
docker logs prometheus -f

# Grafana logs
docker logs grafana -f

# All logs
docker compose -f docker-compose.monitoring.yml logs -f
```

### Check Status
```bash
docker compose -f docker-compose.monitoring.yml ps
```

## Access Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | N/A |
| AlertManager | http://localhost:9093 | N/A |
| Node Exporter | http://localhost:9100 | N/A |

## Testing API (from Docker network)
```bash
# Health check
docker exec heart-disease-api python -c "import requests; print(requests.get('http://localhost:8000/health').json())"

# Single prediction
docker exec heart-disease-api python << 'EOF'
import requests
import json

payload = {
    "age": 45, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 2.3,
    "slope": 1, "ca": 0, "thal": 3
}

response = requests.post('http://localhost:8000/predict', json=payload)
print(json.dumps(response.json(), indent=2))
EOF
```

## Docker Management

### Build Image
```bash
docker build -t mlops-heart-disease:latest .
```

### List Images
```bash
docker images | grep mlops
```

### Remove Old Containers
```bash
docker rm -f $(docker ps -aq --filter "name=mlops")
```

### Prune System
```bash
docker system prune -a --volumes
```

## Environment Troubleshooting

### Check Docker Status
```bash
docker version
docker ps
docker stats
```

### View Container Inspect
```bash
docker inspect heart-disease-api | python -m json.tool
```

### Container Resource Limits
```bash
# CPU and memory usage
docker stats --no-stream
```

## Data & Models

### Dataset Location
```
data/heart_disease.csv          # Raw dataset
data/heart_disease_clean.csv    # Cleaned dataset (used by API)
```

### Model Files
```
models/preprocessing_pipeline.pkl    # Preprocessing
models/logistic_regression_model.pkl # Logistic Regression
models/random_forest_model.pkl       # Random Forest
models/xgboost_model.pkl            # XGBoost (selected)
```

### Logs
```
logs/api.log    # API application logs
```

## Monitoring Queries

### Prometheus Queries
```
# Check if services are up
up

# API request rate
rate(http_requests_total[5m])

# API response latency
histogram_quantile(0.95, http_request_duration_seconds_bucket)

# Model predictions per minute
rate(model_predictions_total[1m])
```

### Grafana Dashboard
Access at http://localhost:3000 and explore:
- API Request Metrics
- Model Performance
- System Resources
- Error Rates

## Production Deployment

### Using Docker Compose Only
```bash
docker compose -f docker-compose.yml up -d
```

### Using Kubernetes (from k8s/ folder)
```bash
# Deploy
./k8s/deploy.sh

# View status
kubectl get pods
kubectl get services

# Cleanup
./k8s/cleanup.sh
```

## Common Issues & Solutions

### Port Already in Use
```bash
# Find what's using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Container Won't Start
```bash
# Check logs
docker logs <container_name>

# Run with interactive mode
docker run -it mlops-heart-disease:latest /bin/bash
```

### Prometheus Not Scraping
```bash
# Check targets in Prometheus UI
# http://localhost:9090/targets

# Or via API
curl -s http://localhost:9090/api/v1/targets | python -m json.tool
```

### Out of Disk Space
```bash
docker system prune --all --volumes --force
```

## Performance Tuning

### Increase Log Level for Debugging
```bash
docker exec heart-disease-api env LOGLEVEL=DEBUG uvicorn app.main:app
```

### Monitor Resource Usage
```bash
watch docker stats
```

### Check Network Connectivity
```bash
docker exec heart-disease-api ping prometheus
docker exec heart-disease-api ping grafana
```

## Backup & Restore

### Backup Prometheus Data
```bash
docker cp prometheus:/prometheus/data ./prometheus_backup
```

### Backup Grafana Dashboards
```bash
docker exec grafana grafana-cli admin export-dashboard > dashboard_backup.json
```

## Final Submission

### Convert Report to .docx
```bash
pandoc report/MLOps_Assignment_Report.md -o report/MLOps_Assignment_Report.docx
```

### Verify All Deliverables
```bash
# Check file existence
ls -la {app,src,models,data,tests,notebooks,k8s,monitoring}/*
ls -la {Dockerfile,docker-compose*.yml,requirements*.txt,environment.yml}
cat DELIVERABLES_CHECKLIST.md
```

### Create Demo Video
- Record 7-8 minute video following script in DELIVERABLES_CHECKLIST.md
- Show data download → EDA → training → tests → Docker → K8s
- Upload to YouTube or add to repository

---

**Last Updated:** January 5, 2026  
**Docker Version:** 29.1.3  
**Status:** ✅ Production Ready
