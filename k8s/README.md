# Kubernetes Deployment Guide

This directory contains Kubernetes manifests and scripts for deploying the Heart Disease Prediction API to Kubernetes clusters.

## Files Overview

| File | Description |
|------|-------------|
| `deployment.yaml` | Main deployment configuration with 3 replicas |
| `service.yaml` | Service definitions (LoadBalancer, ClusterIP, Headless) |
| `ingress.yaml` | Ingress configuration for external access |
| `hpa.yaml` | Horizontal Pod Autoscaler for auto-scaling |
| `rbac.yaml` | RBAC, ServiceAccount, and NetworkPolicy |
| `deploy.sh` | Automated deployment script |
| `cleanup.sh` | Resource cleanup script |

## Prerequisites

### Required Tools
- `kubectl` (v1.20+)
- `docker` (for building images)
- Kubernetes cluster (one of):
  - Minikube (local development)
  - Docker Desktop with Kubernetes
  - GKE/EKS/AKS (cloud providers)

### Optional Tools
- `helm` (for Helm deployments)
- `k9s` (cluster management UI)
- `kubectx/kubens` (context/namespace switching)

## Quick Start

### Option 1: Automated Deployment

```bash
# Make scripts executable
chmod +x k8s/*.sh

# Deploy to Kubernetes
./k8s/deploy.sh

# Access the API
kubectl port-forward svc/heart-disease-api-service 8000:80

# Test the API
curl http://localhost:8000/health
```

### Option 2: Manual Deployment

```bash
# 1. Build Docker image
docker build -t heart-disease-api:latest .

# 2. Load image (for Minikube)
minikube image load heart-disease-api:latest

# 3. Apply manifests
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# 4. Wait for deployment
kubectl rollout status deployment/heart-disease-api

# 5. Get service URL
kubectl get svc heart-disease-api-service
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Internet/Users                        │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │   Ingress Controller  │
         │   (nginx/traefik)     │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   LoadBalancer Svc    │
         │   (Port 80 → 8000)    │
         └───────────┬───────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼───┐        ┌───▼───┐        ┌───▼───┐
│ Pod 1 │        │ Pod 2 │        │ Pod 3 │
│  API  │        │  API  │        │  API  │
└───┬───┘        └───┬───┘        └───┬───┘
    │                │                │
    └────────────────┼────────────────┘
                     │
         ┌───────────▼───────────┐
         │  PersistentVolume     │
         │  (Model Storage)      │
         └───────────────────────┘
```

## Configuration

### Environment Variables (ConfigMap)

Edit `k8s/deployment.yaml` ConfigMap section:

```yaml
data:
  MODEL_NAME: "xgboost"  # or random_forest, logistic_regression
  LOG_LEVEL: "INFO"
  WORKERS: "2"
```

### Resource Limits

Default resources per pod:
- **Requests**: 250m CPU, 512Mi RAM
- **Limits**: 1 CPU, 1Gi RAM

Adjust in `deployment.yaml`:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

### Auto-scaling (HPA)

The HPA automatically scales pods based on:
- **CPU**: Target 70% utilization
- **Memory**: Target 80% utilization
- **Replicas**: Min 2, Max 10

Configure in `hpa.yaml`.

## Accessing the API

### Method 1: Port Forward (Development)

```bash
kubectl port-forward svc/heart-disease-api-service 8000:80

# Access at http://localhost:8000
curl http://localhost:8000/health
```

### Method 2: LoadBalancer (Cloud)

```bash
# Get external IP
kubectl get svc heart-disease-api-service

# Access via external IP
curl http://<EXTERNAL-IP>/health
```

### Method 3: Ingress (Production)

```bash
# Get ingress address
kubectl get ingress heart-disease-api-ingress

# Access via domain
curl http://heart-disease-api.local/health
```

### Method 4: Minikube

```bash
minikube service heart-disease-api-service
```

## Testing the Deployment

### 1. Check Pod Status

```bash
kubectl get pods -l app=heart-disease-api
```

Expected output:
```
NAME                                   READY   STATUS    RESTARTS   AGE
heart-disease-api-7d8c9f8b6d-abc12     1/1     Running   0          2m
heart-disease-api-7d8c9f8b6d-def34     1/1     Running   0          2m
heart-disease-api-7d8c9f8b6d-ghi56     1/1     Running   0          2m
```

### 2. Check Service

```bash
kubectl get svc heart-disease-api-service
```

### 3. View Logs

```bash
# All pods
kubectl logs -f -l app=heart-disease-api

# Specific pod
kubectl logs -f heart-disease-api-7d8c9f8b6d-abc12
```

### 4. Test Health Endpoint

```bash
# Port forward first
kubectl port-forward svc/heart-disease-api-service 8000:80

# Test health
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "xgboost",
  "timestamp": "2025-12-22T..."
}
```

### 5. Test Prediction Endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
  }'
```

## Monitoring

### View Metrics

```bash
# Pod metrics (requires metrics-server)
kubectl top pods -l app=heart-disease-api

# Node metrics
kubectl top nodes
```

### HPA Status

```bash
kubectl get hpa heart-disease-api-hpa

# Detailed view
kubectl describe hpa heart-disease-api-hpa
```

### Events

```bash
kubectl get events --sort-by='.lastTimestamp' | grep heart-disease-api
```

## Scaling

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment heart-disease-api --replicas=5

# Verify
kubectl get pods -l app=heart-disease-api
```

### Auto-scaling (HPA)

Auto-scaling is configured in `hpa.yaml`:
- Scales based on CPU and memory
- Min replicas: 2
- Max replicas: 10

Check HPA status:
```bash
kubectl get hpa
```

## Updating the Deployment

### Method 1: Rolling Update

```bash
# Build new image
docker build -t heart-disease-api:v1.1.0 .

# Update deployment
kubectl set image deployment/heart-disease-api \
  api=heart-disease-api:v1.1.0

# Monitor rollout
kubectl rollout status deployment/heart-disease-api
```

### Method 2: Apply New Manifest

```bash
# Edit deployment.yaml with new image tag
kubectl apply -f k8s/deployment.yaml

# Check rollout
kubectl rollout status deployment/heart-disease-api
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/heart-disease-api

# Rollback to specific revision
kubectl rollout undo deployment/heart-disease-api --to-revision=2

# View rollout history
kubectl rollout history deployment/heart-disease-api
```

## Troubleshooting

### Pods Not Starting

```bash
# Describe pod
kubectl describe pod <pod-name>

# Check events
kubectl get events --field-selector involvedObject.name=<pod-name>

# Check logs
kubectl logs <pod-name>
```

### ImagePullBackOff

```bash
# For Minikube - load image locally
minikube image load heart-disease-api:latest

# For remote registry - check credentials
kubectl create secret docker-registry regcred \
  --docker-server=<your-registry> \
  --docker-username=<username> \
  --docker-password=<password>
```

### Service Not Accessible

```bash
# Check service
kubectl get svc heart-disease-api-service

# Check endpoints
kubectl get endpoints heart-disease-api-service

# Test from within cluster
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://heart-disease-api-service.default.svc.cluster.local:8000/health
```

### High Memory/CPU Usage

```bash
# Check resource usage
kubectl top pods -l app=heart-disease-api

# Increase limits in deployment.yaml
kubectl apply -f k8s/deployment.yaml
```

## Cleanup

### Remove All Resources

```bash
# Automated
./k8s/cleanup.sh

# Manual
kubectl delete -f k8s/hpa.yaml
kubectl delete -f k8s/ingress.yaml
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml
kubectl delete -f k8s/rbac.yaml
```

## Production Considerations

### 1. Security
- [ ] Enable RBAC
- [ ] Use NetworkPolicies
- [ ] Scan images for vulnerabilities
- [ ] Use non-root containers
- [ ] Enable Pod Security Policies/Standards
- [ ] Use secrets for sensitive data
- [ ] Enable TLS/SSL

### 2. High Availability
- [ ] Run multiple replicas (3+)
- [ ] Use PodDisruptionBudgets
- [ ] Configure pod anti-affinity
- [ ] Use multiple availability zones
- [ ] Implement health checks

### 3. Monitoring & Logging
- [ ] Set up Prometheus metrics
- [ ] Configure Grafana dashboards
- [ ] Centralize logs (ELK/Loki)
- [ ] Set up alerting
- [ ] Track SLIs/SLOs

### 4. Resource Management
- [ ] Set appropriate resource requests/limits
- [ ] Configure HPA
- [ ] Use VPA for optimization
- [ ] Monitor resource usage

### 5. Backup & Recovery
- [ ] Backup PersistentVolumes
- [ ] Version control manifests
- [ ] Document recovery procedures
- [ ] Test disaster recovery

## Cloud-Specific Instructions

### Google Kubernetes Engine (GKE)

```bash
# Create cluster
gcloud container clusters create heart-disease-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-2

# Get credentials
gcloud container clusters get-credentials heart-disease-cluster

# Deploy
./k8s/deploy.sh
```

### Amazon EKS

```bash
# Create cluster
eksctl create cluster \
  --name heart-disease-cluster \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 3

# Deploy
./k8s/deploy.sh
```

### Azure AKS

```bash
# Create cluster
az aks create \
  --resource-group myResourceGroup \
  --name heart-disease-cluster \
  --node-count 3 \
  --node-vm-size Standard_D2s_v3

# Get credentials
az aks get-credentials \
  --resource-group myResourceGroup \
  --name heart-disease-cluster

# Deploy
./k8s/deploy.sh
```

## References

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)

---

**Author**: Saif Afzal
**Course**: MLOps (S1-25_AIMLCZG523)
**Institution**: BITS Pilani
**Date**: December 2025
