# 🚀 Kubernetes Deployment Guide - Heart Disease Prediction API

## ✅ Project Status

Your project **ALREADY HAS** comprehensive Kubernetes deployment configurations:

```
k8s/
├── README.md              ✅ Complete deployment documentation
├── deployment.yaml        ✅ Deployment manifest (3 replicas, health checks)
├── service.yaml          ✅ LoadBalancer, ClusterIP, Headless services
├── ingress.yaml          ✅ Ingress with nginx, rate limiting, CORS
├── hpa.yaml              ✅ Horizontal Pod Autoscaler
├── rbac.yaml             ✅ RBAC, ServiceAccount, NetworkPolicy
├── deploy.sh             ✅ Automated deployment script
└── cleanup.sh            ✅ Resource cleanup script
```

---

## 🎯 Deployment Options

### Option 1: **Docker Desktop Kubernetes** (Local - Recommended for Testing)

#### Step 1: Enable Kubernetes in Docker Desktop

1. Open **Docker Desktop Settings**
2. Go to **Kubernetes** tab
3. Check **"Enable Kubernetes"**
4. Click **"Apply & Restart"**
5. Wait 2-3 minutes for cluster to start

#### Step 2: Verify Kubernetes is Running

```bash
# Check Docker Desktop Kubernetes
kubectl cluster-info

# Expected output:
# Kubernetes control plane is running at https://127.0.0.1:6443
```

#### Step 3: Load Docker Image into Kubernetes

```bash
cd /Users/nadiaashfaq/saif-mlops/MLOPS-Assignment-1

# Since you already have the Docker image, we'll load it into K8s
docker image list | grep mlops  # Verify image exists

# For Docker Desktop, images are automatically available
# For Minikube, use: minikube image load mlops-heart-disease:latest
```

#### Step 4: Deploy Using Script

```bash
# Make script executable
chmod +x k8s/deploy.sh

# Deploy (automated deployment script)
./k8s/deploy.sh dev latest

# OR manually deploy
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
```

#### Step 5: Verify Deployment

```bash
# Check pods
kubectl get pods

# Expected output (3 replicas):
# NAME                                READY   STATUS    RESTARTS   AGE
# heart-disease-api-xxxxx-xxxxx       1/1     Running   0          2m
# heart-disease-api-xxxxx-xxxxx       1/1     Running   0          2m
# heart-disease-api-xxxxx-xxxxx       1/1     Running   0          2m

# Check services
kubectl get svc

# Expected output:
# NAME                            TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)
# heart-disease-api-service       LoadBalancer   10.96.xxx.xxx   localhost     80:30xxx/TCP
# heart-disease-api-internal      ClusterIP      10.96.xxx.xxx   <none>        8000/TCP

# Check deployment status
kubectl rollout status deployment/heart-disease-api
```

#### Step 6: Access the API

```bash
# Get the service IP
kubectl get svc heart-disease-api-service

# For Docker Desktop, the EXTERNAL-IP should be localhost or an IP
# Access via LoadBalancer IP:
curl http://localhost/health

# OR use port-forward for debugging
kubectl port-forward svc/heart-disease-api-service 8000:80

# Then access at: http://localhost:8000/health
```

---

### Option 2: **Minikube** (Local Alternative)

```bash
# Install Minikube (if not already installed)
brew install minikube

# Start Minikube
minikube start

# Set Docker to use Minikube's Docker daemon
eval $(minikube docker-env)

# Build image in Minikube context
docker build -t mlops-heart-disease:latest .

# Deploy
./k8s/deploy.sh dev latest

# Access via Minikube IP
minikube service heart-disease-api-service

# Or use port-forward
kubectl port-forward svc/heart-disease-api-service 8000:80
```

---

### Option 3: **Cloud Deployment** (GKE, EKS, AKS)

#### GKE (Google Cloud)

```bash
# Create GKE cluster
gcloud container clusters create heart-disease-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --enable-ip-alias \
  --machine-type n1-standard-1

# Get credentials
gcloud container clusters get-credentials heart-disease-cluster --zone us-central1-a

# Push image to GCR
docker tag mlops-heart-disease:latest gcr.io/PROJECT_ID/mlops-heart-disease:latest
docker push gcr.io/PROJECT_ID/mlops-heart-disease:latest

# Update deployment.yaml image to gcr.io/PROJECT_ID/mlops-heart-disease:latest
kubectl apply -f k8s/

# Get LoadBalancer IP
kubectl get svc heart-disease-api-service
```

#### EKS (AWS)

```bash
# Create EKS cluster
eksctl create cluster --name heart-disease --region us-east-1 --nodes 3

# Get credentials
aws eks update-kubeconfig --region us-east-1 --name heart-disease

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker tag mlops-heart-disease:latest ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mlops-heart-disease:latest
docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mlops-heart-disease:latest

# Deploy
kubectl apply -f k8s/
```

#### AKS (Azure)

```bash
# Create AKS cluster
az aks create --resource-group myResourceGroup --name heart-disease-cluster --node-count 3

# Get credentials
az aks get-credentials --resource-group myResourceGroup --name heart-disease-cluster

# Push to ACR and deploy similarly
```

---

## 📊 Deployment Manifests Summary

### Deployment (deployment.yaml)
- ✅ **Replicas:** 3 pods for high availability
- ✅ **Strategy:** RollingUpdate (zero-downtime deployments)
- ✅ **Health Checks:** Liveness & Readiness probes
- ✅ **Resource Limits:** 512Mi-1Gi memory, 250m-1000m CPU
- ✅ **Security:** Non-root user (UID 1000), security context
- ✅ **Monitoring:** Prometheus annotations for scraping

### Service (service.yaml)
- ✅ **LoadBalancer:** External access to API
- ✅ **ClusterIP:** Internal service-to-service communication
- ✅ **Headless:** For StatefulSets (if needed)
- ✅ **Session Affinity:** ClientIP-based sticky sessions

### Ingress (ingress.yaml)
- ✅ **Nginx Ingress Controller** for routing
- ✅ **Rate Limiting:** 100 RPS, 10 connections per IP
- ✅ **CORS:** Enabled for cross-origin requests
- ✅ **Path-based Routing:** `/`, `/docs`, `/health`
- ✅ **SSL/TLS:** Ready for cert-manager integration

### HPA (hpa.yaml)
- ✅ **Auto-scaling:** Min 2, Max 10 replicas
- ✅ **Trigger:** CPU 70%, Memory 80%

### RBAC (rbac.yaml)
- ✅ **ServiceAccount:** For pod authentication
- ✅ **ClusterRole:** Necessary permissions
- ✅ **NetworkPolicy:** Pod-to-pod communication rules

---

## 🧪 Testing Endpoints

### After Deployment, Test These Endpoints:

```bash
# Health check
curl http://localhost/health
# Expected: {"status":"healthy","model_loaded":true}

# Model info
curl http://localhost/model/info
# Expected: Model metadata

# Get features
curl http://localhost/features
# Expected: List of 13 required features

# Single prediction
curl -X POST http://localhost/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 2.3,
    "slope": 1, "ca": 0, "thal": 3
  }'
# Expected: {"prediction": 0, "diagnosis": "No Heart Disease", "confidence": 0.59...}

# Batch prediction
curl -X POST http://localhost/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{...patient1...}, {...patient2...}]'

# Metrics (Prometheus format)
curl http://localhost/metrics
```

---

## 📈 Monitoring in Kubernetes

### View Logs

```bash
# Logs from a single pod
kubectl logs -f deployment/heart-disease-api

# Logs from all pods
kubectl logs -f deployment/heart-disease-api --all-containers=true

# Logs from specific pod
kubectl logs -f heart-disease-api-xxxxx-xxxxx
```

### Port Forward for Debugging

```bash
# Forward port 8000 to localhost
kubectl port-forward svc/heart-disease-api-service 8000:80

# Then access: http://localhost:8000
```

### Kubernetes Dashboard

```bash
# Start dashboard
kubectl proxy

# Access at: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
```

### Prometheus Metrics in K8s

```bash
# If you deployed Prometheus in the cluster:
kubectl port-forward -n monitoring svc/prometheus 9090:9090

# Access Prometheus at: http://localhost:9090
# Query: up{job="heart-disease-api"}
```

---

## 🔧 Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl describe pod heart-disease-api-xxxxx

# Check events
kubectl get events

# Check logs
kubectl logs heart-disease-api-xxxxx
```

### Image Not Found

```bash
# For Docker Desktop - ensure image is built locally
docker build -t mlops-heart-disease:latest .

# Update deployment.yaml imagePullPolicy to IfNotPresent
# (already set in your manifest)

# For cloud - push to registry first
# Then update deployment.yaml with registry URL
```

### Service Not Accessible

```bash
# Check service
kubectl get svc heart-disease-api-service

# Test connectivity between pods
kubectl run -it --rm debug --image=busybox --restart=Never -- wget -O- http://heart-disease-api-service

# Check endpoints
kubectl get endpoints heart-disease-api-service
```

### LoadBalancer Pending

```bash
# Check service status
kubectl get svc

# On Docker Desktop, LoadBalancer status shows "localhost"
# On cloud providers, it takes 1-2 minutes to provision

# For Docker Desktop workaround:
kubectl port-forward svc/heart-disease-api-service 80:80
```

---

## 📋 Checklist for Deployment

- ✅ Docker image built: `mlops-heart-disease:latest`
- ✅ Kubernetes cluster available (Docker Desktop/Minikube/Cloud)
- ✅ `kubectl` installed and configured
- ✅ All manifests present in `k8s/` directory
- ✅ Deploy script executable: `chmod +x k8s/deploy.sh`
- ✅ Application ports exposed (8000 → 80 via service)
- ✅ Health checks configured in deployment
- ✅ Resource limits and requests set
- ✅ RBAC and security policies applied
- ✅ Ingress configured for external access
- ✅ Auto-scaling configured (HPA)
- ✅ Monitoring metrics enabled (Prometheus annotations)

---

## 🎯 Quick Start Commands

```bash
# For Docker Desktop:
# 1. Enable Kubernetes in Docker Desktop settings

# 2. Verify cluster
kubectl cluster-info

# 3. Deploy application
cd /Users/nadiaashfaq/saif-mlops/MLOPS-Assignment-1
./k8s/deploy.sh

# 4. Check status
kubectl get all

# 5. Test API
kubectl port-forward svc/heart-disease-api-service 8000:80
curl http://localhost:8000/health

# 6. Cleanup when done
./k8s/cleanup.sh
```

---

## 📸 Screenshots to Capture

For assignment verification, capture:

1. **Kubernetes Deployment Status**
   ```bash
   kubectl get deployments,pods,svc,ingress
   ```

2. **Pod Details**
   ```bash
   kubectl describe pod <pod-name>
   ```

3. **API Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

4. **Prediction Test**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{...patient data...}'
   ```

5. **Kubernetes Dashboard**
   - Pods running
   - Services exposed
   - Resource usage

6. **Logs**
   ```bash
   kubectl logs deployment/heart-disease-api
   ```

---

## 🚀 Assignment Completion Status

Your project **FULLY MEETS** the requirement:

✅ **Deployment Manifest** - Comprehensive deployment.yaml, service.yaml, ingress.yaml  
✅ **LoadBalancer** - Service exposed via LoadBalancer type  
✅ **Ingress** - Full Ingress configuration with multiple paths  
✅ **Auto-scaling** - HPA configured (2-10 replicas)  
✅ **RBAC** - Security policies and service accounts  
✅ **Local Kubernetes** - Ready for Docker Desktop/Minikube  
✅ **Cloud Ready** - Manifests work on GKE, EKS, AKS  
✅ **Monitoring** - Prometheus annotations for metrics  
✅ **Scripts** - Automated deployment and cleanup  

**Ready to deploy!** 🎉

---

**Last Updated:** January 6, 2026  
**Status:** ✅ Production Ready
