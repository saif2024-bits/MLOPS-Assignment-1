#!/bin/bash

###############################################################################
# Heart Disease Prediction API - Kubernetes Deployment Script
#
# This script deploys the Heart Disease Prediction API to Kubernetes
#
# Usage:
#   ./k8s/deploy.sh [environment]
#
# Arguments:
#   environment: dev|staging|prod (default: dev)
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT="${1:-dev}"
NAMESPACE="default"
APP_NAME="heart-disease-api"
IMAGE_NAME="heart-disease-api"
IMAGE_TAG="${2:-latest}"

# Print functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Header
echo "=========================================="
echo "  Heart Disease API Deployment"
echo "=========================================="
echo "Environment: $ENVIRONMENT"
echo "Namespace: $NAMESPACE"
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "=========================================="
echo ""

# Check prerequisites
print_info "Checking prerequisites..."

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if kubectl can connect to cluster
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
    exit 1
fi

print_success "Prerequisites check passed"

# Create namespace if it doesn't exist
print_info "Ensuring namespace exists..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
print_success "Namespace ready: $NAMESPACE"

# Build Docker image
print_info "Building Docker image..."
docker build -t $IMAGE_NAME:$IMAGE_TAG ..

# Tag image for registry (if using remote registry)
# docker tag $IMAGE_NAME:$IMAGE_TAG your-registry.com/$IMAGE_NAME:$IMAGE_TAG
# docker push your-registry.com/$IMAGE_NAME:$IMAGE_TAG

print_success "Docker image built: $IMAGE_NAME:$IMAGE_TAG"

# Load image into minikube (if using minikube)
if command -v minikube &> /dev/null && minikube status &> /dev/null; then
    print_info "Detected Minikube - loading image..."
    minikube image load $IMAGE_NAME:$IMAGE_TAG
    print_success "Image loaded into Minikube"
fi

# Apply Kubernetes manifests
print_info "Applying Kubernetes manifests..."

# Apply RBAC
print_info "Applying RBAC configuration..."
kubectl apply -f k8s/rbac.yaml -n $NAMESPACE

# Apply ConfigMap and PVC
print_info "Applying configuration..."
kubectl apply -f k8s/deployment.yaml -n $NAMESPACE

# Apply Deployment
print_info "Applying deployment..."
kubectl apply -f k8s/deployment.yaml -n $NAMESPACE

# Wait for deployment to be ready
print_info "Waiting for deployment to be ready..."
kubectl rollout status deployment/$APP_NAME -n $NAMESPACE --timeout=300s

# Apply Service
print_info "Applying service..."
kubectl apply -f k8s/service.yaml -n $NAMESPACE

# Apply Ingress (optional)
if [ -f "k8s/ingress.yaml" ]; then
    print_info "Applying ingress..."
    kubectl apply -f k8s/ingress.yaml -n $NAMESPACE
fi

# Apply HPA (optional)
if [ -f "k8s/hpa.yaml" ]; then
    print_info "Applying HPA..."
    kubectl apply -f k8s/hpa.yaml -n $NAMESPACE
fi

print_success "All manifests applied successfully"

# Get deployment status
print_info "Deployment status:"
kubectl get deployments -n $NAMESPACE -l app=$APP_NAME

print_info "Pod status:"
kubectl get pods -n $NAMESPACE -l app=$APP_NAME

print_info "Service status:"
kubectl get services -n $NAMESPACE -l app=$APP_NAME

# Get service URL
print_info "Getting service URL..."

# For LoadBalancer
EXTERNAL_IP=$(kubectl get svc heart-disease-api-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -z "$EXTERNAL_IP" ]; then
    EXTERNAL_IP=$(kubectl get svc heart-disease-api-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
fi

# For Minikube
if command -v minikube &> /dev/null && minikube status &> /dev/null; then
    MINIKUBE_IP=$(minikube ip)
    NODE_PORT=$(kubectl get svc heart-disease-api-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
    print_success "API accessible at: http://$MINIKUBE_IP:$NODE_PORT"
    print_info "To test: curl http://$MINIKUBE_IP:$NODE_PORT/health"
elif [ -n "$EXTERNAL_IP" ]; then
    print_success "API accessible at: http://$EXTERNAL_IP"
    print_info "To test: curl http://$EXTERNAL_IP/health"
else
    print_warning "Service URL not yet available. Use 'kubectl get svc' to check status."
    print_info "For port-forward: kubectl port-forward -n $NAMESPACE svc/heart-disease-api-service 8000:80"
fi

# Print useful commands
echo ""
echo "=========================================="
echo "  Useful Commands"
echo "=========================================="
echo "View logs:"
echo "  kubectl logs -f -n $NAMESPACE -l app=$APP_NAME"
echo ""
echo "Get pods:"
echo "  kubectl get pods -n $NAMESPACE -l app=$APP_NAME"
echo ""
echo "Describe deployment:"
echo "  kubectl describe deployment/$APP_NAME -n $NAMESPACE"
echo ""
echo "Port forward (for local access):"
echo "  kubectl port-forward -n $NAMESPACE svc/heart-disease-api-service 8000:80"
echo ""
echo "Scale deployment:"
echo "  kubectl scale deployment/$APP_NAME -n $NAMESPACE --replicas=5"
echo ""
echo "Delete deployment:"
echo "  ./k8s/cleanup.sh"
echo "=========================================="

print_success "Deployment completed successfully!"
