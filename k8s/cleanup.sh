#!/bin/bash

###############################################################################
# Heart Disease Prediction API - Kubernetes Cleanup Script
#
# This script removes all Kubernetes resources for the Heart Disease API
#
# Usage:
#   ./k8s/cleanup.sh
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
NAMESPACE="default"
APP_NAME="heart-disease-api"

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo "=========================================="
echo "  Heart Disease API Cleanup"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo "=========================================="
echo ""

print_info "Deleting Kubernetes resources..."

# Delete HPA
print_info "Deleting HPA..."
kubectl delete -f k8s/hpa.yaml -n $NAMESPACE --ignore-not-found=true

# Delete Ingress
print_info "Deleting Ingress..."
kubectl delete -f k8s/ingress.yaml -n $NAMESPACE --ignore-not-found=true

# Delete Service
print_info "Deleting Service..."
kubectl delete -f k8s/service.yaml -n $NAMESPACE --ignore-not-found=true

# Delete Deployment
print_info "Deleting Deployment..."
kubectl delete -f k8s/deployment.yaml -n $NAMESPACE --ignore-not-found=true

# Delete RBAC
print_info "Deleting RBAC resources..."
kubectl delete -f k8s/rbac.yaml -n $NAMESPACE --ignore-not-found=true

# Wait for pods to terminate
print_info "Waiting for pods to terminate..."
kubectl wait --for=delete pod -l app=$APP_NAME -n $NAMESPACE --timeout=60s || true

print_success "All resources deleted successfully!"

# Show remaining resources
print_info "Remaining resources (should be empty):"
kubectl get all -n $NAMESPACE -l app=$APP_NAME

echo ""
echo "=========================================="
print_success "Cleanup completed!"
echo "=========================================="
