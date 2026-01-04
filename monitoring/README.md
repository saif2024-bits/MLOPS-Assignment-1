# Monitoring & Logging Setup

## Overview

This directory contains the monitoring and logging infrastructure for the Heart Disease Prediction API using Prometheus and Grafana.

## Components

### 1. Prometheus
- **Port**: 9090
- **Purpose**: Metrics collection and storage
- **Retention**: 15 days
- **Scrape Interval**: 15 seconds

### 2. Grafana
- **Port**: 3000
- **Purpose**: Metrics visualization and dashboards
- **Default Credentials**:
  - Username: `admin`
  - Password: `admin`

### 3. AlertManager
- **Port**: 9093
- **Purpose**: Alert routing and notification

### 4. Node Exporter
- **Port**: 9100
- **Purpose**: System-level metrics (CPU, memory, disk)

## Quick Start

### 1. Start Monitoring Stack

```bash
# Build and start all services
docker-compose -f docker-compose.monitoring.yml up -d

# Check service status
docker-compose -f docker-compose.monitoring.yml ps

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f
```

### 2. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| API | http://localhost:8000 | - |
| API Docs | http://localhost:8000/docs | - |
| API Metrics | http://localhost:8000/metrics | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin/admin |
| AlertManager | http://localhost:9093 | - |

### 3. View Grafana Dashboard

1. Open http://localhost:3000
2. Login with admin/admin
3. Navigate to Dashboards → Heart Disease Prediction API Dashboard
4. View real-time metrics and visualizations

## Metrics Collected

### API Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `api_requests_total` | Counter | Total number of API requests by method, endpoint, and status |
| `api_request_duration_seconds` | Histogram | Request duration in seconds by method and endpoint |
| `predictions_total` | Counter | Total predictions made by prediction class |
| `prediction_confidence` | Histogram | Distribution of prediction confidence scores |
| `batch_prediction_size` | Histogram | Size distribution of batch predictions |
| `model_load_time_seconds` | Gauge | Time taken to load the ML model |
| `active_requests` | Gauge | Number of currently active requests |
| `api_errors_total` | Counter | Total API errors by error type and endpoint |
| `api_health_status` | Gauge | API health status (1=healthy, 0=unhealthy) |

### System Metrics (from Node Exporter)

- CPU usage
- Memory usage
- Disk I/O
- Network I/O
- Process statistics

## Dashboard Panels

The Grafana dashboard includes the following panels:

### Overview
1. **API Health Status** - Current health (healthy/unhealthy)
2. **Request Rate** - Requests per second
3. **Total Predictions** - Cumulative prediction count
4. **P95 Latency** - 95th percentile response time

### Performance
5. **Request Rate by Endpoint** - Traffic breakdown by API endpoint
6. **Request Latency Percentiles** - P50, P95, P99 latencies over time

### ML Insights
7. **Predictions by Class** - Pie chart of predictions (heart disease vs. no disease)
8. **Prediction Confidence** - Confidence score distribution over time

### Errors & Load
9. **Error Rate by Type** - Error breakdown by type
10. **Active Requests** - Concurrent request load

## Alerts Configured

### Critical Alerts
- **APIDown**: API has been down for > 1 minute
- **ModelNotLoaded**: ML model failed to load

### Warning Alerts
- **HighErrorRate**: Error rate > 5% for 5 minutes
- **HighRequestLatency**: P95 latency > 2s for 5 minutes
- **HighMemoryUsage**: Memory usage > 90% for 5 minutes
- **HighCPUUsage**: CPU usage > 80% for 5 minutes
- **TooManyActiveRequests**: > 50 concurrent requests for 5 minutes

### Info Alerts
- **LowPredictionConfidence**: Median confidence < 60% for 10 minutes

## Monitoring Best Practices

### 1. Regular Checks

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check metrics endpoint
curl http://localhost:8000/metrics

# Query specific metric
curl -G http://localhost:9090/api/v1/query --data-urlencode 'query=api_requests_total'
```

### 2. Alert Verification

```bash
# Check active alerts
curl http://localhost:9090/api/v1/alerts

# Check AlertManager status
curl http://localhost:9093/api/v1/status
```

### 3. Log Analysis

```bash
# View API logs
docker-compose -f docker-compose.monitoring.yml logs -f api

# View Prometheus logs
docker-compose -f docker-compose.monitoring.yml logs -f prometheus

# View Grafana logs
docker-compose -f docker-compose.monitoring.yml logs -f grafana
```

## Testing the Monitoring

### 1. Generate Test Traffic

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
  }'

# Health check
curl http://localhost:8000/health

# Load test (requires 'hey' or 'ab')
hey -n 1000 -c 10 http://localhost:8000/health
```

### 2. Verify Metrics

```bash
# Check metrics are being collected
curl http://localhost:8000/metrics | grep api_requests_total

# Verify Prometheus is scraping
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="heart-disease-api")'
```

### 3. View in Grafana

1. Go to http://localhost:3000
2. Navigate to Heart Disease API Dashboard
3. You should see:
   - Request rate increasing
   - Latency metrics
   - Prediction counts
   - No errors (hopefully!)

## Customization

### Adding New Metrics

1. **Define metric in `app/monitoring.py`**:
```python
NEW_METRIC = Counter(
    'new_metric_name',
    'Description of metric',
    ['label1', 'label2'],
    registry=registry
)
```

2. **Track metric in `app/main.py`**:
```python
from app.monitoring import NEW_METRIC
NEW_METRIC.labels(label1='value1', label2='value2').inc()
```

3. **Add to Grafana dashboard**:
   - Edit `monitoring/grafana/dashboards/heart-disease-api-dashboard.json`
   - Add new panel with PromQL query

### Modifying Alerts

Edit `monitoring/prometheus/alerts.yml`:

```yaml
- alert: MyNewAlert
  expr: my_metric > threshold
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Alert summary"
    description: "Alert description"
```

### Configuring Notifications

Edit `monitoring/alertmanager/config.yml`:

```yaml
receivers:
  - name: 'email-notifications'
    email_configs:
      - to: 'team@example.com'
        from: 'alerts@example.com'
        smarthost: smtp.gmail.com:587
        auth_username: 'alerts@example.com'
        auth_password: 'password'
```

## Troubleshooting

### Prometheus Not Scraping

```bash
# Check Prometheus logs
docker-compose -f docker-compose.monitoring.yml logs prometheus

# Verify API is reachable
docker exec prometheus wget -O- http://api:8000/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq
```

### Grafana Dashboard Not Showing Data

1. Check Prometheus datasource is configured
2. Verify Prometheus is collecting metrics
3. Check dashboard panel queries for errors
4. Ensure time range is appropriate

### Alerts Not Firing

```bash
# Check alert rules
curl http://localhost:9090/api/v1/rules

# Verify AlertManager is receiving alerts
curl http://localhost:9093/api/v1/alerts
```

## Production Considerations

### 1. Persistent Storage

Ensure volumes are backed up:
```bash
docker volume ls | grep monitoring
docker run --rm -v prometheus_data:/data -v $(pwd):/backup ubuntu tar czf /backup/prometheus-backup.tar.gz /data
```

### 2. Security

- Change default Grafana password
- Enable HTTPS for all services
- Restrict network access
- Use secrets for sensitive configuration

### 3. Retention

Adjust retention based on storage:
```yaml
# In prometheus command args
- '--storage.tsdb.retention.time=30d'  # Keep data for 30 days
```

### 4. High Availability

For production, consider:
- Prometheus HA with Thanos
- Grafana clustering
- Alert deduplication
- Remote storage backends

## Cleanup

```bash
# Stop all services
docker-compose -f docker-compose.monitoring.yml down

# Remove volumes (CAUTION: deletes all data)
docker-compose -f docker-compose.monitoring.yml down -v

# Remove specific volume
docker volume rm prometheus_data
```

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [AlertManager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [Prometheus Client Python](https://github.com/prometheus/client_python)

---

**Created**: December 2025
**Author**: Saif Afzal
**Course**: MLOps (S1-25_AIMLCZG523)
**Institution**: BITS Pilani
