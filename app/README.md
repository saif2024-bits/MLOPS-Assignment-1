# Heart Disease Prediction API

FastAPI-based REST API for predicting heart disease using machine learning models.

## Features

- **RESTful API** with FastAPI
- **Pydantic validation** for input/output
- **Batch predictions** support
- **Interactive API documentation** (Swagger UI)
- **Health check endpoints**
- **CORS enabled** for frontend integration
- **Docker containerized** for easy deployment
- **Production-ready** with proper logging and error handling

## API Endpoints

### General

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

### Model Information

- `GET /model/info` - Get model information and performance metrics
- `GET /features` - Get list of required features with descriptions

### Predictions

- `POST /predict` - Make a single prediction
- `POST /predict/batch` - Make batch predictions (up to 100 patients)

## Quick Start

### Option 1: Local Development

```bash
# Install dependencies
pip install -r requirements_production.txt

# Run the API
python app/main.py

# OR use uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Docker

```bash
# Build Docker image
docker build -t heart-disease-api .

# Run container
docker run -p 8000:8000 heart-disease-api

# OR use docker-compose
docker-compose up --build
```

## API Usage Examples

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
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

Response:
```json
{
  "prediction": 0,
  "diagnosis": "No Heart Disease",
  "confidence": 0.954,
  "probabilities": {
    "no_disease": 0.954,
    "disease": 0.046
  },
  "model_used": "xgboost",
  "timestamp": "2025-12-22T10:30:45.123456",
  "model_performance": {
    "test_accuracy": 0.8689,
    "test_roc_auc": 0.9610
  }
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
      },
      {
        "age": 67, "sex": 1, "cp": 4, "trestbps": 160,
        "chol": 286, "fbs": 0, "restecg": 2, "thalach": 108,
        "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3
      }
    ]
  }'
```

### Python Client Example

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Patient data
patient = {
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
}

# Make prediction
response = requests.post(url, json=patient)
result = response.json()

print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Feature Descriptions

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| age | int | 1-120 | Age in years |
| sex | int | 0-1 | Sex (1 = male, 0 = female) |
| cp | int | 1-4 | Chest pain type |
| trestbps | int | 80-200 | Resting blood pressure (mm Hg) |
| chol | int | 100-600 | Serum cholesterol (mg/dl) |
| fbs | int | 0-1 | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| restecg | int | 0-2 | Resting ECG results |
| thalach | int | 60-220 | Maximum heart rate achieved |
| exang | int | 0-1 | Exercise induced angina (1 = yes, 0 = no) |
| oldpeak | float | 0-10 | ST depression induced by exercise |
| slope | int | 1-3 | Slope of peak exercise ST segment |
| ca | int | 0-3 | Number of major vessels colored by fluoroscopy |
| thal | int | 3,6,7 | Thalassemia |

## Testing

### Run API Tests

```bash
# Make sure API is running first
python app/main.py

# In another terminal, run tests
python app/test_api.py
```

### Manual Testing

Visit http://localhost:8000/docs for interactive Swagger UI documentation where you can test all endpoints directly in your browser.

## Environment Variables

- `MODEL_DIR` - Directory containing model files (default: `models`)
- `MODEL_NAME` - Model name to load (default: `xgboost`)
- `PYTHONUNBUFFERED` - Set to 1 for real-time logging
- `PYTHONDONTWRITEBYTECODE` - Set to 1 to prevent .pyc files

## Docker Commands

```bash
# Build image
docker build -t heart-disease-api .

# Run container
docker run -d \
  -p 8000:8000 \
  --name heart-disease-api \
  -e MODEL_NAME=xgboost \
  heart-disease-api

# View logs
docker logs -f heart-disease-api

# Stop container
docker stop heart-disease-api

# Remove container
docker rm heart-disease-api
```

## Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build -d
```

## Production Deployment

### Security Recommendations

1. **Use HTTPS** - Deploy behind a reverse proxy (nginx, traefik) with SSL/TLS
2. **API Authentication** - Add API key or JWT authentication
3. **Rate Limiting** - Implement rate limiting to prevent abuse
4. **CORS Configuration** - Restrict allowed origins in production
5. **Input Validation** - Already implemented with Pydantic
6. **Logging** - Configure proper logging and monitoring
7. **Resource Limits** - Set memory and CPU limits in Docker

### Example nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Troubleshooting

### Model Not Loading

- Ensure models directory exists and contains required files
- Check MODEL_NAME environment variable
- Verify preprocessing_pipeline.pkl exists

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

### Permission Errors in Docker

- Check file permissions
- Ensure appuser has read access to models directory

## License

MIT License

## Author

Saif Afzal
BITS Pilani - MLOps Course (S1-25_AIMLCZG523)
December 2025
