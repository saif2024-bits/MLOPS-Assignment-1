# Model Card: Logistic_Regression

## Model Details
- **Model Type**: Logistic Regression
- **Task**: Binary Classification (Heart Disease Detection)
- **Framework**: scikit-learn / XGBoost
- **Created**: 2025-12-22

## Performance
- **Test Accuracy**: 0.8852459016393442
- **Test ROC-AUC**: 0.9534632034632035
- **Test Precision**: 0.8387096774193549
- **Test Recall**: 0.9285714285714286
- **Test F1**: 0.8813559322033898

## Training Data
- **Dataset**: UCI Heart Disease Dataset
- **Samples**: 303
- **Features**: 13 clinical features
- **Train/Test Split**: 80/20 stratified

## Usage
```python
from model_pipeline import HeartDiseasePredictor

# Initialize predictor
predictor = HeartDiseasePredictor(model_dir='models/')

# Load model
predictor.load_models(model_name='logistic_regression')

# Make prediction
patient_data = {...}  # Patient features
result = predictor.predict(patient_data)
print(result)
```

## Limitations
- Trained on UCI dataset (limited diversity)
- Requires specific feature format
- Not FDA approved for clinical use
- For research/educational purposes only

## Ethical Considerations
- Model should not be used as sole diagnostic tool
- Always consult healthcare professionals
- Potential bias in training data
- Privacy concerns with patient data
