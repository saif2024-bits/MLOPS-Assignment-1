"""
Complete Model Pipeline - Heart Disease Prediction
End-to-end pipeline for reproducible predictions
"""

import pickle
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple
from datetime import datetime


class HeartDiseasePredictor:
    """
    End-to-end prediction pipeline for heart disease classification

    This class encapsulates the entire ML pipeline including:
    - Data preprocessing
    - Feature engineering
    - Model prediction
    - Results formatting

    Example:
    --------
    >>> predictor = HeartDiseasePredictor()
    >>> predictor.load_models('models/')
    >>> result = predictor.predict(patient_data)
    >>> print(result)
    """

    def __init__(self, model_dir: str = 'models/'):
        """
        Initialize the predictor

        Parameters:
        -----------
        model_dir : str
            Directory containing saved models and pipeline
        """
        self.model_dir = Path(model_dir)
        self.preprocessing_pipeline = None
        self.model = None
        self.model_name = None
        self.model_metadata = {}

    def load_models(self, model_name: str = 'xgboost'):
        """
        Load preprocessing pipeline and trained model

        Parameters:
        -----------
        model_name : str
            Name of model to load ('logistic_regression', 'random_forest', 'xgboost')

        Returns:
        --------
        self
        """
        # Load preprocessing pipeline
        pipeline_path = self.model_dir / 'preprocessing_pipeline.pkl'
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Preprocessing pipeline not found at {pipeline_path}")

        with open(pipeline_path, 'rb') as f:
            self.preprocessing_pipeline = pickle.load(f)
        print(f"✅ Loaded preprocessing pipeline from {pipeline_path}")

        # Load model
        model_path = self.model_dir / f'{model_name}_model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✅ Loaded {model_name} model from {model_path}")

        self.model_name = model_name

        # Load model metadata if available
        results_path = self.model_dir / 'training_results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
                # Find model name in results
                for key in results['results'].keys():
                    if model_name.replace('_', ' ').title() in key:
                        self.model_metadata = results['results'][key]
                        break

        return self

    def preprocess(self, data: Union[pd.DataFrame, Dict, np.ndarray]) -> np.ndarray:
        """
        Preprocess input data using trained pipeline

        Parameters:
        -----------
        data : DataFrame, dict, or array
            Input patient data

        Returns:
        --------
        np.ndarray
            Preprocessed features ready for prediction
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline not loaded. Call load_models() first.")

        # Convert to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, np.ndarray):
            # Assume correct feature order
            feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                           'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            data = pd.DataFrame(data.reshape(1, -1) if data.ndim == 1 else data,
                              columns=feature_names)

        # Apply preprocessing pipeline
        X_processed = self.preprocessing_pipeline.transform(data)

        return X_processed

    def predict(self, data: Union[pd.DataFrame, Dict, np.ndarray],
                return_proba: bool = True) -> Dict:
        """
        Make prediction on input data

        Parameters:
        -----------
        data : DataFrame, dict, or array
            Patient data for prediction
        return_proba : bool
            Whether to return probability scores

        Returns:
        --------
        dict
            Prediction results with metadata
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_models() first.")

        # Preprocess data
        X_processed = self.preprocess(data)

        # Make prediction
        prediction = self.model.predict(X_processed)

        # Get probability if available
        if return_proba and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_processed)
            prob_no_disease = probabilities[0][0]
            prob_disease = probabilities[0][1]
        else:
            prob_no_disease = None
            prob_disease = None

        # Format results
        result = {
            'prediction': int(prediction[0]),
            'diagnosis': 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease',
            'confidence': float(prob_disease) if prob_disease is not None else None,
            'probabilities': {
                'no_disease': float(prob_no_disease) if prob_no_disease is not None else None,
                'disease': float(prob_disease) if prob_disease is not None else None
            },
            'model_used': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'model_performance': {
                'test_accuracy': self.model_metadata.get('test_accuracy'),
                'test_roc_auc': self.model_metadata.get('test_roc_auc')
            } if self.model_metadata else None
        }

        return result

    def predict_batch(self, data: pd.DataFrame) -> List[Dict]:
        """
        Make predictions on batch of data

        Parameters:
        -----------
        data : DataFrame
            Multiple patient records

        Returns:
        --------
        list of dict
            Prediction results for each record
        """
        results = []

        for idx in range(len(data)):
            row = data.iloc[idx:idx+1]
            result = self.predict(row)
            result['record_id'] = idx
            results.append(result)

        return results

    def get_feature_names(self) -> List[str]:
        """
        Get expected feature names for input data

        Returns:
        --------
        list
            Feature names in expected order
        """
        return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    def get_feature_info(self) -> Dict:
        """
        Get information about expected features

        Returns:
        --------
        dict
            Feature descriptions and valid ranges
        """
        return {
            'age': {'description': 'Age in years', 'range': '29-77'},
            'sex': {'description': 'Sex', 'values': '1=male, 0=female'},
            'cp': {'description': 'Chest pain type', 'values': '1-4'},
            'trestbps': {'description': 'Resting blood pressure (mm Hg)', 'range': '94-200'},
            'chol': {'description': 'Serum cholesterol (mg/dl)', 'range': '126-564'},
            'fbs': {'description': 'Fasting blood sugar > 120 mg/dl', 'values': '1=true, 0=false'},
            'restecg': {'description': 'Resting ECG results', 'values': '0-2'},
            'thalach': {'description': 'Maximum heart rate achieved', 'range': '71-202'},
            'exang': {'description': 'Exercise induced angina', 'values': '1=yes, 0=no'},
            'oldpeak': {'description': 'ST depression induced by exercise', 'range': '0-6.2'},
            'slope': {'description': 'Slope of peak exercise ST segment', 'values': '1-3'},
            'ca': {'description': 'Number of major vessels (0-3)', 'values': '0-3'},
            'thal': {'description': 'Thalassemia', 'values': '3=normal, 6=fixed defect, 7=reversible defect'}
        }

    def save_prediction_log(self, prediction: Dict, log_file: str = 'predictions.log'):
        """
        Save prediction to log file

        Parameters:
        -----------
        prediction : dict
            Prediction result
        log_file : str
            Path to log file
        """
        log_path = self.model_dir / log_file

        with open(log_path, 'a') as f:
            f.write(json.dumps(prediction) + '\n')

    @staticmethod
    def create_sample_input() -> pd.DataFrame:
        """
        Create sample input data for testing

        Returns:
        --------
        DataFrame
            Sample patient data
        """
        return pd.DataFrame([{
            'age': 63,
            'sex': 1,
            'cp': 1,
            'trestbps': 145,
            'chol': 233,
            'fbs': 1,
            'restecg': 2,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 2.3,
            'slope': 3,
            'ca': 0,
            'thal': 6
        }])


def save_model_package(model, preprocessing_pipeline, model_name: str,
                       metadata: Dict, output_dir: str = 'models/'):
    """
    Save complete model package with all components

    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    preprocessing_pipeline : Pipeline
        Fitted preprocessing pipeline
    model_name : str
        Name of the model
    metadata : dict
        Model performance metrics and info
    output_dir : str
        Directory to save package
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save model
    model_file = output_path / f'{model_name}_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved model to {model_file}")

    # Save preprocessing pipeline
    pipeline_file = output_path / 'preprocessing_pipeline.pkl'
    with open(pipeline_file, 'wb') as f:
        pickle.dump(preprocessing_pipeline, f)
    print(f"✅ Saved pipeline to {pipeline_file}")

    # Save metadata
    metadata_file = output_path / f'{model_name}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Saved metadata to {metadata_file}")

    # Create model card
    model_card = f"""# Model Card: {model_name.title()}

## Model Details
- **Model Type**: {metadata.get('model_type', 'Unknown')}
- **Task**: Binary Classification (Heart Disease Detection)
- **Framework**: scikit-learn / XGBoost
- **Created**: {datetime.now().strftime('%Y-%m-%d')}

## Performance
- **Test Accuracy**: {metadata.get('test_accuracy', 'N/A')}
- **Test ROC-AUC**: {metadata.get('test_roc_auc', 'N/A')}
- **Test Precision**: {metadata.get('test_precision', 'N/A')}
- **Test Recall**: {metadata.get('test_recall', 'N/A')}
- **Test F1**: {metadata.get('test_f1', 'N/A')}

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
predictor.load_models(model_name='{model_name}')

# Make prediction
patient_data = {{...}}  # Patient features
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
"""

    card_file = output_path / f'{model_name}_model_card.md'
    with open(card_file, 'w') as f:
        f.write(model_card)
    print(f"✅ Saved model card to {card_file}")


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("="*70)
    print("MODEL PIPELINE - DEMONSTRATION")
    print("="*70)

    # Initialize predictor
    predictor = HeartDiseasePredictor(model_dir='models/')

    # Load best model (XGBoost)
    print("\n1. Loading models...")
    predictor.load_models(model_name='xgboost')

    # Create sample input
    print("\n2. Creating sample input...")
    sample_data = predictor.create_sample_input()
    print(sample_data.to_string())

    # Get feature info
    print("\n3. Feature information:")
    feature_info = predictor.get_feature_info()
    for feature, info in list(feature_info.items())[:3]:
        print(f"   {feature}: {info['description']}")
    print("   ...")

    # Make prediction
    print("\n4. Making prediction...")
    result = predictor.predict(sample_data)

    print(f"\n   Prediction: {result['diagnosis']}")
    print(f"   Confidence: {result['confidence']:.2%}" if result['confidence'] else "")
    print(f"   Model: {result['model_used']}")

    # Test with dict input
    print("\n5. Testing with dict input...")
    patient_dict = {
        'age': 54, 'sex': 1, 'cp': 4, 'trestbps': 140, 'chol': 239,
        'fbs': 0, 'restecg': 0, 'thalach': 160, 'exang': 0,
        'oldpeak': 1.2, 'slope': 1, 'ca': 0, 'thal': 3
    }
    result2 = predictor.predict(patient_dict)
    print(f"   Prediction: {result2['diagnosis']}")
    print(f"   Confidence: {result2['confidence']:.2%}" if result2['confidence'] else "")

    print("\n" + "="*70)
    print("MODEL PIPELINE READY FOR PRODUCTION!")
    print("="*70)
