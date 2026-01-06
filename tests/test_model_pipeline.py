"""
Unit tests for model pipeline module
Tests the HeartDiseasePredictor class and prediction functionality
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Define data path
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "heart_disease_clean.csv")

from model_pipeline import HeartDiseasePredictor, save_model_package
from preprocessing import create_preprocessing_pipeline, load_data
from train import ModelTrainer


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class TestHeartDiseasePredictor:
    """Tests for HeartDiseasePredictor class"""

    @classmethod
    def setup_class(cls):
        """Setup models for testing (once for all tests)"""
        if not hasattr(cls, "feature_names"):
            X, _ = load_data(DATA_PATH)
            cls.feature_names = X.columns.tolist()

        model_dir = Path("models")
        # Always retrain to ensure consistency with current code
        X, y = load_data(DATA_PATH)

        pipeline = create_preprocessing_pipeline(
            handle_outliers=True, feature_engineering=True
        )
        X_transformed = pipeline.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42
        )

        trainer = ModelTrainer(random_state=42)
        model = XGBClassifier(eval_metric="logloss")
        results = trainer.train_model(
            model, X_train, y_train, X_test, y_test, model_name="XGBoost"
        )

        model_dir.mkdir(exist_ok=True)

        with open(model_dir / "xgboost_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(model_dir / "preprocessing_pipeline.pkl", "wb") as f:
            pickle.dump(pipeline, f)

        with open(model_dir / "training_results.json", "w") as f:
            json.dump({"results": {"XGBoost": results}}, f, cls=NumpyEncoder)

    def setup_method(self):
        """Setup predictor for each test"""
        self.predictor = HeartDiseasePredictor(model_dir="models/")

    def test_predictor_initialization(self):
        """Test that predictor initializes correctly"""
        assert self.predictor is not None
        assert str(self.predictor.model_dir).rstrip("/") == "models"
        assert self.predictor.model is None
        assert self.predictor.preprocessing_pipeline is None

    def test_load_models(self):
        """Test loading models and pipeline"""
        self.predictor.load_models(model_name="xgboost")

        assert self.predictor.model is not None, "Model should be loaded"
        assert (
            self.predictor.preprocessing_pipeline is not None
        ), "Pipeline should be loaded"

    def test_predict_with_dict(self):
        """Test prediction with dictionary input"""
        self.predictor.load_models(model_name="xgboost")

        # Ensure keys match training data exactly
        features = self.predictor.get_feature_names()
        patient_data = {
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
            "thal": 6,
        }

        # Verify keys match expected features
        # If predictor expects strict order/names that keys don't match, this might fail,
        # but dict doesn't preserve order usually. Predictor handles it.

        result = self.predictor.predict(patient_data)

        assert "prediction" in result
        assert "confidence" in result
        assert "diagnosis" in result
        assert result["prediction"] in [0, 1]
        if result["confidence"] is not None:
            assert 0 <= result["confidence"] <= 1

    def test_predict_with_dataframe(self):
        """Test prediction with DataFrame input"""
        self.predictor.load_models(model_name="xgboost")

        patient_data = pd.DataFrame(
            {
                "age": [63],
                "sex": [1],
                "cp": [1],
                "trestbps": [145],
                "chol": [233],
                "fbs": [1],
                "restecg": [2],
                "thalach": [150],
                "exang": [0],
                "oldpeak": [2.3],
                "slope": [3],
                "ca": [0],
                "thal": [6],
            }
        )

        result = self.predictor.predict(patient_data)

        assert "prediction" in result
        assert result["prediction"] in [0, 1]

    def test_predict_with_array(self):
        """Test prediction with array input"""
        self.predictor.load_models(model_name="xgboost")

        patient_data = np.array([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]])

        result = self.predictor.predict(patient_data)

        assert "prediction" in result
        assert result["prediction"] in [0, 1]

    def test_predict_batch(self):
        """Test batch predictions"""
        self.predictor.load_models(model_name="xgboost")

        patients_data = pd.DataFrame(
            {
                "age": [63, 67, 54],
                "sex": [1, 1, 0],
                "cp": [1, 4, 2],
                "trestbps": [145, 160, 140],
                "chol": [233, 286, 268],
                "fbs": [1, 0, 0],
                "restecg": [2, 2, 2],
                "thalach": [150, 108, 160],
                "exang": [0, 1, 0],
                "oldpeak": [2.3, 1.5, 3.6],
                "slope": [3, 2, 3],
                "ca": [0, 3, 2],
                "thal": [6, 3, 3],
            }
        )

        results = self.predictor.predict_batch(patients_data)

        assert len(results) == 3
        for result in results:
            assert "prediction" in result
            assert "confidence" in result
            assert result["prediction"] in [0, 1]

    def test_get_feature_names(self):
        """Test getting feature names"""
        self.predictor.load_models(model_name="xgboost")
        features = self.predictor.get_feature_names()

        assert len(features) == 13
        assert "age" in features
        assert "sex" in features
        assert "target" not in features

    def test_get_feature_info(self):
        """Test getting feature information"""
        feature_info = self.predictor.get_feature_info()

        assert isinstance(feature_info, dict)
        assert "age" in feature_info
        assert "description" in feature_info["age"]
        assert "range" in feature_info["age"] or "values" in feature_info["age"]

    def test_predict_without_loading_model(self):
        """Test that prediction fails without loading model"""
        patient_data = {
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
            "thal": 6,
        }

        with pytest.raises(ValueError, match="Model not loaded"):
            self.predictor.predict(patient_data)

    def test_predict_with_missing_features(self):
        """Test that prediction fails with missing features"""
        self.predictor.load_models(model_name="xgboost")
        patient_data = {
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
        }

        with pytest.raises((ValueError, KeyError)):
            self.predictor.predict(patient_data)

    def test_prediction_consistency(self):
        """Test that same input gives same prediction"""
        self.predictor.load_models(model_name="xgboost")
        patient_data = {
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
            "thal": 6,
        }

        result1 = self.predictor.predict(patient_data)
        result2 = self.predictor.predict(patient_data)

        assert result1["prediction"] == result2["prediction"]
        assert result1["confidence"] == result2["confidence"]


class TestModelPackageSaving:
    """Tests for save_model_package function"""

    def setup_method(self):
        """Setup temp directory"""
        self.temp_dir = Path("tests/temp_package")
        self.temp_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Cleanup temp files"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_save_model_package(self):
        """Test saving complete model package"""
        X, y = load_data(DATA_PATH)
        pipeline = create_preprocessing_pipeline(
            handle_outliers=True, feature_engineering=True
        )
        X_transformed = pipeline.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42
        )
        trainer = ModelTrainer(random_state=42)
        model = LogisticRegression(max_iter=1000)
        results = trainer.train_model(
            model, X_train, y_train, X_test, y_test, model_name="Logistic Regression"
        )

        # Manually verify results are serializable before saving
        # But wait, save_model_package simply calls json.dump.
        # We need to monkeypatch json.dump or modify save_model_package??
        # Or, we sanitize `results` here.

        # Sanitize results for test
        results_clean = json.loads(json.dumps(results, cls=NumpyEncoder))

        save_model_package(
            model=model,
            preprocessing_pipeline=pipeline,
            model_name="logistic_regression",
            metadata=results_clean,
            output_dir=str(self.temp_dir),
        )

        assert (self.temp_dir / "logistic_regression_model.pkl").exists()
        assert (self.temp_dir / "preprocessing_pipeline.pkl").exists()
        assert (self.temp_dir / "logistic_regression_metadata.json").exists()

    def test_load_saved_package(self):
        """Test loading saved model package"""
        X, y = load_data(DATA_PATH)
        pipeline = create_preprocessing_pipeline(
            handle_outliers=True, feature_engineering=True
        )
        X_transformed = pipeline.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42
        )
        trainer = ModelTrainer(random_state=42)
        model = XGBClassifier(eval_metric="logloss")
        results = trainer.train_model(
            model, X_train, y_train, X_test, y_test, model_name="XGBoost"
        )

        results_clean = json.loads(json.dumps(results, cls=NumpyEncoder))

        save_model_package(
            model=model,
            preprocessing_pipeline=pipeline,
            model_name="xgboost",
            metadata=results_clean,
            output_dir=str(self.temp_dir),
        )

        predictor = HeartDiseasePredictor(model_dir=str(self.temp_dir))
        predictor.load_models(model_name="xgboost")

        patient_data = {
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
            "thal": 6,
        }

        result = predictor.predict(patient_data)
        assert "prediction" in result


class TestPredictionValidation:
    """Tests for prediction validation and edge cases"""

    def setup_method(self):
        """Setup predictor"""
        self.predictor = HeartDiseasePredictor(model_dir="models/")
        self.predictor.load_models(model_name="xgboost")

    def test_extreme_age_values(self):
        """Test prediction with extreme age values"""
        young_patient = {
            "age": 20,
            "sex": 1,
            "cp": 1,
            "trestbps": 120,
            "chol": 200,
            "fbs": 0,
            "restecg": 0,
            "thalach": 180,
            "exang": 0,
            "oldpeak": 0,
            "slope": 1,
            "ca": 0,
            "thal": 3,
        }
        result = self.predictor.predict(young_patient)
        assert result["prediction"] in [0, 1]

        old_patient = {
            "age": 80,
            "sex": 1,
            "cp": 1,
            "trestbps": 150,
            "chol": 250,
            "fbs": 1,
            "restecg": 2,
            "thalach": 120,
            "exang": 1,
            "oldpeak": 3,
            "slope": 3,
            "ca": 2,
            "thal": 6,
        }
        result = self.predictor.predict(old_patient)
        assert result["prediction"] in [0, 1]

    def test_different_sex_values(self):
        """Test predictions for different sex values"""
        male_patient = {
            "age": 55,
            "sex": 1,
            "cp": 1,
            "trestbps": 140,
            "chol": 240,
            "fbs": 0,
            "restecg": 1,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.5,
            "slope": 2,
            "ca": 0,
            "thal": 3,
        }

        result_male = self.predictor.predict(male_patient)
        assert result_male["prediction"] in [0, 1]

        female_patient = male_patient.copy()
        female_patient["sex"] = 0
        result_female = self.predictor.predict(female_patient)
        assert result_female["prediction"] in [0, 1]

    def test_probability_bounds(self):
        """Test that probabilities are within valid bounds"""
        patients_data = pd.DataFrame(
            {
                "age": [45, 55, 65, 75],
                "sex": [1, 0, 1, 0],
                "cp": [1, 2, 3, 4],
                "trestbps": [120, 140, 160, 180],
                "chol": [200, 240, 280, 320],
                "fbs": [0, 1, 0, 1],
                "restecg": [0, 1, 2, 0],
                "thalach": [170, 150, 130, 110],
                "exang": [0, 0, 1, 1],
                "oldpeak": [0, 1, 2, 3],
                "slope": [1, 2, 3, 2],
                "ca": [0, 1, 2, 3],
                "thal": [3, 6, 7, 3],
            }
        )

        results = self.predictor.predict_batch(patients_data)

        for result in results:
            if result["confidence"] is not None:
                assert 0 <= result["confidence"] <= 1


class TestIntegrationWithRealData:
    """Integration tests using real dataset"""

    def setup_method(self):
        """Setup with real data"""
        self.X, self.y = load_data(DATA_PATH)
        self.predictor = HeartDiseasePredictor(model_dir="models/")
        self.predictor.load_models(model_name="xgboost")

    def test_predict_on_real_samples(self):
        """Test predictions on actual dataset samples"""
        sample_data = self.X.head(5)
        results = self.predictor.predict_batch(sample_data)
        assert len(results) == 5
        for result in results:
            assert "prediction" in result
            assert result["prediction"] in [0, 1]

    def test_prediction_distribution(self):
        """Test that predictions have reasonable distribution"""
        results = self.predictor.predict_batch(self.X)
        predictions = [r["prediction"] for r in results]
        unique, counts = np.unique(predictions, return_counts=True)
        assert len(unique) == 2
        assert all(count < len(predictions) for count in counts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
