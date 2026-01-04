"""
Unit tests for model training module
Tests model training, evaluation, and persistence
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import ModelTrainer
from preprocessing import load_data


class TestModelTrainer:
    """Tests for ModelTrainer class"""

    def setup_method(self):
        """Setup test data and trainer"""
        # Load actual data
        self.X, self.y = load_data('data/heart_disease_clean.csv')

        # Split data once for all tests
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Create trainer
        self.trainer = ModelTrainer(random_state=42)
        self.trainer.initialize_models()

    def test_trainer_initialization(self):
        """Test that trainer initializes correctly"""
        assert self.trainer is not None
        assert self.trainer.random_state == 42
        assert self.trainer.results == {}
        assert len(self.trainer.models) == 3

    def test_train_logistic_regression(self):
        """Test training logistic regression model"""
        model = self.trainer.models['Logistic Regression']
        results = self.trainer.train_model(
            model, self.X_train, self.y_train, self.X_test, self.y_test,
            model_name='Logistic Regression'
        )

        assert 'train_accuracy' in results, "Should have train accuracy"
        assert 'test_accuracy' in results, "Should have test accuracy"
        assert 'test_roc_auc' in results, "Should have test ROC-AUC"
        assert results['test_accuracy'] > 0.5, "Accuracy should be reasonable"

    def test_train_random_forest(self):
        """Test training random forest model"""
        model = self.trainer.models['Random Forest']
        results = self.trainer.train_model(
            model, self.X_train, self.y_train, self.X_test, self.y_test,
            model_name='Random Forest'
        )

        assert 'train_accuracy' in results, "Should have train accuracy"
        assert 'test_accuracy' in results, "Should have test accuracy"
        assert results['test_accuracy'] > 0.5, "Accuracy should be reasonable"

    def test_train_xgboost(self):
        """Test training XGBoost model"""
        model = self.trainer.models['XGBoost']
        results = self.trainer.train_model(
            model, self.X_train, self.y_train, self.X_test, self.y_test,
            model_name='XGBoost'
        )

        assert 'train_accuracy' in results, "Should have train accuracy"
        assert 'test_accuracy' in results, "Should have test accuracy"
        assert results['test_accuracy'] > 0.5, "Accuracy should be reasonable"

    def test_cross_validation(self):
        """Test cross-validation functionality"""
        # Cross-validate all models
        cv_results = self.trainer.cross_validate_models(self.X, self.y, cv=3)

        assert len(cv_results) == 3, "Should have CV results for all 3 models"

        for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
            assert model_name in cv_results, f"Should have CV results for {model_name}"
            assert 'accuracy_mean' in cv_results[model_name], "Should have accuracy mean"
            assert 'accuracy_std' in cv_results[model_name], "Should have accuracy std"
            assert 'roc_auc_mean' in cv_results[model_name], "Should have ROC-AUC mean"
            assert cv_results[model_name]['accuracy_mean'] > 0.5, "CV accuracy should be reasonable"

    def test_model_persistence(self):
        """Test saving and loading models"""
        # Train a model
        model = self.trainer.models['Logistic Regression']
        self.trainer.train_model(
            model, self.X_train, self.y_train, self.X_test, self.y_test,
            model_name='Logistic Regression'
        )

        # Create temporary directory for testing
        temp_dir = Path('tests/temp_models')
        temp_dir.mkdir(exist_ok=True)

        # Save model
        model_path = temp_dir / 'test_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Load model
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        # Test predictions match
        original_pred = model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)

        np.testing.assert_array_equal(original_pred, loaded_pred)

        # Cleanup
        model_path.unlink()
        temp_dir.rmdir()

    def test_metrics_calculation(self):
        """Test that all metrics are calculated correctly"""
        model = self.trainer.models['Logistic Regression']
        results = self.trainer.train_model(
            model, self.X_train, self.y_train, self.X_test, self.y_test,
            model_name='Logistic Regression'
        )

        required_metrics = [
            'train_accuracy', 'test_accuracy',
            'test_precision', 'test_recall',
            'test_f1', 'test_roc_auc'
        ]

        for metric in required_metrics:
            assert metric in results, f"Should have {metric}"
            assert 0 <= results[metric] <= 1, f"{metric} should be between 0 and 1"

    def test_train_all_models(self):
        """Test training all models"""
        self.trainer.train_all_models(self.X_train, self.y_train, self.X_test, self.y_test)

        assert len(self.trainer.results) == 3, "Should have 3 results"

        expected_models = ['Logistic Regression', 'Random Forest', 'XGBoost']
        for model_name in expected_models:
            assert model_name in self.trainer.results, f"Should have results for {model_name}"

    def test_best_model_selection(self):
        """Test that best model is selected correctly"""
        self.trainer.train_all_models(self.X_train, self.y_train, self.X_test, self.y_test)

        # Get best model by ROC-AUC
        best_name = max(
            self.trainer.results.items(),
            key=lambda x: x[1]['test_roc_auc']
        )[0]

        assert best_name in ['Logistic Regression', 'Random Forest', 'XGBoost']
        assert self.trainer.results[best_name]['test_roc_auc'] > 0.7

    def test_reproducibility(self):
        """Test that training is reproducible with same random state"""
        trainer1 = ModelTrainer(random_state=42)
        trainer1.initialize_models()
        trainer2 = ModelTrainer(random_state=42)
        trainer2.initialize_models()

        model1 = trainer1.models['Logistic Regression']
        model2 = trainer2.models['Logistic Regression']

        results1 = trainer1.train_model(
            model1, self.X_train, self.y_train, self.X_test, self.y_test,
            model_name='Logistic Regression'
        )
        results2 = trainer2.train_model(
            model2, self.X_train, self.y_train, self.X_test, self.y_test,
            model_name='Logistic Regression'
        )

        # Results should be identical
        assert results1['test_accuracy'] == results2['test_accuracy']
        assert results1['test_roc_auc'] == results2['test_roc_auc']

    def test_model_predictions_shape(self):
        """Test that model predictions have correct shape"""
        model = self.trainer.models['Logistic Regression']
        self.trainer.train_model(
            model, self.X_train, self.y_train, self.X_test, self.y_test,
            model_name='Logistic Regression'
        )

        predictions = model.predict(self.X_test)
        probabilities = model.predict_proba(self.X_test)

        assert predictions.shape[0] == self.X_test.shape[0], "Predictions shape should match test size"
        assert probabilities.shape == (self.X_test.shape[0], 2), "Probabilities shape should be (n_samples, 2)"

    def test_model_predictions_binary(self):
        """Test that predictions are binary (0 or 1)"""
        model = self.trainer.models['Logistic Regression']
        self.trainer.train_model(
            model, self.X_train, self.y_train, self.X_test, self.y_test,
            model_name='Logistic Regression'
        )

        predictions = model.predict(self.X_test)

        assert set(predictions).issubset({0, 1}), "Predictions should be binary (0 or 1)"


class TestModelSaving:
    """Tests for model saving and loading"""

    def setup_method(self):
        """Setup test data"""
        self.X, self.y = load_data('data/heart_disease_clean.csv')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.trainer = ModelTrainer(random_state=42)
        self.trainer.initialize_models()

        # Create temp directory
        self.temp_dir = Path('tests/temp_models')
        self.temp_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Cleanup temp files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_save_results_json(self):
        """Test saving results to JSON"""
        self.trainer.train_all_models(self.X_train, self.y_train, self.X_test, self.y_test)

        results_path = self.temp_dir / 'results.json'
        self.trainer.save_results(str(results_path))

        assert results_path.exists(), "Results file should be created"

        # Load and verify
        with open(results_path, 'r') as f:
            loaded_results = json.load(f)

        assert 'best_model' in loaded_results
        assert 'results' in loaded_results
        assert len(loaded_results['results']) == 3

    def test_save_models_pickle(self):
        """Test saving models as pickle files"""
        self.trainer.train_all_models(self.X_train, self.y_train, self.X_test, self.y_test)

        # Save each model (models are already trained and in self.trainer.models)
        for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
            model = self.trainer.models[model_name]
            model_path = self.temp_dir / f'{model_name.lower().replace(" ", "_")}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            assert model_path.exists(), f"Model file should be created for {model_name}"

    def test_loaded_model_predictions(self):
        """Test that loaded model makes same predictions"""
        model = self.trainer.models['Logistic Regression']
        self.trainer.train_model(
            model, self.X_train, self.y_train, self.X_test, self.y_test,
            model_name='Logistic Regression'
        )

        # Save and load
        model_path = self.temp_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        # Compare predictions
        original_pred = model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)

        np.testing.assert_array_equal(original_pred, loaded_pred)


class TestModelPerformance:
    """Tests for model performance requirements"""

    def setup_method(self):
        """Setup test data"""
        self.X, self.y = load_data('data/heart_disease_clean.csv')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.trainer = ModelTrainer(random_state=42)
        self.trainer.initialize_models()

    def test_minimum_accuracy_threshold(self):
        """Test that models meet minimum accuracy threshold"""
        self.trainer.train_all_models(self.X_train, self.y_train, self.X_test, self.y_test)

        for model_name, results in self.trainer.results.items():
            assert results['test_accuracy'] > 0.6, \
                f"{model_name} should have accuracy > 60%"

    def test_minimum_roc_auc_threshold(self):
        """Test that models meet minimum ROC-AUC threshold"""
        self.trainer.train_all_models(self.X_train, self.y_train, self.X_test, self.y_test)

        for model_name, results in self.trainer.results.items():
            assert results['test_roc_auc'] > 0.7, \
                f"{model_name} should have ROC-AUC > 70%"

    def test_no_overfitting(self):
        """Test that models are not severely overfitting"""
        self.trainer.train_all_models(self.X_train, self.y_train, self.X_test, self.y_test)

        for model_name, results in self.trainer.results.items():
            train_acc = results['train_accuracy']
            test_acc = results['test_accuracy']

            # Allow up to 20% gap between train and test accuracy (XGBoost can achieve 100% train)
            assert train_acc - test_acc < 0.20, \
                f"{model_name} may be overfitting (train: {train_acc}, test: {test_acc})"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
