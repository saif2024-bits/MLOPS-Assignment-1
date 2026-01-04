"""
Unit tests for preprocessing module
Tests data loading, feature engineering, and preprocessing pipeline
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import (
    load_data,
    create_preprocessing_pipeline,
    FeatureEngineering,
    OutlierHandler,
    get_feature_info
)


class TestDataLoading:
    """Tests for data loading functionality"""

    def test_load_data_returns_correct_shapes(self):
        """Test that load_data returns correct X and y shapes"""
        X, y = load_data('data/heart_disease_clean.csv')

        assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
        assert isinstance(y, pd.Series), "y should be a Series"
        assert X.shape[0] == 303, "Should have 303 samples"
        assert X.shape[1] == 13, "Should have 13 features"
        assert y.shape[0] == 303, "y should have 303 samples"

    def test_load_data_correct_columns(self):
        """Test that loaded data has correct columns"""
        X, y = load_data('data/heart_disease_clean.csv')

        expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                          'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

        assert list(X.columns) == expected_columns, "Columns should match expected features"

    def test_load_data_no_missing_values(self):
        """Test that loaded data has no missing values"""
        X, y = load_data('data/heart_disease_clean.csv')

        assert X.isnull().sum().sum() == 0, "X should have no missing values"
        assert y.isnull().sum() == 0, "y should have no missing values"

    def test_load_data_target_binary(self):
        """Test that target is binary (0 or 1)"""
        X, y = load_data('data/heart_disease_clean.csv')

        assert set(y.unique()).issubset({0, 1}), "Target should be binary (0, 1)"


class TestFeatureEngineering:
    """Tests for FeatureEngineering transformer"""

    def setup_method(self):
        """Setup test data"""
        self.X_sample = pd.DataFrame({
            'age': [63, 67, 54],
            'sex': [1, 1, 0],
            'cp': [1, 4, 2],
            'trestbps': [145, 160, 140],
            'chol': [233, 286, 268],
            'fbs': [1, 0, 0],
            'restecg': [2, 2, 2],
            'thalach': [150, 108, 160],
            'exang': [0, 1, 0],
            'oldpeak': [2.3, 1.5, 3.6],
            'slope': [3, 2, 3],
            'ca': [0, 3, 2],
            'thal': [6, 3, 3]
        })

    def test_feature_engineering_creates_new_features(self):
        """Test that feature engineering creates expected new features"""
        fe = FeatureEngineering(create_interactions=True)
        X_transformed = fe.fit_transform(self.X_sample)

        # Original features (13) + new features (7) = 20
        assert X_transformed.shape[1] == 20, "Should create 7 new features"

    def test_feature_engineering_preserves_original_features(self):
        """Test that original features are preserved"""
        fe = FeatureEngineering(create_interactions=True)
        X_transformed = fe.fit_transform(self.X_sample)

        # Check original columns exist
        for col in self.X_sample.columns:
            assert col in X_transformed.columns, f"Original feature {col} should be preserved"

    def test_feature_engineering_no_interactions(self):
        """Test feature engineering with interactions disabled"""
        fe = FeatureEngineering(create_interactions=False)
        X_transformed = fe.fit_transform(self.X_sample)

        # Should not create new features
        assert X_transformed.shape[1] == 13, "Should not create new features"

    def test_feature_engineering_new_feature_names(self):
        """Test that new features have correct names"""
        fe = FeatureEngineering(create_interactions=True)
        X_transformed = fe.fit_transform(self.X_sample)

        expected_new_features = ['age_risk', 'chol_risk', 'bp_risk', 'hr_reserve',
                                'exercise_capacity', 'cp_exang_interaction', 'age_chol_interaction']

        for feature in expected_new_features:
            assert feature in X_transformed.columns, f"New feature {feature} should exist"


class TestOutlierHandler:
    """Tests for OutlierHandler transformer"""

    def setup_method(self):
        """Setup test data with outliers"""
        self.X_with_outliers = pd.DataFrame({
            'age': [30, 40, 50, 60, 150],  # 150 is outlier
            'chol': [200, 220, 240, 260, 1000]  # 1000 is outlier
        })

    def test_outlier_handler_clips_values(self):
        """Test that outlier handler clips extreme values"""
        oh = OutlierHandler(method='iqr', factor=1.5)
        X_transformed = oh.fit_transform(self.X_with_outliers)

        # Transformed values should be within reasonable range
        assert X_transformed['age'].max() < 150, "Outlier should be clipped"
        assert X_transformed['chol'].max() < 1000, "Outlier should be clipped"

    def test_outlier_handler_preserves_normal_values(self):
        """Test that normal values are preserved"""
        oh = OutlierHandler(method='iqr', factor=1.5)
        X_transformed = oh.fit_transform(self.X_with_outliers)

        # Normal values should be unchanged
        assert X_transformed['age'].iloc[0] == 30, "Normal values should be preserved"
        assert X_transformed['age'].iloc[1] == 40, "Normal values should be preserved"

    def test_outlier_handler_fit_transform_consistency(self):
        """Test that fit and transform produce consistent results"""
        oh = OutlierHandler(method='iqr', factor=1.5)

        # Fit on data
        oh.fit(self.X_with_outliers)
        X_transformed1 = oh.transform(self.X_with_outliers)

        # Fit-transform
        X_transformed2 = oh.fit_transform(self.X_with_outliers)

        pd.testing.assert_frame_equal(X_transformed1, X_transformed2)


class TestPreprocessingPipeline:
    """Tests for complete preprocessing pipeline"""

    def setup_method(self):
        """Setup test data"""
        self.X, self.y = load_data('data/heart_disease_clean.csv')

    def test_pipeline_creation(self):
        """Test that pipeline is created successfully"""
        pipeline = create_preprocessing_pipeline(
            handle_outliers=True,
            feature_engineering=True
        )

        assert pipeline is not None, "Pipeline should be created"
        assert len(pipeline.steps) == 3, "Pipeline should have 3 steps"

    def test_pipeline_fit_transform(self):
        """Test that pipeline can fit and transform data"""
        pipeline = create_preprocessing_pipeline(
            handle_outliers=True,
            feature_engineering=True
        )

        X_transformed = pipeline.fit_transform(self.X)

        assert X_transformed.shape[0] == 303, "Should preserve number of samples"
        assert X_transformed.shape[1] == 20, "Should have 20 features after transformation"

    def test_pipeline_transform_scales_data(self):
        """Test that pipeline scales data (mean≈0, std≈1)"""
        pipeline = create_preprocessing_pipeline(
            handle_outliers=True,
            feature_engineering=True
        )

        X_transformed = pipeline.fit_transform(self.X)

        # Check that data is scaled
        assert abs(X_transformed.mean()) < 0.5, "Mean should be close to 0"
        assert abs(X_transformed.std() - 1) < 0.5, "Std should be close to 1"

    def test_pipeline_without_feature_engineering(self):
        """Test pipeline with feature engineering disabled"""
        pipeline = create_preprocessing_pipeline(
            handle_outliers=True,
            feature_engineering=False
        )

        X_transformed = pipeline.fit_transform(self.X)

        # Should have original 13 features only
        assert X_transformed.shape[1] == 13, "Should not create new features"

    def test_pipeline_reproducibility(self):
        """Test that pipeline produces consistent results"""
        pipeline1 = create_preprocessing_pipeline(
            handle_outliers=True,
            feature_engineering=True
        )
        pipeline2 = create_preprocessing_pipeline(
            handle_outliers=True,
            feature_engineering=True
        )

        X_transformed1 = pipeline1.fit_transform(self.X)
        X_transformed2 = pipeline2.fit_transform(self.X)

        # Results should be identical
        np.testing.assert_array_almost_equal(X_transformed1, X_transformed2)


class TestGetFeatureInfo:
    """Tests for get_feature_info function"""

    def test_feature_info_returns_dict(self):
        """Test that feature info returns a dictionary"""
        info = get_feature_info()

        assert isinstance(info, dict), "Should return dictionary"

    def test_feature_info_has_required_keys(self):
        """Test that feature info has required keys"""
        info = get_feature_info()

        required_keys = ['numerical_features', 'categorical_features', 'feature_descriptions']
        for key in required_keys:
            assert key in info, f"Should have {key} key"

    def test_feature_info_correct_feature_count(self):
        """Test that feature info has correct number of features"""
        info = get_feature_info()

        num_features = len(info['numerical_features'])
        cat_features = len(info['categorical_features'])

        assert num_features == 5, "Should have 5 numerical features"
        assert cat_features == 8, "Should have 8 categorical features"
        assert num_features + cat_features == 13, "Total should be 13 features"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
