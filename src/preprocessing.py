"""
Preprocessing module for Heart Disease Prediction
Implements sklearn pipelines for data transformation
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering
    Creates interaction features and domain-specific transformations
    """

    def __init__(self, create_interactions=True):
        """
        Initialize feature engineering transformer

        Parameters:
        -----------
        create_interactions : bool, default=True
            Whether to create interaction features
        """
        self.create_interactions = create_interactions

    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)"""
        return self

    def transform(self, X):
        """
        Transform the input data by creating new features

        Parameters:
        -----------
        X : pd.DataFrame
            Input features

        Returns:
        --------
        pd.DataFrame
            Transformed features with additional engineered features
        """
        X_copy = X.copy()

        if self.create_interactions:
            # Age-based risk categories
            X_copy["age_risk"] = pd.cut(
                X_copy["age"], bins=[0, 45, 55, 65, 100], labels=[0, 1, 2, 3]
            ).astype(float)

            # Cholesterol risk levels
            X_copy["chol_risk"] = pd.cut(
                X_copy["chol"], bins=[0, 200, 240, 500], labels=[0, 1, 2]
            ).astype(float)

            # Blood pressure risk
            X_copy["bp_risk"] = pd.cut(
                X_copy["trestbps"], bins=[0, 120, 140, 200], labels=[0, 1, 2]
            ).astype(float)

            # Heart rate reserve (indicator of fitness)
            max_hr = 220 - X_copy["age"]
            X_copy["hr_reserve"] = max_hr - X_copy["thalach"]

            # Exercise capacity (interaction)
            X_copy["exercise_capacity"] = X_copy["thalach"] / (X_copy["age"] + 1)

            # Critical interaction: chest pain type with exercise angina
            X_copy["cp_exang_interaction"] = X_copy["cp"] * X_copy["exang"]

            # Age and cholesterol interaction
            X_copy["age_chol_interaction"] = (X_copy["age"] / 100) * (
                X_copy["chol"] / 100
            )

        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        if input_features is None:
            return None

        output_features = list(input_features)
        if self.create_interactions:
            output_features.extend(
                [
                    "age_risk",
                    "chol_risk",
                    "bp_risk",
                    "hr_reserve",
                    "exercise_capacity",
                    "cp_exang_interaction",
                    "age_chol_interaction",
                ]
            )
        return np.array(output_features)


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handles outliers using IQR-based clipping
    """

    def __init__(self, method="iqr", factor=1.5):
        """
        Initialize outlier handler

        Parameters:
        -----------
        method : str, default='iqr'
            Method for outlier detection ('iqr' or 'zscore')
        factor : float, default=1.5
            Multiplier for IQR or z-score threshold
        """
        self.method = method
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        """
        Fit the transformer by computing outlier bounds

        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : ignored

        Returns:
        --------
        self
        """
        numerical_cols = X.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if self.method == "iqr":
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR
            else:  # zscore
                mean = X[col].mean()
                std = X[col].std()
                lower_bound = mean - self.factor * std
                upper_bound = mean + self.factor * std

            self.bounds_[col] = (lower_bound, upper_bound)

        return self

    def transform(self, X):
        """
        Transform by clipping outliers to computed bounds

        Parameters:
        -----------
        X : pd.DataFrame
            Input features

        Returns:
        --------
        pd.DataFrame
            Transformed features with clipped outliers
        """
        X_copy = X.copy()

        for col, (lower, upper) in self.bounds_.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].clip(lower, upper)

        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        return input_features


def load_data(filepath):
    """
    Load heart disease dataset

    Parameters:
    -----------
    filepath : str
        Path to the CSV file

    Returns:
    --------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    """
    df = pd.read_csv(filepath)

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y


def create_preprocessing_pipeline(handle_outliers=True, feature_engineering=True):
    """
    Create a comprehensive preprocessing pipeline

    Parameters:
    -----------
    handle_outliers : bool, default=True
        Whether to handle outliers
    feature_engineering : bool, default=True
        Whether to perform feature engineering

    Returns:
    --------
    Pipeline
        Sklearn pipeline for preprocessing
    """
    steps = []

    # Step 1: Outlier handling (optional)
    if handle_outliers:
        steps.append(("outlier_handler", OutlierHandler(method="iqr", factor=1.5)))

    # Step 2: Feature engineering (optional)
    if feature_engineering:
        steps.append(
            ("feature_engineering", FeatureEngineering(create_interactions=True))
        )

    # Step 3: Scaling
    steps.append(("scaler", StandardScaler()))

    pipeline = Pipeline(steps)

    return pipeline


def save_pipeline(pipeline, filepath):
    """
    Save preprocessing pipeline to disk

    Parameters:
    -----------
    pipeline : Pipeline
        Sklearn pipeline to save
    filepath : str
        Path where to save the pipeline
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved to: {filepath}")


def load_pipeline(filepath):
    """
    Load preprocessing pipeline from disk

    Parameters:
    -----------
    filepath : str
        Path to the saved pipeline

    Returns:
    --------
    Pipeline
        Loaded sklearn pipeline
    """
    with open(filepath, "rb") as f:
        pipeline = pickle.load(f)
    print(f"Pipeline loaded from: {filepath}")
    return pipeline


def get_feature_info():
    """
    Get information about dataset features

    Returns:
    --------
    dict
        Dictionary containing feature information
    """
    feature_info = {
        "numerical_features": ["age", "trestbps", "chol", "thalach", "oldpeak"],
        "categorical_features": [
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "ca",
            "thal",
        ],
        "feature_descriptions": {
            "age": "Age in years",
            "sex": "Sex (1 = male; 0 = female)",
            "cp": "Chest pain type (1-4)",
            "trestbps": "Resting blood pressure (mm Hg)",
            "chol": "Serum cholesterol (mg/dl)",
            "fbs": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
            "restecg": "Resting electrocardiographic results (0-2)",
            "thalach": "Maximum heart rate achieved",
            "exang": "Exercise induced angina (1 = yes; 0 = no)",
            "oldpeak": "ST depression induced by exercise",
            "slope": "Slope of peak exercise ST segment (1-3)",
            "ca": "Number of major vessels colored by fluoroscopy (0-3)",
            "thal": "Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)",
            "target": "Heart disease presence (1 = disease; 0 = no disease)",
        },
    }
    return feature_info


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("=" * 70)
    print("PREPROCESSING PIPELINE DEMONSTRATION")
    print("=" * 70)

    # Load data
    data_path = "../data/heart_disease_clean.csv"
    X, y = load_data(data_path)

    print(f"\nOriginal data shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # Create pipeline
    pipeline = create_preprocessing_pipeline(
        handle_outliers=True, feature_engineering=True
    )

    print(f"\nPipeline steps:")
    for i, (name, transformer) in enumerate(pipeline.steps, 1):
        print(f"  {i}. {name}: {transformer.__class__.__name__}")

    # Fit and transform
    X_transformed = pipeline.fit_transform(X)

    print(f"\nTransformed data shape: {X_transformed.shape}")
    print(f"New features created: {X_transformed.shape[1] - X.shape[1]}")

    # Save pipeline
    save_path = "../models/preprocessing_pipeline.pkl"
    save_pipeline(pipeline, save_path)

    # Test loading
    loaded_pipeline = load_pipeline(save_path)
    X_test = loaded_pipeline.transform(X)

    print(f"\nPipeline loading successful!")
    print(f"Test transform shape: {X_test.shape}")

    # Feature info
    feature_info = get_feature_info()
    print(
        f"\nTotal features: {len(feature_info['numerical_features']) + len(feature_info['categorical_features'])}"
    )
    print(f"  - Numerical: {len(feature_info['numerical_features'])}")
    print(f"  - Categorical: {len(feature_info['categorical_features'])}")

    print("\n" + "=" * 70)
    print("PREPROCESSING MODULE READY!")
    print("=" * 70)
