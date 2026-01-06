"""
Model Training module for Heart Disease Prediction
Implements training pipeline for multiple ML models with cross-validation
"""

import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

# Add project root to path to allow imports from anywhere
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import create_preprocessing_pipeline, load_data


class ModelTrainer:
    """
    Class for training and evaluating multiple ML models
    """

    def __init__(self, random_state=42):
        """
        Initialize ModelTrainer

        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessing_pipeline = None

    def initialize_models(self):
        """
        Initialize models with default hyperparameters
        """
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=self.random_state, solver="liblinear", C=1.0
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric="logloss",
            ),
        }
        print(f"Initialized {len(self.models)} models")

    def train_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """
        Train a single model and evaluate it

        Parameters:
        -----------
        model : sklearn estimator
            Model to train
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        model_name : str
            Name of the model

        Returns:
        --------
        dict
            Dictionary containing all evaluation metrics
        """
        print(f"\nTraining {model_name}...")

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_train_proba = model.decision_function(X_train)
            y_test_proba = model.decision_function(X_test)

        # Calculate metrics
        metrics = {
            "model_name": model_name,
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "train_precision": precision_score(y_train, y_train_pred),
            "test_precision": precision_score(y_test, y_test_pred),
            "train_recall": recall_score(y_train, y_train_pred),
            "test_recall": recall_score(y_test, y_test_pred),
            "train_f1": f1_score(y_train, y_train_pred),
            "test_f1": f1_score(y_test, y_test_pred),
            "train_roc_auc": roc_auc_score(y_train, y_train_proba),
            "test_roc_auc": roc_auc_score(y_test, y_test_proba),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred),
            "classification_report": classification_report(y_test, y_test_pred),
            "predictions": y_test_pred,
            "probabilities": y_test_proba,
        }

        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Test ROC-AUC: {metrics['test_roc_auc']:.4f}")

        return metrics

    def cross_validate_models(self, X, y, cv=5):
        """
        Perform cross-validation for all models

        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target
        cv : int, default=5
            Number of cross-validation folds

        Returns:
        --------
        dict
            Cross-validation results for each model
        """
        print(f"\nPerforming {cv}-fold cross-validation...")

        cv_results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for model_name, model in self.models.items():
            print(f"\nCross-validating {model_name}...")

            # Accuracy scores
            accuracy_scores = cross_val_score(
                model, X, y, cv=skf, scoring="accuracy", n_jobs=-1
            )

            # ROC-AUC scores
            roc_auc_scores = cross_val_score(
                model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1
            )

            # Precision scores
            precision_scores = cross_val_score(
                model, X, y, cv=skf, scoring="precision", n_jobs=-1
            )

            # Recall scores
            recall_scores = cross_val_score(
                model, X, y, cv=skf, scoring="recall", n_jobs=-1
            )

            # F1 scores
            f1_scores = cross_val_score(model, X, y, cv=skf, scoring="f1", n_jobs=-1)

            cv_results[model_name] = {
                "accuracy_mean": accuracy_scores.mean(),
                "accuracy_std": accuracy_scores.std(),
                "accuracy_scores": accuracy_scores,
                "roc_auc_mean": roc_auc_scores.mean(),
                "roc_auc_std": roc_auc_scores.std(),
                "roc_auc_scores": roc_auc_scores,
                "precision_mean": precision_scores.mean(),
                "precision_std": precision_scores.std(),
                "precision_scores": precision_scores,
                "recall_mean": recall_scores.mean(),
                "recall_std": recall_scores.std(),
                "recall_scores": recall_scores,
                "f1_mean": f1_scores.mean(),
                "f1_std": f1_scores.std(),
                "f1_scores": f1_scores,
            }

            print(
                f"  Accuracy: {cv_results[model_name]['accuracy_mean']:.4f} (+/- {cv_results[model_name]['accuracy_std']:.4f})"
            )
            print(
                f"  ROC-AUC: {cv_results[model_name]['roc_auc_mean']:.4f} (+/- {cv_results[model_name]['roc_auc_std']:.4f})"
            )

        return cv_results

    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train all initialized models

        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Test data

        Returns:
        --------
        dict
            Results for all models
        """
        self.results = {}

        for model_name, model in self.models.items():
            self.results[model_name] = self.train_model(
                model, X_train, y_train, X_test, y_test, model_name
            )

        return self.results

    def select_best_model(self, metric="test_roc_auc"):
        """
        Select the best model based on specified metric

        Parameters:
        -----------
        metric : str, default='test_roc_auc'
            Metric to use for model selection

        Returns:
        --------
        tuple
            (best_model_name, best_model, best_score)
        """
        best_score = -1
        best_name = None

        for model_name, results in self.results.items():
            score = results[metric]
            if score > best_score:
                best_score = score
                best_name = model_name

        self.best_model_name = best_name
        self.best_model = self.models[best_name]

        print(f"\nBest model: {best_name}")
        print(f"Best {metric}: {best_score:.4f}")

        return best_name, self.best_model, best_score

    def save_model(self, model, filepath):
        """
        Save trained model to disk

        Parameters:
        -----------
        model : sklearn estimator
            Trained model
        filepath : str
            Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to: {filepath}")

    def save_results(self, filepath):
        """
        Save training results to JSON

        Parameters:
        -----------
        filepath : str
            Path to save results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert results to serializable format
        serializable_results = {}
        for model_name, results in self.results.items():
            serializable_results[model_name] = {
                "train_accuracy": float(results["train_accuracy"]),
                "test_accuracy": float(results["test_accuracy"]),
                "train_precision": float(results["train_precision"]),
                "test_precision": float(results["test_precision"]),
                "train_recall": float(results["train_recall"]),
                "test_recall": float(results["test_recall"]),
                "train_f1": float(results["train_f1"]),
                "test_f1": float(results["test_f1"]),
                "train_roc_auc": float(results["train_roc_auc"]),
                "test_roc_auc": float(results["test_roc_auc"]),
                "confusion_matrix": results["confusion_matrix"].tolist(),
            }

        # Add metadata
        output = {
            "timestamp": datetime.now().isoformat(),
            "best_model": self.best_model_name,
            "results": serializable_results,
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=4)

        print(f"Results saved to: {filepath}")

    def plot_model_comparison(self, save_path=None):
        """
        Create visualization comparing all models

        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        metrics = [
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
        ]
        model_names = list(self.results.keys())

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        # Plot each metric
        for idx, metric in enumerate(metrics):
            scores = [self.results[model][metric] for model in model_names]

            axes[idx].bar(
                range(len(model_names)),
                scores,
                color=["steelblue", "coral", "lightgreen"],
            )
            axes[idx].set_xticks(range(len(model_names)))
            axes[idx].set_xticklabels(model_names, rotation=45, ha="right")
            axes[idx].set_ylabel("Score")
            axes[idx].set_title(
                f'{metric.replace("_", " ").title()}', fontweight="bold"
            )
            axes[idx].set_ylim([0, 1.1])
            axes[idx].grid(axis="y", alpha=0.3)

            # Add value labels
            for i, v in enumerate(scores):
                axes[idx].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

        # Plot confusion matrices
        for idx, model_name in enumerate(model_names[:3]):
            if idx + 5 < len(axes):
                cm = self.results[model_name]["confusion_matrix"]
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=axes[idx + 5],
                    cbar=False,
                    square=True,
                )
                axes[idx + 5].set_title(
                    f"{model_name} - Confusion Matrix", fontweight="bold"
                )
                axes[idx + 5].set_ylabel("True Label")
                axes[idx + 5].set_xlabel("Predicted Label")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Comparison plot saved to: {save_path}")

        plt.show()

    def plot_roc_curves(self, X_test, y_test, save_path=None):
        """
        Plot ROC curves for all models

        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=(10, 8))

        colors = ["blue", "red", "green"]

        for idx, (model_name, model) in enumerate(self.models.items()):
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)

            plt.plot(
                fpr,
                tpr,
                color=colors[idx],
                linewidth=2,
                label=f"{model_name} (AUC = {auc:.3f})",
            )

        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"ROC curves saved to: {save_path}")

        plt.show()


def main():
    """
    Main training pipeline
    """
    # Define paths relative to project root
    data_path = PROJECT_ROOT / "data" / "heart_disease_clean.csv"
    models_dir = PROJECT_ROOT / "models"
    screenshots_dir = PROJECT_ROOT / "screenshots"

    # Create directories if they don't exist
    models_dir.mkdir(exist_ok=True)
    screenshots_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("HEART DISEASE PREDICTION - MODEL TRAINING")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    X, y = load_data(str(data_path))
    print(f"   Data shape: {X.shape}")
    print(f"   Target distribution:\n{y.value_counts()}")

    # Create preprocessing pipeline
    print("\n2. Creating preprocessing pipeline...")
    preprocessing_pipeline = create_preprocessing_pipeline(
        handle_outliers=True, feature_engineering=True
    )

    # Transform data
    print("\n3. Preprocessing data...")
    X_transformed = preprocessing_pipeline.fit_transform(X)
    print(f"   Transformed shape: {X_transformed.shape}")

    # Train-test split
    print("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train size: {X_train.shape[0]}")
    print(f"   Test size: {X_test.shape[0]}")

    # Initialize trainer
    print("\n5. Initializing models...")
    trainer = ModelTrainer(random_state=42)
    trainer.initialize_models()
    trainer.preprocessing_pipeline = preprocessing_pipeline

    # Cross-validation
    print("\n6. Cross-validation...")
    cv_results = trainer.cross_validate_models(X_transformed, y, cv=5)

    # Train all models
    print("\n7. Training models...")
    trainer.train_all_models(X_train, y_train, X_test, y_test)

    # Select best model
    print("\n8. Selecting best model...")
    best_name, best_model, best_score = trainer.select_best_model(metric="test_roc_auc")

    # Save models and results
    print("\n9. Saving models and results...")
    for model_name, model in trainer.models.items():
        safe_name = model_name.lower().replace(" ", "_")
        model_path = models_dir / f"{safe_name}_model.pkl"
        trainer.save_model(model, str(model_path))

    results_path = models_dir / "training_results.json"
    trainer.save_results(str(results_path))

    # Save preprocessing pipeline
    from preprocessing import save_pipeline

    pipeline_path = models_dir / "preprocessing_pipeline.pkl"
    save_pipeline(preprocessing_pipeline, str(pipeline_path))

    # Generate visualizations
    print("\n10. Generating visualizations...")
    comparison_path = screenshots_dir / "model_comparison.png"
    roc_path = screenshots_dir / "roc_curves.png"
    trainer.plot_model_comparison(save_path=str(comparison_path))
    trainer.plot_roc_curves(X_test, y_test, save_path=str(roc_path))

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nBest Model: {best_name}")
    print(f"Test ROC-AUC: {best_score:.4f}")
    print(f"\nAll models saved to: {models_dir}")
    print(f"Results saved to: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
