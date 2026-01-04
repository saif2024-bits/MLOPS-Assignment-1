"""
MLflow-integrated Model Training for Heart Disease Prediction
Tracks experiments, parameters, metrics, and artifacts using MLflow
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import pickle
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from pathlib import Path

from preprocessing import load_data, create_preprocessing_pipeline


class MLflowModelTrainer:
    """
    MLflow-integrated trainer for tracking experiments
    """

    def __init__(self, experiment_name="heart-disease-prediction", random_state=42):
        """
        Initialize MLflow Model Trainer

        Parameters:
        -----------
        experiment_name : str
            Name of the MLflow experiment
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.experiment_name = experiment_name
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessing_pipeline = None

        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment: {experiment_name}")

    def initialize_models(self):
        """
        Initialize models with hyperparameters
        """
        self.models = {
            'Logistic Regression': {
                'model': LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                    solver='liblinear',
                    C=1.0
                ),
                'params': {
                    'max_iter': 1000,
                    'solver': 'liblinear',
                    'C': 1.0,
                    'random_state': self.random_state
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': self.random_state
                }
            },
            'XGBoost': {
                'model': XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    eval_metric='logloss'
                ),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': self.random_state,
                    'eval_metric': 'logloss'
                }
            }
        }
        print(f"Initialized {len(self.models)} models")

    def train_model_with_mlflow(self, model_name, model_config, X_train, y_train, X_test, y_test, cv_results=None):
        """
        Train model and log everything to MLflow

        Parameters:
        -----------
        model_name : str
            Name of the model
        model_config : dict
            Model and parameters configuration
        X_train, y_train : Training data
        X_test, y_test : Test data
        cv_results : dict, optional
            Cross-validation results
        """
        with mlflow.start_run(run_name=model_name):

            # Log model type
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("framework", "sklearn" if model_name != "XGBoost" else "xgboost")

            # Log parameters
            print(f"\n{'='*70}")
            print(f"Training {model_name} with MLflow tracking...")
            print(f"{'='*70}")

            for param_name, param_value in model_config['params'].items():
                mlflow.log_param(param_name, param_value)

            # Log dataset info
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("test_samples", X_test.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])

            # Train model
            model = model_config['model']
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Probabilities
            if hasattr(model, 'predict_proba'):
                y_train_proba = model.predict_proba(X_train)[:, 1]
                y_test_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_train_proba = model.decision_function(X_train)
                y_test_proba = model.decision_function(X_test)

            # Calculate metrics
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'train_precision': precision_score(y_train, y_train_pred),
                'test_precision': precision_score(y_test, y_test_pred),
                'train_recall': recall_score(y_train, y_train_pred),
                'test_recall': recall_score(y_test, y_test_pred),
                'train_f1': f1_score(y_train, y_train_pred),
                'test_f1': f1_score(y_test, y_test_pred),
                'train_roc_auc': roc_auc_score(y_train, y_train_proba),
                'test_roc_auc': roc_auc_score(y_test, y_test_proba)
            }

            # Log all metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log CV results if available
            if cv_results and model_name in cv_results:
                mlflow.log_metric("cv_accuracy_mean", cv_results[model_name]['accuracy_mean'])
                mlflow.log_metric("cv_accuracy_std", cv_results[model_name]['accuracy_std'])
                mlflow.log_metric("cv_roc_auc_mean", cv_results[model_name]['roc_auc_mean'])
                mlflow.log_metric("cv_roc_auc_std", cv_results[model_name]['roc_auc_std'])

            # Calculate overfitting metric
            overfitting = metrics['train_accuracy'] - metrics['test_accuracy']
            mlflow.log_metric("overfitting_gap", overfitting)

            # Create confusion matrix plot
            cm = confusion_matrix(y_test, y_test_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'])
            ax.set_title(f'{model_name} - Confusion Matrix', fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')

            # Save and log confusion matrix
            cm_path = f"../screenshots/mlflow_{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
            os.makedirs(os.path.dirname(cm_path), exist_ok=True)
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(cm_path)
            plt.close()

            # Create ROC curve plot
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='blue', linewidth=2,
                   label=f'ROC Curve (AUC = {metrics["test_roc_auc"]:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{model_name} - ROC Curve', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

            # Save and log ROC curve
            roc_path = f"../screenshots/mlflow_{model_name.lower().replace(' ', '_')}_roc_curve.png"
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(roc_path)
            plt.close()

            # Log feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_

                # Create feature importance plot
                fig, ax = plt.subplots(figsize=(10, 8))
                sorted_idx = np.argsort(importances)[-15:]  # Top 15
                ax.barh(range(len(sorted_idx)), importances[sorted_idx],
                       color='steelblue', alpha=0.8, edgecolor='black')
                ax.set_yticks(range(len(sorted_idx)))
                ax.set_yticklabels([f'Feature {i}' for i in sorted_idx])
                ax.set_xlabel('Importance')
                ax.set_title(f'{model_name} - Feature Importance (Top 15)', fontweight='bold')
                ax.grid(axis='x', alpha=0.3)

                # Save and log feature importance
                fi_path = f"../screenshots/mlflow_{model_name.lower().replace(' ', '_')}_feature_importance.png"
                plt.savefig(fi_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(fi_path)
                plt.close()

            # Log model (use sklearn for all models to avoid XGBoost compatibility issues)
            mlflow.sklearn.log_model(model, "model")

            # Save model locally as well
            model_path = f"../models/mlflow_{model_name.lower().replace(' ', '_')}_model.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path)

            # Print summary
            print(f"\nModel: {model_name}")
            print(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
            print(f"  Test Precision: {metrics['test_precision']:.4f}")
            print(f"  Test Recall:    {metrics['test_recall']:.4f}")
            print(f"  Test F1:        {metrics['test_f1']:.4f}")
            print(f"  Test ROC-AUC:   {metrics['test_roc_auc']:.4f}")
            print(f"  Overfitting:    {overfitting:.4f}")
            print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")

            # Store results
            self.results[model_name] = {
                **metrics,
                'confusion_matrix': cm,
                'predictions': y_test_pred,
                'probabilities': y_test_proba,
                'run_id': mlflow.active_run().info.run_id
            }

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
        cv : int
            Number of folds

        Returns:
        --------
        dict
            CV results for each model
        """
        print(f"\n{'='*70}")
        print(f"Performing {cv}-fold cross-validation...")
        print(f"{'='*70}")

        cv_results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for model_name, model_config in self.models.items():
            print(f"\nCross-validating {model_name}...")
            model = model_config['model']

            # Multiple scoring metrics
            accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
            roc_auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
            precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision', n_jobs=-1)
            recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall', n_jobs=-1)
            f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1', n_jobs=-1)

            cv_results[model_name] = {
                'accuracy_mean': accuracy_scores.mean(),
                'accuracy_std': accuracy_scores.std(),
                'accuracy_scores': accuracy_scores,
                'roc_auc_mean': roc_auc_scores.mean(),
                'roc_auc_std': roc_auc_scores.std(),
                'roc_auc_scores': roc_auc_scores,
                'precision_mean': precision_scores.mean(),
                'precision_std': precision_scores.std(),
                'recall_mean': recall_scores.mean(),
                'recall_std': recall_scores.std(),
                'f1_mean': f1_scores.mean(),
                'f1_std': f1_scores.std()
            }

            print(f"  Accuracy: {cv_results[model_name]['accuracy_mean']:.4f} (+/- {cv_results[model_name]['accuracy_std']:.4f})")
            print(f"  ROC-AUC:  {cv_results[model_name]['roc_auc_mean']:.4f} (+/- {cv_results[model_name]['roc_auc_std']:.4f})")

        return cv_results

    def compare_experiments(self):
        """
        Create comparison visualization of all experiments
        """
        print(f"\n{'='*70}")
        print("Creating experiment comparison visualizations...")
        print(f"{'='*70}")

        # Metrics comparison
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']
        model_names = list(self.results.keys())

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        for idx, metric in enumerate(metrics):
            scores = [self.results[model][metric] for model in model_names]

            axes[idx].bar(range(len(model_names)), scores,
                         color=['steelblue', 'coral', 'lightgreen'],
                         alpha=0.8, edgecolor='black')
            axes[idx].set_xticks(range(len(model_names)))
            axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
            axes[idx].set_ylabel('Score')
            axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[idx].set_ylim([0, 1.1])
            axes[idx].grid(axis='y', alpha=0.3)

            for i, v in enumerate(scores):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

        # ROC curves comparison
        axes[5].set_title('ROC Curves Comparison', fontweight='bold')
        colors = ['blue', 'red', 'green']

        for idx, (model_name, results) in enumerate(self.results.items()):
            # Reconstruct ROC curve from saved probabilities
            # Note: We need y_test which we'll pass separately
            axes[5].plot([0, 1], [0, 1], 'k--', linewidth=1)
            axes[5].text(0.5, 0.4 + idx*0.1,
                        f'{model_name}: AUC={results["test_roc_auc"]:.3f}',
                        color=colors[idx], fontweight='bold')

        axes[5].set_xlabel('False Positive Rate')
        axes[5].set_ylabel('True Positive Rate')
        axes[5].grid(alpha=0.3)

        plt.tight_layout()

        # Save comparison plot
        comparison_path = "../screenshots/mlflow_experiment_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {comparison_path}")
        plt.close()

        return comparison_path

    def select_best_model(self, metric='test_roc_auc'):
        """
        Select best model based on metric

        Parameters:
        -----------
        metric : str
            Metric to use for selection

        Returns:
        --------
        tuple
            (best_model_name, best_score, best_run_id)
        """
        best_score = -1
        best_name = None
        best_run_id = None

        for model_name, results in self.results.items():
            score = results[metric]
            if score > best_score:
                best_score = score
                best_name = model_name
                best_run_id = results['run_id']

        self.best_model_name = best_name
        self.best_model = self.models[best_name]['model']

        print(f"\n{'='*70}")
        print(f"BEST MODEL SELECTION")
        print(f"{'='*70}")
        print(f"Best model: {best_name}")
        print(f"Best {metric}: {best_score:.4f}")
        print(f"MLflow run ID: {best_run_id}")
        print(f"{'='*70}")

        return best_name, best_score, best_run_id


def main():
    """
    Main MLflow training pipeline
    """
    print("="*70)
    print("HEART DISEASE PREDICTION - MLFLOW EXPERIMENT TRACKING")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    X, y = load_data("../data/heart_disease_clean.csv")
    print(f"   Data shape: {X.shape}")

    # Preprocessing
    print("\n2. Creating preprocessing pipeline...")
    preprocessing_pipeline = create_preprocessing_pipeline(
        handle_outliers=True,
        feature_engineering=True
    )
    X_transformed = preprocessing_pipeline.fit_transform(X)
    print(f"   Transformed shape: {X_transformed.shape}")

    # Train-test split
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Initialize MLflow trainer
    print("\n4. Initializing MLflow trainer...")
    trainer = MLflowModelTrainer(experiment_name="heart-disease-prediction", random_state=42)
    trainer.initialize_models()
    trainer.preprocessing_pipeline = preprocessing_pipeline

    # Cross-validation
    print("\n5. Cross-validation...")
    cv_results = trainer.cross_validate_models(X_transformed, y, cv=5)

    # Train all models with MLflow tracking
    print("\n6. Training models with MLflow tracking...")
    for model_name, model_config in trainer.models.items():
        trainer.train_model_with_mlflow(
            model_name, model_config,
            X_train, y_train, X_test, y_test,
            cv_results=cv_results
        )

    # Select best model
    print("\n7. Selecting best model...")
    best_name, best_score, best_run_id = trainer.select_best_model(metric='test_roc_auc')

    # Create comparison
    print("\n8. Creating experiment comparison...")
    trainer.compare_experiments()

    # Save results summary
    print("\n9. Saving results...")
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment_name': trainer.experiment_name,
        'best_model': best_name,
        'best_score': float(best_score),
        'best_run_id': best_run_id,
        'all_results': {
            model: {
                'test_accuracy': float(res['test_accuracy']),
                'test_roc_auc': float(res['test_roc_auc']),
                'run_id': res['run_id']
            }
            for model, res in trainer.results.items()
        }
    }

    results_path = "../models/mlflow_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"   Results saved to: {results_path}")

    # Print MLflow UI instructions
    print("\n" + "="*70)
    print("MLFLOW EXPERIMENT TRACKING COMPLETE!")
    print("="*70)
    print(f"\nBest Model: {best_name}")
    print(f"Test ROC-AUC: {best_score:.4f}")
    print(f"\nTo view experiments in MLflow UI:")
    print(f"  cd /Users/saif.afzal/Documents/M.Tech/MLOPS/heart-disease-mlops")
    print(f"  mlflow ui")
    print(f"  Then open: http://127.0.0.1:5000")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
