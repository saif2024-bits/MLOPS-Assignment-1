"""
Data Download Script for Heart Disease UCI Dataset
Author: MLOps Assignment
Date: 2025-12-13

This script downloads the Heart Disease dataset from UCI Machine Learning Repository
and saves it locally for further processing.
"""

import pandas as pd
import requests
import os
from io import StringIO

def download_heart_disease_data():
    """
    Download Heart Disease dataset from UCI repository.
    The dataset contains 14 attributes for heart disease diagnosis.

    Returns:
        pd.DataFrame: Downloaded dataset
    """
    # UCI Heart Disease dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    # Column names for the dataset
    column_names = [
        'age',           # Age in years
        'sex',           # Sex (1 = male; 0 = female)
        'cp',            # Chest pain type (1-4)
        'trestbps',      # Resting blood pressure (mm Hg)
        'chol',          # Serum cholesterol (mg/dl)
        'fbs',           # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        'restecg',       # Resting electrocardiographic results (0-2)
        'thalach',       # Maximum heart rate achieved
        'exang',         # Exercise induced angina (1 = yes; 0 = no)
        'oldpeak',       # ST depression induced by exercise
        'slope',         # Slope of peak exercise ST segment (1-3)
        'ca',            # Number of major vessels colored by fluoroscopy (0-3)
        'thal',          # Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
        'target'         # Diagnosis (0 = no disease, 1-4 = disease)
    ]

    print("Downloading Heart Disease dataset from UCI repository...")

    try:
        # Download the data
        response = requests.get(url)
        response.raise_for_status()

        # Parse the data
        df = pd.read_csv(StringIO(response.text), names=column_names, na_values='?')

        # Convert target to binary (0 = no disease, 1 = disease present)
        df['target'] = (df['target'] > 0).astype(int)

        print(f"Dataset downloaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        return df

    except Exception as e:
        print(f"Error downloading data: {e}")
        raise

def save_data(df, filepath='heart_disease.csv'):
    """
    Save the dataset to a CSV file.

    Args:
        df (pd.DataFrame): Dataset to save
        filepath (str): Path where to save the file
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filepath)

    df.to_csv(full_path, index=False)
    print(f"\nDataset saved to: {full_path}")

    return full_path

def main():
    """Main function to download and save the dataset."""
    print("="*70)
    print("Heart Disease UCI Dataset - Data Acquisition")
    print("="*70)

    # Download the data
    df = download_heart_disease_data()

    # Save to CSV
    filepath = save_data(df)

    # Display basic info
    print("\n" + "="*70)
    print("Dataset Information:")
    print("="*70)
    print(f"Total Records: {len(df)}")
    print(f"Total Features: {len(df.columns) - 1}")
    print(f"Target Variable: binary (0 = no disease, 1 = disease)")
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print("\n" + "="*70)
    print("Data download complete! Ready for EDA.")
    print("="*70)

if __name__ == "__main__":
    main()
