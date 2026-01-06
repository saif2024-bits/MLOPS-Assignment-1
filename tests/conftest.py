"""
Pytest configuration and fixtures for heart disease prediction tests
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Define data path as a module-level variable
DATA_PATH = str(PROJECT_ROOT / "data" / "heart_disease_clean.csv")


@pytest.fixture(scope="session")
def data_path():
    """Fixture providing the path to the test data"""
    return DATA_PATH
