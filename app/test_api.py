"""
API Testing Script
Tests the Heart Disease Prediction API endpoints
"""

import requests
import json
from typing import Dict, List

# API base URL
BASE_URL = "http://localhost:8000"


def test_root():
    """Test root endpoint"""
    print("\n" + "=" * 60)
    print("TEST 1: Root Endpoint")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Root endpoint should return 200"
    print("✅ Root endpoint test PASSED")


def test_health():
    """Test health check endpoint"""
    print("\n" + "=" * 60)
    print("TEST 2: Health Check Endpoint")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Health endpoint should return 200"
    data = response.json()
    assert data['status'] == 'healthy', "API should be healthy"
    assert data['model_loaded'] == True, "Model should be loaded"
    print("✅ Health check test PASSED")


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "=" * 60)
    print("TEST 3: Model Info Endpoint")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Model info endpoint should return 200"
    data = response.json()
    assert 'model_name' in data, "Should have model_name"
    assert 'features_required' in data, "Should have features_required"
    assert len(data['features_required']) == 13, "Should have 13 features"
    print("✅ Model info test PASSED")


def test_features():
    """Test features endpoint"""
    print("\n" + "=" * 60)
    print("TEST 4: Features Endpoint")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/features")
    print(f"Status Code: {response.status_code}")
    print(f"Response (first 500 chars): {json.dumps(response.json(), indent=2)[:500]}...")

    assert response.status_code == 200, "Features endpoint should return 200"
    data = response.json()
    assert 'features' in data, "Should have features"
    assert data['count'] == 13, "Should have 13 features"
    print("✅ Features test PASSED")


def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "=" * 60)
    print("TEST 5: Single Prediction Endpoint")
    print("=" * 60)

    # Test patient data
    patient = {
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
        "thal": 6
    }

    print(f"Input: {json.dumps(patient, indent=2)}")

    response = requests.post(f"{BASE_URL}/predict", json=patient)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Prediction should return 200"
    data = response.json()
    assert 'prediction' in data, "Should have prediction"
    assert 'diagnosis' in data, "Should have diagnosis"
    assert data['prediction'] in [0, 1], "Prediction should be 0 or 1"
    print("✅ Single prediction test PASSED")

    return data


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "=" * 60)
    print("TEST 6: Batch Prediction Endpoint")
    print("=" * 60)

    # Test batch data
    patients = {
        "patients": [
            {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
                "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
                "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
            },
            {
                "age": 67, "sex": 1, "cp": 4, "trestbps": 160,
                "chol": 286, "fbs": 0, "restecg": 2, "thalach": 108,
                "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3
            },
            {
                "age": 54, "sex": 0, "cp": 2, "trestbps": 140,
                "chol": 268, "fbs": 0, "restecg": 2, "thalach": 160,
                "exang": 0, "oldpeak": 3.6, "slope": 3, "ca": 2, "thal": 3
            }
        ]
    }

    print(f"Input: {len(patients['patients'])} patients")

    response = requests.post(f"{BASE_URL}/predict/batch", json=patients)
    print(f"\nStatus Code: {response.status_code}")
    result = response.json()
    print(f"Response: {result['total']} predictions")

    for i, pred in enumerate(result['predictions']):
        print(f"\nPatient {i+1}: {pred['diagnosis']} (confidence: {pred.get('confidence', 'N/A')})")

    assert response.status_code == 200, "Batch prediction should return 200"
    assert result['total'] == 3, "Should have 3 predictions"
    assert len(result['predictions']) == 3, "Should have 3 predictions in list"
    print("\n✅ Batch prediction test PASSED")


def test_invalid_input():
    """Test error handling with invalid input"""
    print("\n" + "=" * 60)
    print("TEST 7: Invalid Input Handling")
    print("=" * 60)

    # Missing required field
    invalid_patient = {
        "age": 63,
        "sex": 1,
        # Missing cp field
        "trestbps": 145,
    }

    response = requests.post(f"{BASE_URL}/predict", json=invalid_patient)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    assert response.status_code == 422, "Should return 422 for invalid input"
    print("✅ Invalid input handling test PASSED")


def test_out_of_range_values():
    """Test error handling with out of range values"""
    print("\n" + "=" * 60)
    print("TEST 8: Out of Range Values Handling")
    print("=" * 60)

    # Age out of range
    invalid_patient = {
        "age": 150,  # Too high
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
        "thal": 6
    }

    response = requests.post(f"{BASE_URL}/predict", json=invalid_patient)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    assert response.status_code == 422, "Should return 422 for out of range values"
    print("✅ Out of range values handling test PASSED")


def run_all_tests():
    """Run all API tests"""
    print("\n" + "=" * 60)
    print("HEART DISEASE PREDICTION API - TEST SUITE")
    print("=" * 60)
    print(f"Testing API at: {BASE_URL}")

    tests = [
        test_root,
        test_health,
        test_model_info,
        test_features,
        test_single_prediction,
        test_batch_prediction,
        test_invalid_input,
        test_out_of_range_values
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} FAILED: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {str(e)}")
            failed += 1

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    print("=" * 60)

    return passed, failed


if __name__ == "__main__":
    try:
        passed, failed = run_all_tests()
        exit(0 if failed == 0 else 1)
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the API is running at http://localhost:8000")
        print("\nTo start the API:")
        print("  python app/main.py")
        print("  OR")
        print("  uvicorn app.main:app --reload")
        exit(1)
