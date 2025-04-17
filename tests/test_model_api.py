import unittest
import sys
import os
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api import app  # change to actual filename if different

client = TestClient(app)

class TestFastAPIEndpoints(unittest.TestCase):

    def test_read_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "welcome to my model")

    def test_predict_endpoint(self):
        payload = {
            "Gender": 1.0,
            "Married": 1.0,
            "Dependents": 0.0,
            "Education": 0.0,
            "Self_Employed": 0.0,
            "ApplicantIncome": 5000.0,
            "CoapplicantIncome": 0.0,
            "LoanAmount": 150.0,
            "Loan_Amount_Term": 360.0,
            "Credit_History": 1.0,
            "Property_Area": 2.0
        }

        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)

        prediction = response.json()
        self.assertTrue(isinstance(prediction, int) or isinstance(prediction, float), "Prediction must be a number")
        self.assertIn(prediction, [0, 1], "Prediction must be either 0 or 1")

if __name__ == "__main__":
    unittest.main()