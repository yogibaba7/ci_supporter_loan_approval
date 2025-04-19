import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch

import sys
import os

# Add root path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_serving.app import app  # or wherever your FastAPI app is defined

client = TestClient(app)

class TestWebsiteApp(unittest.TestCase):

    def test_homepage_loads(self):
        """Test if the homepage loads successfully."""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    @patch("app.requests.post")  # mock external API call
    def test_form_submission_success(self, mock_post):
        """Test form POST and result rendering."""
        # Mocking the API's JSON response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = 1  # pretend API says approved

        form_data = {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5000,
            "CoapplicantIncome": 0,
            "LoanAmount": 150,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Property_Area": "Urban"
        }

        response = client.post("/predict", data=form_data)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Loan Approved", response.text)

    @patch("app.requests.post")
    def test_form_submission_failure(self, mock_post):
        """Test form submission with API returning 'not approved'."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = 0  # pretend API says not approved

        form_data = {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 3000,
            "CoapplicantIncome": 1500,
            "LoanAmount": 120,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Property_Area": "Semiurban"
        }

        response = client.post("/predict", data=form_data)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Loan Not Approved", response.text)


if __name__ == "__main__":
    unittest.main()

