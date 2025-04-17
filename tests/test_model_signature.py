import unittest
import mlflow.pyfunc
from mlflow.types.schema import Schema, ColSpec
import os

class TestModelSignature(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load credentials from environment
        try:
            dagshub_token = os.getenv("DAGSHUB_PAT")
        except Exception as e:
            print(f"error -> {e}")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Set MLflow tracking URI
        dagshub_url = "https://dagshub.com"
        repo_owner = "yogibaba7"
        repo_name = "ci_supporter_loan_approval"
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        # Model name and alias to test
        cls.model_name = "mymodel"
        cls.model_uri = f"models:/{cls.model_name}@production"   # using alias instead of stage
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

    def test_model_signature(self):
        signature = self.model.metadata.signature
        self.assertIsNotNone(signature, "Model signature is missing!")

        expected_input_schema = Schema([
            ColSpec("double", "Gender"),
            ColSpec("double", "Married"),
            ColSpec("double", "Dependents"),
            ColSpec("double", "Education"),
            ColSpec("double", "Self_Employed"),
            ColSpec("double", "ApplicantIncome"),
            ColSpec("double", "CoapplicantIncome"),
            ColSpec("double", "LoanAmount"),
            ColSpec("double", "Loan_Amount_Term"),
            ColSpec("double", "Credit_History"),
            ColSpec("double", "Property_Area"),
        ])

        expected_output_schema = Schema([
            ColSpec("double", "Loan_Status")
        ])

        self.assertEqual(signature.inputs, expected_input_schema, "Input schema mismatch")
        self.assertEqual(signature.outputs, expected_output_schema, "Output schema mismatch")

if __name__ == "__main__":
    unittest.main()