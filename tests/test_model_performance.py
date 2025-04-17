import unittest
import pandas as pd
import numpy as np
import mlflow.pyfunc
import os
from sklearn.metrics import accuracy_score

class TestModelPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load DagsHub credentials
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Set tracking URI and model alias
        dagshub_url = "https://dagshub.com"
        repo_owner = "yogibaba7"
        repo_name = "ci_supporter_loan_approval"
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        model_name = "mymodel"
        cls.model = mlflow.pyfunc.load_model(f"models:/{model_name}@staging")

        # Load test data
        cls.test_data = pd.read_csv("data/processed/test_processed.csv")
        cls.X_test = cls.test_data.drop("Loan_Status", axis=1)
        cls.y_true = cls.test_data["Loan_Status"]

    def test_model_accuracy(self):
        # Get predictions
        y_pred = self.model.predict(self.X_test)

        # Compute accuracy
        accuracy = accuracy_score(self.y_true, y_pred)

        # Set threshold (you can change this)
        threshold = 0.80

        print(f"Model accuracy: {accuracy:.4f}")
        self.assertGreaterEqual(accuracy, threshold, f"Model accuracy {accuracy:.4f} is below the threshold {threshold}")

if __name__ == "__main__":
    unittest.main()

