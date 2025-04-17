import unittest
import mlflow
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("model_loading_test")

class TestModelLoading(unittest.TestCase):

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
        cls.model_uri = f"models:/{cls.model_name}@staging"

    def test_model_loading(self):
        """
        Test if model can be successfully loaded from MLflow model registry
        using alias 'production'.
        """
        try:
            logger.info(f"Attempting to load model from URI: {self.model_uri}")
            model = mlflow.pyfunc.load_model(self.model_uri)
            self.assertIsNotNone(model)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.fail("Model loading failed.")

if __name__ == "__main__":
    unittest.main()

