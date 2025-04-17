import unittest
import mlflow.pyfunc
from mlflow.types.schema import Schema, ColSpec

class TestModelSignature(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_uri = "models:/mymodel@production"  # using alias instead of stage
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
            ColSpec("double")
        ])

        self.assertEqual(signature.inputs, expected_input_schema, "Input schema mismatch")
        self.assertEqual(signature.outputs, expected_output_schema, "Output schema mismatch")

if __name__ == "__main__":
    unittest.main()