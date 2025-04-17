from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np 
import pandas as pd
import os
import mlflow
from mlflow.pyfunc import load_model

# load model
# Load credentials from environment
dagshub_token = os.getenv("DAGSHUB_PAT")
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
model_name = "mymodel"
model_uri = f"models:/{model_name}@production"
model = load_model(model_uri)

app = FastAPI()

# Define the expected input format
class InputData(BaseModel):
    Gender: float
    Married : float
    Dependents : float
    Education : float
    Self_Employed : float
    ApplicantIncome: float
    CoapplicantIncome : float
    LoanAmount : float
    Loan_Amount_Term : float
    Credit_History : float
    Property_Area : float

@app.get("/")
def read_root():
    return 'welcome to my model'

@app.post("/predict")
def predict(data: InputData):
    input_array = pd.DataFrame([[data.Gender, data.Married, data.Dependents,data.Education,data.Self_Employed,data.ApplicantIncome,data.CoapplicantIncome,
                             data.LoanAmount,data.Loan_Amount_Term,data.Credit_History,data.Property_Area]],columns=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome'
                                                                                                                     ,'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area'])

    prediction = model.predict(input_array)
    return int(prediction[0])



