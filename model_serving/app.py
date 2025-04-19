
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import requests
import joblib
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 🌐 Your deployed model API URL
API_URL = "https://ci-supporter-loan-approval.onrender.com"  # <-- replace with your actual URL


# process input ------------------------------------------------>
# base_path = os.path.dirname(__file__)  # path to model_serving/

# imputer = joblib.load(os.path.join(base_path, "..", "models", "imputer.pkl"))
# scaler = joblib.load(os.path.join(base_path, "..", "models", "scaler.pkl"))
# encoder = joblib.load(os.path.join(base_path, "..", "models", "encoder.pkl"))
imputer = joblib.load("models/imputer.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")

# ✨ Preprocessing function
def preprocess_input(data: dict):
    df = pd.DataFrame([data])
    
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

    # Encode (assume one-hot or similar)
    df[cat_cols] = encoder.transform(df[cat_cols])
    #df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out())
    
    # Scale
    df[num_cols] = scaler.transform(df[num_cols])

    input_list = df.values.tolist()[0]
    print(input_list)
    return input_list


user_input = {
    "Gender": 'Male',
    "Married": 'Yes',
    "Dependents": '0',
    "Education": 'Graduate',
    "Self_Employed": 'Yes',
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 0,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": 'Rural'
}

print(preprocess_input(user_input))



# 🏠 Homepage route
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# 🔮 Prediction route
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Gender: str = Form(...),
    Married: str = Form(...),
    Dependents: str = Form(...),
    Education: str = Form(...),
    Self_Employed: str = Form(...),
    ApplicantIncome: float = Form(...),
    CoapplicantIncome: float = Form(...),
    LoanAmount: float = Form(...),
    Loan_Amount_Term: float = Form(...),
    Credit_History: float = Form(...),
    Property_Area: str = Form(...)
):
    raw_input = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area
    }

    try:
        # 🔧 Preprocess input
        clean_input = preprocess_input(raw_input)
        print(clean_input)

        # 📡 Call deployed API with cleaned input
        response = requests.post(API_URL, json={"input": clean_input})

        prediction = response.json()
        result_text = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"
    except Exception as e:
        result_text = f"Error: {e}"

    return templates.TemplateResponse("index.html", {"request": request, "result": result_text})