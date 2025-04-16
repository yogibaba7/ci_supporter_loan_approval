
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import requests
import joblib
import numpy as np

from process_input import preprocess_input

app = FastAPI()
templates = Jinja2Templates(directory="templates")



# 🌐 Your deployed model API URL
API_URL = "https://ci-supporter-loan-approval.onrender.com"  # <-- replace with your actual URL


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