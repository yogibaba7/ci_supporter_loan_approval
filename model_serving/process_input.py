import joblib
import numpy as np 
import pandas as pd 


import joblib
import os

base_path = os.path.dirname(__file__)  # path to model_serving/

imputer = joblib.load(os.path.join(base_path, "..", "models", "imputer.pkl"))
scaler = joblib.load(os.path.join(base_path, "..", "models", "scaler.pkl"))
encoder = joblib.load(os.path.join(base_path, "..", "models", "encoder.pkl"))


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

# if __name__=='__main__':
#     preprocess_input()
    