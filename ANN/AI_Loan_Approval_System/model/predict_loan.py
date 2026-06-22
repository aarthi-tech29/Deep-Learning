import numpy as np
import joblib

from tensorflow.keras.models import load_model

# ==========================
# LOAD MODEL
# ==========================

model = load_model(
    "model/loan_model.h5"
)

# ==========================
# LOAD SCALER
# ==========================

scaler = joblib.load(
    "model/scaler.pkl"
)

# ==========================
# LOAD ENCODER
# ==========================

encoder = joblib.load(
    "model/loan_encoder.pkl"
)

# ==========================
# USER INPUT
# ==========================

credit_score = int(
    input("Credit Score: ")
)

salary = int(
    input("Salary: ")
)

existing_loans = int(
    input("Existing Loans: ")
)

age = int(
    input("Age: ")
)

employment_years = int(
    input("Employment Years: ")
)

# ==========================
# PREPARE DATA
# ==========================

customer = np.array([[

    credit_score,
    salary,
    existing_loans,
    age,
    employment_years

]])

customer = scaler.transform(
    customer
)

# ==========================
# PREDICT
# ==========================

prediction = model.predict(
    customer
)

probability = float(
    prediction[0][0]
)

predicted_class = (
    1 if probability >= 0.5 else 0
)

result = encoder.inverse_transform(
    [predicted_class]
)[0]

# ==========================
# RISK SCORE
# ==========================

if result == "Approved":

    risk_score = round(
        (1 - probability) * 100,
        2
    )

else:

    risk_score = round(
        probability * 100,
        2
    )

# ==========================
# OUTPUT
# ==========================

print("\n=========================")
print("Loan Prediction Result")
print("=========================")

print("Decision :", result)

if result == "Approved":

    print(
        "Risk Score :",
        risk_score,
        "% (Low Risk)"
    )

else:

    print(
        "Risk Score :",
        risk_score,
        "% (High Risk)"
    )
    
# loan approved  
# Credit Score: 780
# Salary: 90000
# Existing Loans: 0
# Age: 35
# Employment Years: 10

# loan rejected
# Credit Score: 550
# Salary: 30000
# Existing Loans: 3
# Age: 24
# Employment Years: 2