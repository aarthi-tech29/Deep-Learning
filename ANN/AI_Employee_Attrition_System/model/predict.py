import numpy as np
import joblib

from tensorflow.keras.models import load_model

# --------------------------------------
# LOAD MODEL
# --------------------------------------

model = load_model(
    "model/attrition_model.h5"
)

# --------------------------------------
# LOAD SCALER
# --------------------------------------

scaler = joblib.load(
    "model/scaler.pkl"
)

# --------------------------------------
# SAMPLE EMPLOYEE DATA
# --------------------------------------

age = 30
income = 45000
job_satisfaction = 2
years_at_company = 3
performance_rating = 3
department = 1

employee = np.array([[
    age,
    income,
    job_satisfaction,
    years_at_company,
    performance_rating,
    department
]])

# --------------------------------------
# SCALE DATA
# --------------------------------------

employee = scaler.transform(
    employee
)

# --------------------------------------
# PREDICTION
# --------------------------------------

prediction = model.predict(
    employee
)

probability = round(
    float(prediction[0][0]) * 100,
    2
)

print("\nAttrition Probability:", probability,"%")

if probability >= 50:

    print(
        "Prediction : Likely To Resign"
    )

else:

    print(
        "Prediction : Not Likely To Resign"
    )