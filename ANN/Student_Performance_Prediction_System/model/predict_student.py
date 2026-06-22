import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

# Load Model
model = load_model(
    "model/performance_model.h5"
)

# Load Scaler
scaler = joblib.load(
    "model/scaler.pkl"
)

# Load Encoder
encoder = joblib.load(
    "model/result_encoder.pkl"
)

# User Input

attendance = int(
    input("Attendance (%): ")
)

internal_marks = int(
    input("Internal Marks: ")
)

assignment = int(
    input("Assignment Completion (%): ")
)

lab = int(
    input("Lab Performance (%): ")
)

student = np.array([[

    attendance,
    internal_marks,
    assignment,
    lab

]])

student = scaler.transform(
    student
)

prediction = model.predict(
    student
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

# Rank Prediction

average_score = (
    attendance +
    internal_marks +
    assignment +
    lab
) / 4

if average_score >= 90:
    rank = 1

elif average_score >= 80:
    rank = 5

elif average_score >= 70:
    rank = 10

elif average_score >= 60:
    rank = 20

else:
    rank = 30

# Weak Subject Analysis

if internal_marks < 50:

    weak_subject = "Theory Subjects"

elif lab < 50:

    weak_subject = "Lab Subjects"

else:

    weak_subject = "None"

print("\n===================")
print("Student Report")
print("===================")

print("Result :", result)

print(
    "Probability :",
    round(probability * 100, 2),
    "%"
)

print(
    "Predicted Rank :",
    rank
)

print(
    "Weak Subject :",
    weak_subject
)

# Attendance (%): 92
# Internal Marks: 88
# Assignment Completion (%): 95
# Lab Performance (%): 90

# Attendance (%): 55
# Internal Marks: 42
# Assignment Completion (%): 50
# Lab Performance (%): 45