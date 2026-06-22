from flask import Flask, render_template, request, redirect, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from reportlab.pdfgen import canvas
from flask import send_file

import pandas as pd
import numpy as np
import joblib
import os

from database.db_connection import get_connection

app = Flask(__name__)

app.secret_key = "medical_secret_key"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------- HOME ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- LOGIN ----------------

@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/login_validation", methods=["POST"])
def login_validation():

    username = request.form["username"]
    password = request.form["password"]

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username=%s AND password=%s",
        (username, password)
    )

    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user:
        session["username"] = username
        return redirect("/dashboard")

    return render_template(
        "login.html",
        error="Invalid Username or Password"
    )


# ---------------- DASHBOARD ----------------

@app.route("/dashboard")
def dashboard():

    if "username" not in session:
        return redirect("/login")

    df = pd.read_csv(
        "dataset/medical_diagnosis.csv"
    )
    try:
        with open("model/accuracy.txt", "r") as f:
            model_accuracy = float(f.read().strip())

        print("Accuracy File:", model_accuracy)

    except Exception as e:
        print(e)
        model_accuracy = 0

    total_patients = len(df)

    diabetes_cases = len(
        df[df["Disease"] == "Diabetes"]
    )

    heart_cases = len(
        df[df["Disease"] == "Heart Disease"]
    )

    try:
        with open(
            "model/accuracy.txt",
            "r"
        ) as f:

            model_accuracy = float(
                f.read().strip()
            )

    except:
        model_accuracy = 0

    return render_template(
        "dashboard.html",
        total_patients=total_patients,
        diabetes_cases=diabetes_cases,
        heart_cases=heart_cases,
        model_accuracy=model_accuracy
    )


# ---------------- UPLOAD ----------------

@app.route("/upload")
def upload():

    if "username" not in session:
        return redirect("/login")

    return render_template(
        "upload.html"
    )


@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():

    file = request.files["dataset"]

    filename = secure_filename(
        file.filename
    )

    path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(path)

    return render_template(
        "upload.html",
        message="Dataset Uploaded Successfully!"
    )


# ---------------- TRAIN ----------------

@app.route("/train")
def train():

    if "username" not in session:
        return redirect("/login")

    return render_template(
        "train.html"
    )


@app.route("/train_model", methods=["POST"])
def train_model():

    result = os.system(
        "python model/train_model.py"
    )

    print("Train Status:", result)

    try:

        with open(
            "model/accuracy.txt",
            "r"
        ) as f:

            accuracy = float(
                f.read().strip()
            )

    except:
        accuracy = 0

    return render_template(
        "train.html",
        accuracy=accuracy,
        model_saved=True
    )


# ---------------- PREDICTION ----------------

@app.route("/prediction")
def prediction():

    if "username" not in session:
        return redirect("/login")

    return render_template(
        "prediction.html"
    )


@app.route("/predict_disease", methods=["POST"])
def predict_disease():

    bp = int(request.form["bp"])
    sugar = int(request.form["sugar"])
    heart = int(request.form["heart"])
    oxygen = int(request.form["oxygen"])

    symptom = request.form["symptom"]

    symptom_encoder = joblib.load(
        "model/symptom_encoder.pkl"
    )
    print("Selected symptom:", symptom)
    print(symptom_encoder.classes_)
    symptom_value = symptom_encoder.transform(
        [symptom]
    )[0]

    scaler = joblib.load(
        "model/scaler.pkl"
    )

    model = load_model(
        "model/disease_model.h5"
    )

    patient = np.array([[
        bp,
        sugar,
        heart,
        oxygen,
        symptom_value
    ]])

    patient = scaler.transform(
        patient
    )

    prediction = model.predict(
        patient
    )

    disease_index = np.argmax(
        prediction
    )

    probability = round(
        float(np.max(prediction)) * 100,
        2
    )

    disease_encoder = joblib.load(
    "model/label_encoder.pkl"
    )

    result = disease_encoder.inverse_transform(
        [disease_index]
    )[0]

    # Store report data in session
    session["disease"] = result
    session["probability"] = probability

    if probability >= 80:
        risk = "High Risk"
    elif probability >= 50:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    session["risk"] = risk

    return render_template(
        "prediction.html",
        result=result,
        risk=risk,
        probability=probability
    )


# ---------------- ANALYTICS ----------------

@app.route("/analytics")
def analytics():

    df = pd.read_csv(
        "dataset/medical_diagnosis.csv"
    )

    healthy_cases = len(
        df[df["Disease"] == "Healthy"]
    )

    diabetes_cases = len(
        df[df["Disease"] == "Diabetes"]
    )

    heart_cases = len(
        df[df["Disease"] == "Heart Disease"]
    )

    return render_template(
        "analytics.html",
        healthy_cases=healthy_cases,
        diabetes_cases=diabetes_cases,
        heart_cases=heart_cases
    )

# ---------------- REPORT ----------------

@app.route("/report")
def report():

    if "username" not in session:
        return redirect("/login")

    return render_template(
        "report.html"
    )


from reportlab.pdfgen import canvas
import os

@app.route("/generate_report", methods=["POST"])
def generate_report():

    patient_name = request.form["patient_name"]

    bp = int(request.form["bp"])
    sugar = int(request.form["sugar"])
    heart = int(request.form["heart"])
    oxygen = int(request.form["oxygen"])

    symptom = request.form["symptom"]

    symptom_encoder = joblib.load(
        "model/symptom_encoder.pkl"
    )

    symptom_value = symptom_encoder.transform(
        [symptom]
    )[0]

    scaler = joblib.load(
        "model/scaler.pkl"
    )

    model = load_model(
        "model/disease_model.h5"
    )

    patient = np.array([[

        bp,
        sugar,
        heart,
        oxygen,
        symptom_value

    ]])

    patient = scaler.transform(patient)

    prediction = model.predict(patient)

    disease_index = np.argmax(prediction)

    probability = round(
        float(np.max(prediction)) * 100,
        2
    )

    disease_encoder = joblib.load(
        "model/label_encoder.pkl"
    )

    disease = disease_encoder.inverse_transform(
        [disease_index]
    )[0]

    if probability >= 80:
        risk = "High Risk"
    elif probability >= 50:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    if not os.path.exists("reports"):
        os.makedirs("reports")

    pdf_file = f"{patient_name}_report.pdf"

    c = canvas.Canvas(
        f"reports/{pdf_file}"
    )

    c.setFont("Helvetica-Bold", 18)
    c.drawString(180, 800, "Medical Report")

    c.setFont("Helvetica", 12)

    c.drawString(100, 740,
                 f"Patient Name : {patient_name}")

    c.drawString(100, 710,
                 f"Blood Pressure : {bp}")

    c.drawString(100, 680,
                 f"Sugar Level : {sugar}")

    c.drawString(100, 650,
                 f"Heart Rate : {heart}")

    c.drawString(100, 620,
                 f"Oxygen Level : {oxygen}")

    c.drawString(100, 590,
                 f"Symptom : {symptom}")

    c.drawString(100, 540,
                 f"Disease : {disease}")

    c.drawString(100, 510,
                 f"Risk Analysis : {risk}")

    c.drawString(100, 480,
                 f"Probability : {probability}%")

    c.save()

    return render_template(
        "report.html",
        patient_name=patient_name,
        bp=bp,
        sugar=sugar,
        heart=heart,
        oxygen=oxygen,
        disease=disease,
        risk=risk,
        probability=probability,
        pdf_file=pdf_file
    )
    


@app.route("/download_report/<filename>")
def download_report(filename):

    return send_file(
        f"reports/{filename}",
        as_attachment=True
    )


# ---------------- LOGOUT ----------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")


# ---------------- MAIN ----------------

if __name__ == "__main__":

    if not os.path.exists(
        UPLOAD_FOLDER
    ):
        os.makedirs(
            UPLOAD_FOLDER
        )

    app.run(debug=True)
    
# Sample 1 – Diabetes
# Blood Pressure : 150
# Sugar Level    : 220
# Heart Rate     : 95
# Oxygen Level   : 97
# Symptoms       : Fever

# Sample 2 – Heart Disease
# Blood Pressure : 180
# Sugar Level    : 140
# Heart Rate     : 120
# Oxygen Level   : 90
# Symptoms       : Chest Pain

# Sample 3 – Healthy / Low Risk
# Blood Pressure : 120
# Sugar Level    : 95
# Heart Rate     : 72
# Oxygen Level   : 99
# Symptoms       : Normal

# Sample 4 – Borderline Case
# Blood Pressure : 135
# Sugar Level    : 130
# Heart Rate     : 88
# Oxygen Level   : 96
# Symptoms       : Headache

# Sample 5 – Severe Heart Condition
# Blood Pressure : 190
# Sugar Level    : 160
# Heart Rate     : 130
# Oxygen Level   : 85
# Symptoms       : Chest Pain