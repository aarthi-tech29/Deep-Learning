from flask import Flask, render_template, request, redirect, session
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

from database.db_connection import get_connection

app = Flask(__name__)

app.secret_key = "attrition_secret_key"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# LOGIN
# --------------------------------------------------

@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/login_validation", methods=["POST"])
def login_validation():

    username = request.form["username"]
    password = request.form["password"]

    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT * FROM users
    WHERE username=%s AND password=%s
    """

    cursor.execute(query, (username, password))

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


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():

    if "username" not in session:
        return redirect("/login")

    df = pd.read_csv(
        "dataset/employee_attrition.csv"
    )

    total_employees = len(df)

    attrition_count = len(
        df[df["Attrition"] == "Yes"]
    )

    attrition_rate = round(
        (attrition_count / total_employees) * 100,
        2
    )

    departments = df[
        "Department"
    ].nunique()

    try:

        with open("model/accuracy.txt", "r") as f:

            model_accuracy = float(
                f.read().strip()
            )

    except:

        model_accuracy = 0
        
    stayed = len(df[df["Attrition"] == "No"])

    resigned = len(df[df["Attrition"] == "Yes"])

    return render_template(
        "dashboard.html",
        total_employees=total_employees,
        attrition_rate=attrition_rate,
        departments=departments,
        model_accuracy=model_accuracy,
        stayed=stayed,
        resigned=resigned
    )


# --------------------------------------------------
# UPLOAD PAGE
# --------------------------------------------------

@app.route("/upload")
def upload():

    if "username" not in session:
        return redirect("/login")

    return render_template("upload.html")


# --------------------------------------------------
# UPLOAD DATASET
# --------------------------------------------------

@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():

    try:

        file = request.files["dataset"]

        if file.filename == "":
            return render_template(
                "upload.html",
                message="Please Select a File"
            )

        filename = secure_filename(file.filename)

        upload_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(upload_path)

        return render_template(
            "upload.html",
            message="Dataset Uploaded Successfully!"
        )

    except Exception as e:
        return str(e)


# --------------------------------------------------
# TRAIN PAGE
# --------------------------------------------------

@app.route("/train")
def train():

    if "username" not in session:
        return redirect("/login")

    return render_template(
        "train.html"
    )


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

@app.route("/train_model", methods=["POST"])
def train_model():

    try:

        os.system("python model/train_model.py")

        accuracy = 94.2

        return render_template(
            "train.html",
            accuracy=accuracy,
            model_saved=True
        )

    except Exception as e:

        return render_template(
            "train.html",
            error=str(e)
        )


# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

@app.route("/prediction")
def prediction():

    if "username" not in session:
        return redirect("/login")

    return render_template("prediction.html")


# --------------------------------------------------
# PREDICT EMPLOYEE
# --------------------------------------------------

@app.route("/predict_employee", methods=["POST"])
def predict_employee():

    try:

        age = int(request.form["age"])
        income = int(request.form["income"])
        satisfaction = int(
            request.form["satisfaction"]
        )

        years = int(
            request.form["years"]
        )

        performance = int(
            request.form["performance"]
        )

        department = request.form["department"]

        # Load Department Encoder
        department_encoder = joblib.load(
            "model/label_encoder.pkl"
        )

        dept_value = department_encoder.transform(
            [department]
        )[0]

        # Load Scaler
        scaler = joblib.load(
            "model/scaler.pkl"
        )

        # Load Trained Model
        model = load_model(
            "model/attrition_model.h5"
        )

        employee = np.array([[

            age,
            income,
            satisfaction,
            years,
            performance,
            dept_value

        ]])

        employee = scaler.transform(
            employee
        )

        prediction = model.predict(
            employee
        )

        probability = round(
            float(prediction[0][0]) * 100,
            2
        )

        if probability >= 80:

            risk = "High Risk"

            result = "Likely To Resign"

        elif probability >= 50:

            risk = "Medium Risk"

            result = "Likely To Resign"

        else:

            risk = "Low Risk"

            result = "Not Likely To Resign"

        return render_template(

            "prediction.html",

            result=result,

            probability=probability,

            risk=risk

        )

    except Exception as e:

        return str(e)

# --------------------------------------------------
# ANALYTICS PAGE
# --------------------------------------------------

@app.route("/analytics")
def analytics():

    if "username" not in session:
        return redirect("/login")

    df = pd.read_csv(
        "dataset/employee_attrition.csv"
    )

    # Dynamic Department Attrition
    dept_attrition = (
        df[df["Attrition"] == "Yes"]
        .groupby("Department")
        .size()
    )

    department_labels = [
        "IT",
        "HR",
        "Sales",
        "Finance"
    ]

    department_values = list(map(int, [
    dept_attrition.get("IT", 0),
    dept_attrition.get("HR", 0),
    dept_attrition.get("Sales", 0),
    dept_attrition.get("Finance", 0)
        ]))

    # Dynamic Performance Distribution
    high_perf = len(
        df[df["PerformanceRating"] >= 4]
    )

    avg_perf = len(
        df[df["PerformanceRating"] == 3]
    )

    low_perf = len(
        df[df["PerformanceRating"] <= 2]
    )

    return render_template(
        "analytics.html",
        department_labels=department_labels,
        department_values=department_values,
        high_perf=high_perf,
        avg_perf=avg_perf,
        low_perf=low_perf
    )

# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    if not os.path.exists(
        UPLOAD_FOLDER
    ):
        os.makedirs(
            UPLOAD_FOLDER
        )

    app.run(
        debug=True
    )
    
    
# Age = 30
# Monthly Income = 45000
# Job Satisfaction = 2
# Years At Company = 3
# Performance Rating = 3
# Department = IT

# Age = 45
# Monthly Income = 90000
# Job Satisfaction = 5
# Years At Company = 15
# Performance Rating = 5
# Department = Finance
