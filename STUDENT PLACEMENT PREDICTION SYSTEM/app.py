from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import mysql.connector
import joblib

scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Aarthi123",
    database="placement_db"
)

cursor = db.cursor()

model = tf.keras.models.load_model("placement_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    aptitude = float(request.form['aptitude'])
    coding = float(request.form['coding'])
    communication = float(request.form['communication'])
    cgpa = float(request.form['cgpa'])

    data = np.array([[aptitude, coding, communication, cgpa]])
    data = scaler.transform(data)

    prediction = model.predict(data)
    print(prediction)

    result = "Placed" if prediction[0][0] > 0.5 else "Not Placed"
    
    cursor.execute("""
    INSERT INTO student_predictions
    (aptitude,coding,communication,cgpa,result)
    VALUES (%s,%s,%s,%s,%s)
    """,
    (aptitude, coding, communication, cgpa, result))

    db.commit()

    return render_template(
        'result.html',
        result=result
    )

if __name__ == '__main__':
    app.run(debug=True)
    
    
# Input
# Aptitude - 95
# Coding - 92
# Communication - 90
# CGPA - 9.2 // placed

# Aptitude - 65
# Coding - 60
# Communication - 58
# CGPA - 6.5 // not placed
