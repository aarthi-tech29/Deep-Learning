from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

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

    prediction = model.predict(data)

    result = "Placed" if prediction[0][0] > 0.5 else "Not Placed"

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
