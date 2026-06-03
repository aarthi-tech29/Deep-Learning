from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import csv

app = Flask(__name__)

model = load_model("xray_model.h5")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    img = image.load_img(
        filepath,
        target_size=(128,128)
    )

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred >= 0.5:
        result = "Pneumonia"
        probability = pred * 100
    else:
        result = "Normal"
        probability = (1 - pred) * 100

    with open(
        "history.csv",
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            filepath,
            result,
            round(probability,2)
        ])

    return render_template(
        "result.html",
        prediction=result,
        probability=round(probability,2),
        image_path=filepath
    )

@app.route('/dashboard')
def dashboard():

    records = []

    if os.path.exists("history.csv"):

        with open("history.csv","r") as f:
            reader = csv.reader(f)
            records = list(reader)

    return render_template(
        "dashboard.html",
        records=records
    )

if __name__ == "__main__":
    app.run(debug=True)