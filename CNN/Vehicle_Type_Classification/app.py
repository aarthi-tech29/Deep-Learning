from flask import Flask,render_template,request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import csv

app = Flask(__name__)

model = load_model("vehicle_model.h5")

classes = [
    "Auto",
    "Bike",
    "Bus",
    "Car",
    "Truck"
]

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER,exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
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
    img_array = img_array/255.0
    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array)

    result = classes[np.argmax(prediction)]

    confidence = round(
        np.max(prediction)*100,
        2
    )

    with open(
        "prediction_history.csv",
        "a",
        newline=""
    ) as filecsv:

        writer = csv.writer(filecsv)

        writer.writerow([
            filepath,
            result,
            confidence
        ])

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence,
        image_path=filepath
    )

@app.route('/history')
def history():

    data = []

    if os.path.exists(
        "prediction_history.csv"
    ):

        with open(
            "prediction_history.csv",
            "r"
        ) as f:

            reader = csv.reader(f)

            data = list(reader)

    return render_template(
        "history.html",
        history=data
    )

if __name__=="__main__":
    app.run(debug=True)