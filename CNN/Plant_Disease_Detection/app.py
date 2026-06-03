from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("plant_disease_model.h5")

classes = [
    "Healthy",
    "Leaf Spot",
    "Powdery Mildew",
    "Rust Disease"
]

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


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
        target_size=(128, 128)
    )

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]

    confidence = round(
        np.max(prediction) * 100,
        2
    )

    return render_template(
        'result.html',
        prediction=predicted_class,
        confidence=confidence,
        image_path=filepath
    )


if __name__ == '__main__':
    app.run(debug=True)