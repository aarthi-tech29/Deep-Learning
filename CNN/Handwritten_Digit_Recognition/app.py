from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("digit_model.h5")

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

    # Load image
    img = image.load_img(
        filepath,
        color_mode="grayscale",
        target_size=(28, 28)
    )

    img = image.img_to_array(img)

    # Invert colors
    img = 255 - img

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_digit = np.argmax(prediction)

    confidence = round(
        np.max(prediction) * 100,
        2
    )

    return render_template(
        "result.html",
        digit=predicted_digit,
        confidence=confidence,
        image_path=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)