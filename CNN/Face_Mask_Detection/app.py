from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("face_mask_model.h5")

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
    img_array = img_array/255.0
    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        result = "No Mask"
        confidence = prediction * 100
    else:
        result = "Mask"
        confidence = (1 - prediction) * 100

    return render_template(
        "result.html",
        prediction=result,
        confidence=round(confidence,2),
        image_path=filepath
    )


if __name__ == '__main__':
    app.run(debug=True)