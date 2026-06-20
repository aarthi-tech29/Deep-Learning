from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocess import clean_text

app = Flask(__name__)

model = load_model(
    "model/next_word_lstm.h5"
)

with open(
    "model/tokenizer.pkl",
    "rb"
) as f:

    tokenizer = pickle.load(f)

with open(
    "model/max_sequence.pkl",
    "rb"
) as f:

    max_sequence_len = pickle.load(f)


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    data = request.get_json()

    text = data["text"]

    text = clean_text(text)

    sequence = tokenizer.texts_to_sequences(
        [text]
    )[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    predicted = np.argmax(
        model.predict(
            sequence,
            verbose=0
        ),
        axis=-1
    )

    next_word = ""

    for word, index in tokenizer.word_index.items():

        if index == predicted:

            next_word = word
            break

    return jsonify({
        "word": next_word
    })


if __name__ == "__main__":

    app.run(
        debug=True
    )