from flask import Flask, render_template, request, jsonify
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocess import preprocess_text

app = Flask(__name__)

# ==========================
# LOAD FILES
# ==========================

model = load_model(
    "model/sentiment_lstm.h5"
)

with open(
    "model/tokenizer.pkl",
    "rb"
) as f:
    tokenizer = pickle.load(f)

with open(
    "model/label_encoder.pkl",
    "rb"
) as f:
    encoder = pickle.load(f)

# ==========================
# HOME
# ==========================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# ==========================
# PREDICT
# ==========================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        review = data.get(
            "review",
            ""
        )

        review = preprocess_text(
            review
        )

        sequence = tokenizer.texts_to_sequences(
            [review]
        )

        padded = pad_sequences(
            sequence,
            maxlen=50,
            padding="post"
        )

        prediction = model.predict(
            padded,
            verbose=0
        )[0][0]

        print("Review:", review)
        print("Prediction:", prediction)

        if prediction >= 0.5:

            sentiment = "Positive 😊"

            confidence = round(
                float(prediction * 100),
                2
            )

        else:

            sentiment = "Negative 😞"

            confidence = round(
                float((1 - prediction) * 100),
                2
            )

        return jsonify({
            "sentiment": sentiment,
            "confidence": float(confidence)
        })

    except Exception as e:

        print(e)

        return jsonify({

            "sentiment":
            "Error",

            "confidence":
            0

        })

# ==========================
# RUN
# ==========================

if __name__ == "__main__":

    app.run(
        debug=True
    )
    
# Positive Reviews
# This product is amazing
# I love this product very much
# Excellent quality and fast delivery
# Very satisfied with my purchase
# Best product I have ever bought

# Negative Reviews
# I hate this product
# Very bad quality
# Waste of money
# Terrible experience
# Worst purchase ever

# I love this movie
# I hate this movie

# This movie was fantastic. The acting was excellent and the story was amazing. I enjoyed every minute of it.
# This movie was terrible. I hated every minute of it. The acting was awful and the story was boring.