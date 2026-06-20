from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

import pickle

from utils.preprocess import preprocess_email
from utils.pos_tagger import get_pos_tags

app = Flask(__name__)

# ====================================
# LOAD MODEL
# ====================================

with open(
    "model/spam_model.pkl",
    "rb"
) as f:

    model = pickle.load(
        f
    )

with open(
    "model/vectorizer.pkl",
    "rb"
) as f:

    vectorizer = pickle.load(
        f
    )

# ====================================
# HOME
# ====================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# ====================================
# PREDICT
# ====================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        email_text = data.get(
            "email",
            ""
        )

        if email_text.strip() == "":

            return jsonify({

                "prediction":
                "Please Enter Email",

                "tags":
                []

            })

        # ==========================
        # PREPROCESS
        # ==========================

        cleaned = preprocess_email(
            email_text
        )

        # ==========================
        # POS TAGGING
        # ==========================

        tags = get_pos_tags(
            cleaned
        )

        # ==========================
        # VECTORIZE
        # ==========================

        vector = vectorizer.transform(
            [cleaned]
        )

        # ==========================
        # PREDICT
        # ==========================

        prediction = model.predict(
            vector
        )[0]

        if prediction.lower() == "spam":

            result = "🚨 SPAM EMAIL"

        else:

            result = "✅ HAM EMAIL"

        return jsonify({

            "prediction":
            result,

            "tags":
            str(tags)

        })

    except Exception as e:

        print(e)

        return jsonify({

            "prediction":
            "Error",

            "tags":
            []

        })

# ====================================
# RUN
# ====================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
# Spam Email   
# Congratulations!

# You have won $1000 cash prize.

# Click here immediately to claim your reward.

# Ham Email
# Hello Team,

# Please attend the project meeting tomorrow at 10 AM.

# Thank you.

# POS tags
# Common POS Tags
# Tag	Meaning
# NN	Noun
# NNS	Plural Noun
# VB	Verb
# VBD	Verb Past Tense
# VBG	Verb + ing
# JJ	Adjective
# RB	Adverb
# PRP	Pronoun
# DT	Determiner (the, a, an)
