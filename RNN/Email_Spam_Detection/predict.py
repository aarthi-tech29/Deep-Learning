import pickle

from utils.preprocess import preprocess_email
from utils.pos_tagger import get_pos_tags

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
# PREDICT FUNCTION
# ====================================

def predict_spam(email_text):

    cleaned = preprocess_email(
        email_text
    )

    # POS Tagging Requirement
    tags = get_pos_tags(
        cleaned
    )

    print(
        "\nPOS Tags:"
    )

    print(
        tags
    )

    vector = vectorizer.transform(
        [cleaned]
    )

    prediction = model.predict(
        vector
    )[0]

    return prediction


# ====================================
# TEST
# ====================================

if __name__ == "__main__":

    email = input(
        "Enter Email: "
    )

    result = predict_spam(
        email
    )

    print(
        "\nPrediction:",
        result.upper()
    )