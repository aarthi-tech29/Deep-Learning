import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocess import preprocess_text

# Load Model
model = load_model(
    "model/sentiment_lstm.h5"
)

# Load Tokenizer
with open(
    "model/tokenizer.pkl",
    "rb"
) as f:
    tokenizer = pickle.load(f)

# Load Label Encoder
with open(
    "model/label_encoder.pkl",
    "rb"
) as f:
    encoder = pickle.load(f)


def predict_sentiment(review):

    review = preprocess_text(review)

    sequence = tokenizer.texts_to_sequences(
        [review]
    )

    padded = pad_sequences(
        sequence,
        maxlen=100,
        padding="post"
    )

    prediction = model.predict(
            padded,
            verbose=0
        )[0][0]

    print("Prediction Value:", prediction)

    if prediction >= 0.5:
            return "Positive 😊"
    else:
            return "Negative 😞"

if __name__ == "__main__":

    review = input(
        "Enter Review: "
    )

    result = predict_sentiment(
        review
    )

    print(
        "Sentiment:",
        result
    )
    print("Classes:", encoder.classes_)