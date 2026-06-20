import json
import random
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("model/chatbot_model.h5")

tokenizer = pickle.load(
    open("model/tokenizer.pkl","rb")
)

encoder = pickle.load(
    open("model/label_encoder.pkl","rb")
)

max_len = pickle.load(
    open("model/max_len.pkl","rb")
)

with open("data/intents.json") as file:
    intents = json.load(file)

def get_response(text):

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=max_len
    )

    prediction = model.predict(padded)

    tag = encoder.inverse_transform(
        [np.argmax(prediction)]
    )[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(
                intent["responses"]
            )

    return "Sorry, I didn't understand."