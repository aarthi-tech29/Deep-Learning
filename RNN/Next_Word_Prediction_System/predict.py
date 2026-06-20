import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================
# LOAD MODEL
# ==========================

model = load_model(
    "model/next_word_lstm.h5"
)

# ==========================
# LOAD TOKENIZER
# ==========================

with open(
    "model/tokenizer.pkl",
    "rb"
) as f:

    tokenizer = pickle.load(f)

# ==========================
# LOAD MAX SEQUENCE LENGTH
# ==========================

with open(
    "model/max_sequence.pkl",
    "rb"
) as f:

    max_sequence_len = pickle.load(f)

# ==========================
# PREDICT LOOP
# ==========================

while True:

    text = input(
        "\nEnter Text (type exit to quit): "
    )

    if text.lower() == "exit":
        break

    token_list = tokenizer.texts_to_sequences(
        [text.lower()]
    )[0]

    if len(token_list) == 0:

        print(
            "\nWords not found in vocabulary."
        )

        continue

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    prediction = model.predict(
        token_list,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction,
        axis=-1
    )[0]

    predicted_word = ""

    for word, index in tokenizer.word_index.items():

        if index == predicted_index:

            predicted_word = word
            break

    confidence = float(
        prediction[0][predicted_index]
    ) * 100

    print("\n====================")
    print("Input:", text)
    print("Predicted Word:", predicted_word)
    print(
        f"Confidence: {confidence:.2f}%"
    )

    print("\nTop 5 Predictions:")

    top5 = np.argsort(
        prediction[0]
    )[-5:]

    for idx in reversed(top5):

        word = tokenizer.index_word.get(
            idx,
            "Unknown"
        )

        score = float(
            prediction[0][idx]
        ) * 100

        print(
            f"{word} -> {score:.2f}%"
        )

    print("====================")