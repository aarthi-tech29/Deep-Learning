import json
import pickle
import random
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

with open("intents.json") as file:
    intents = json.load(file)

words = []
classes = []
documents = []

for intent in intents["intents"]:

    for pattern in intent["patterns"]:

        tokens = word_tokenize(pattern)

        words.extend(tokens)

        documents.append(
            (tokens, intent["tag"])
        )

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = sorted(
    list(set(words))
)

classes = sorted(
    list(set(classes))
)

pickle.dump(
    words,
    open("words.pkl", "wb")
)

pickle.dump(
    classes,
    open("classes.pkl", "wb")
)

training = []

for doc in documents:

    bag = []

    token_words = doc[0]

    for word in words:
        bag.append(
            1 if word in token_words else 0
        )

    output_row = [0] * len(classes)

    output_row[
        classes.index(doc[1])
    ] = 1

    training.append(
        [bag, output_row]
    )

random.shuffle(training)

training = np.array(
    training,
    dtype=object
)

X = list(training[:,0])
Y = list(training[:,1])

model = Sequential()

model.add(
    Dense(128,
    input_shape=(len(X[0]),),
    activation='relu')
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(64,
    activation='relu')
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
    len(Y[0]),
    activation='softmax')
)

sgd = SGD(
    learning_rate=0.01
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

model.fit(
    np.array(X),
    np.array(Y),
    epochs=200,
    batch_size=5
)

model.save(
    "chatbot_model.h5"
)

print("Model Trained")