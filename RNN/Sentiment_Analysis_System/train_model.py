import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocess import preprocess_text

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("data/reviews.csv")

reviews = df["review"].fillna("").astype(str)
reviews = reviews.apply(preprocess_text)

labels = df["sentiment"].str.lower()

# ==========================
# ENCODE LABELS
# ==========================

encoder = LabelEncoder()

y = encoder.fit_transform(labels)

print("Classes:", encoder.classes_)
print("negative =", encoder.transform(["negative"]))
print("positive =", encoder.transform(["positive"]))

# ==========================
# TOKENIZATION
# ==========================

tokenizer = Tokenizer(
    num_words=5000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(reviews)

X = tokenizer.texts_to_sequences(reviews)

X = pad_sequences(
    X,
    maxlen=50,
    padding="post"
)

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# LSTM MODEL
# ==========================

model = Sequential()

model.add(
    Embedding(
        input_dim=5000,
        output_dim=128
    )
)

model.add(
    LSTM(
        64
    )
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

# ==========================
# COMPILE
# ==========================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# TRAIN
# ==========================

model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# ==========================
# EVALUATE
# ==========================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)

# ==========================
# SAVE MODEL
# ==========================

os.makedirs(
    "model",
    exist_ok=True
)

model.save(
    "model/sentiment_lstm.h5"
)

with open(
    "model/tokenizer.pkl",
    "wb"
) as f:
    pickle.dump(
        tokenizer,
        f
    )

with open(
    "model/label_encoder.pkl",
    "wb"
) as f:
    pickle.dump(
        encoder,
        f
    )

print("Model Saved Successfully")