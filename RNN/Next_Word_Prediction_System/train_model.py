import os
import re
import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ==================================
# LOAD CORPUS
# ==================================

with open(
    "data/corpus.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read().lower()

# ==================================
# REMOVE UNWANTED WORDS
# ==================================

remove_words = [
    "chapter",
    "volume",
    "project",
    "gutenberg"
]

for word in remove_words:

    text = text.replace(
        word,
        " "
    )

# Remove Roman numerals

text = re.sub(
    r"\b[ivxlcdm]+\b",
    " ",
    text
)

# Remove years and numbers

text = re.sub(
    r"\d+",
    " ",
    text
)

# Remove punctuation

text = re.sub(
    r"[^a-zA-Z\s]",
    " ",
    text
)

# Remove extra spaces

text = re.sub(
    r"\s+",
    " ",
    text
).strip()

print("\nCorpus Preview:\n")
print(text[:500])

# ==================================
# TOKENIZER
# ==================================

tokenizer = Tokenizer()

tokenizer.fit_on_texts(
    [text]
)

total_words = len(
    tokenizer.word_index
) + 1

print(
    "\nVocabulary Size:",
    total_words
)

# ==================================
# CREATE SEQUENCES
# ==================================

words = text.split()

input_sequences = []

WINDOW_SIZE = 20

for i in range(
    WINDOW_SIZE,
    len(words)
):

    sequence = words[
        i - WINDOW_SIZE:i + 1
    ]

    token_list = tokenizer.texts_to_sequences(
        [" ".join(sequence)]
    )[0]

    input_sequences.append(
        token_list
    )

print(
    "Total Sequences:",
    len(input_sequences)
)

# ==================================
# PADDING
# ==================================

max_sequence_len = WINDOW_SIZE + 1

input_sequences = np.array(

    pad_sequences(
        input_sequences,
        maxlen=max_sequence_len,
        padding="pre"
    )

)

print(
    "Max Sequence Length:",
    max_sequence_len
)

# ==================================
# X AND Y
# ==================================

X = input_sequences[:, :-1]

y = input_sequences[:, -1]

y = to_categorical(
    y,
    num_classes=total_words
)

# ==================================
# MODEL
# ==================================

model = Sequential()

model.add(
    Embedding(
        input_dim=total_words,
        output_dim=128,
        input_length=max_sequence_len - 1
    )
)

model.add(
    LSTM(
        128
    )
)

model.add(
    Dense(
        total_words,
        activation="softmax"
    )
)

# ==================================
# COMPILE
# ==================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.build(
    input_shape=(None, max_sequence_len - 1)
)

model.summary()

# ==================================
# TRAIN
# ==================================

history = model.fit(
    X,
    y,
    epochs=20,
    batch_size=128,
    verbose=1
)

# ==================================
# SAVE MODEL
# ==================================

os.makedirs(
    "model",
    exist_ok=True
)

model.save(
    "model/next_word_lstm.h5"
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
    "model/max_sequence.pkl",
    "wb"
) as f:

    pickle.dump(
        max_sequence_len,
        f
    )

print(
    "\nModel Saved Successfully!"
)