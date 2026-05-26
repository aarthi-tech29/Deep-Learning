import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# tensorflow → main deep learning library
# keras → helps build neural networks easily
# layers → used to add layers to the model

# =========================
# LOAD DATASET
# =========================

# Keep only top 10,000 common words
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

# | Variable  | Meaning                     |
# | --------- | --------------------------- |
# | `x_train` | movie reviews for training  |
# | `y_train` | answers (positive/negative) |
# | `x_test`  | reviews for testing         |
# | `y_test`  | correct test answers        |


# Make all reviews same length
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# Some reviews are:
# short
# long
# Neural networks need same-size input.
# So padding makes every review length = 200.

# =========================
# BUILD MODEL
# =========================

model = keras.Sequential([

    # Word embedding layer
    layers.Embedding(input_dim=10000, output_dim=32, input_length=200), #Converts word numbers into meaningful vectors.

    # Flatten data
    layers.Flatten(),# Converts 2D data into 1D.

    # Hidden layer
    layers.Dense(64, activation='relu'),# A fully connected neural network layer.
# 64 neurons.
# Each neuron learns patterns.
# activation='relu' helps model learn complex patterns.

    # Output layer
    layers.Dense(1, activation='sigmoid')
])
# Only two outputs:
# positive
# negative

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# Adjusts weights to reduce mistakes.
# Loss Function
# binary_crossentropy
# Used for 2-class classification.
# Measures prediction error.
# Lower loss = better model.

# =========================
# TRAIN MODEL
# =========================

model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.2
)

# epochs=3
# How many times model studies full dataset.
# 3 means:
# read all data 3 times

# batch_size=32
# Instead of all data at once:
# train 32 reviews at a time
# Faster and memory efficient.

# validation_split=0.2
# 20% training data used for checking performance.

# =========================
# TEST MODEL
# =========================

loss, accuracy = model.evaluate(x_test, y_test) # Tests model on unseen data.

print("Accuracy:", accuracy)

# Movie Reviews
#       ↓
# Convert words to numbers
#       ↓
# Padding
#       ↓
# Embedding Layer
#       ↓
# Dense Layers
#       ↓
# Positive / Negative Prediction