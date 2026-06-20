import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


# ==========================
# CONFIGURATION
# ==========================

STOCK_SYMBOL = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"

SEQUENCE_LENGTH = 60

MODEL_PATH = "model/stock_lstm.h5"
SCALER_PATH = "model/scaler.pkl"

os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)


# ==========================
# DOWNLOAD STOCK DATA
# ==========================

print("Downloading stock data...")

df = yf.download(
    STOCK_SYMBOL,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True
)

df.reset_index(inplace=True)

df.to_csv(
    "data/stock_data.csv",
    index=False
)

print("Dataset saved.")


# ==========================
# FEATURES
# ==========================

features = df[['Close', 'Volume']]

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(features)

pickle.dump(
    scaler,
    open(SCALER_PATH, "wb")
)

print("Scaler saved.")


# ==========================
# CREATE SEQUENCES
# ==========================

X = []
y = []

for i in range(SEQUENCE_LENGTH, len(scaled_data)):

    X.append(
        scaled_data[i-SEQUENCE_LENGTH:i]
    )

    y.append(
        scaled_data[i, 0]
    )

X = np.array(X)
y = np.array(y)

print("X Shape:", X.shape)
print("Y Shape:", y.shape)


# ==========================
# TRAIN TEST SPLIT
# ==========================

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]


# ==========================
# BUILD LSTM MODEL
# ==========================

model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(60, 2)
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    LSTM(
        64,
        return_sequences=False
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(1)
)

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

model.summary()


# ==========================
# TRAIN
# ==========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=32
)


# ==========================
# SAVE MODEL
# ==========================

model.save(MODEL_PATH)

print("\nModel Saved Successfully")
print(MODEL_PATH)