import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(
    "dataset/employee_attrition.csv"
)

# --------------------------------------------------
# ENCODE DEPARTMENT
# --------------------------------------------------

department_encoder = LabelEncoder()

df["Department"] = department_encoder.fit_transform(
    df["Department"]
)

# --------------------------------------------------
# ENCODE ATTRITION
# Yes = 1
# No = 0
# --------------------------------------------------

attrition_encoder = LabelEncoder()

df["Attrition"] = attrition_encoder.fit_transform(
    df["Attrition"]
)

# --------------------------------------------------
# FEATURES & TARGET
# --------------------------------------------------

X = df.drop(
    "Attrition",
    axis=1
)

y = df["Attrition"]

# --------------------------------------------------
# SCALING
# --------------------------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(X)

# Save scaler
joblib.dump(
    scaler,
    "model/scaler.pkl"
)

# Save department encoder
joblib.dump(
    department_encoder,
    "model/label_encoder.pkl"
)

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# BUILD ANN MODEL
# --------------------------------------------------

model = Sequential()

model.add(
    Dense(
        64,
        activation="relu",
        input_shape=(X_train.shape[1],)
    )
)

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(
        16,
        activation="relu"
    )
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

# --------------------------------------------------
# COMPILE MODEL
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# --------------------------------------------------
# EVALUATE
# --------------------------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
with open("model/accuracy.txt", "w") as f:
    f.write(str(round(accuracy * 100, 2)))

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model.save(
    "model/attrition_model.h5"
)

print("\nModel Saved Successfully")

print("Scaler Saved Successfully")

print("Label Encoder Saved Successfully")