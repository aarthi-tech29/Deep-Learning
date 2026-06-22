import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv(
    "dataset/loan_approval.csv"
)

# ==========================
# ENCODE TARGET
# ==========================

loan_encoder = LabelEncoder()

df["LoanApproved"] = loan_encoder.fit_transform(
    df["LoanApproved"]
)

# Save encoder

joblib.dump(
    loan_encoder,
    "model/loan_encoder.pkl"
)

print("Classes:", loan_encoder.classes_)

# ==========================
# FEATURES & TARGET
# ==========================

X = df.drop(
    "LoanApproved",
    axis=1
)

y = df["LoanApproved"]

# ==========================
# SCALING
# ==========================

scaler = StandardScaler()

X = scaler.fit_transform(X)

joblib.dump(
    scaler,
    "model/scaler.pkl"
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
# ANN MODEL
# ==========================

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
    epochs=100,
    batch_size=4,
    verbose=1
)

# ==========================
# EVALUATE
# ==========================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

accuracy = round(
    accuracy * 100,
    2
)

print("\nModel Accuracy:", accuracy, "%")

# ==========================
# SAVE ACCURACY
# ==========================

with open(
    "model/accuracy.txt",
    "w"
) as f:

    f.write(
        str(accuracy)
    )

# ==========================
# SAVE MODEL
# ==========================

model.save(
    "model/loan_model.h5"
)

print("\nModel Saved Successfully")
print("Scaler Saved Successfully")
print("Encoder Saved Successfully")