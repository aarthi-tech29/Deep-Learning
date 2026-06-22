import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Dataset
df = pd.read_csv(
    "dataset/student_performance.csv"
)

# Encode Result
result_encoder = LabelEncoder()

df["Result"] = result_encoder.fit_transform(
    df["Result"]
)

joblib.dump(
    result_encoder,
    "model/result_encoder.pkl"
)

# Features
X = df[[
    "Attendance",
    "InternalMarks",
    "AssignmentCompletion",
    "LabPerformance"
]]

# Target
y = df["Result"]

# Scaling
scaler = StandardScaler()

X = scaler.fit_transform(X)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ANN Model
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
        1,
        activation="sigmoid"
    )
)

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    verbose=1
)

# Accuracy
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

accuracy = round(
    accuracy * 100,
    2
)

print(
    "Model Accuracy:",
    accuracy,
    "%"
)

with open(
    "model/accuracy.txt",
    "w"
) as f:

    f.write(
        str(accuracy)
    )

# Save Model
model.save(
    "model/performance_model.h5"
)

print(
    "Model Saved Successfully"
)