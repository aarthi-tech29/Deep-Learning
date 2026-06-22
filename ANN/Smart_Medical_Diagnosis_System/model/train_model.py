import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Dataset
df = pd.read_csv(
    "dataset/medical_diagnosis.csv",
    keep_default_na=False
)

df["Symptoms"] = df["Symptoms"].replace("", "None")

# Encode Symptoms
symptom_encoder = LabelEncoder()
df["Symptoms"] = symptom_encoder.fit_transform(df["Symptoms"])

joblib.dump(
    symptom_encoder,
    "model/symptom_encoder.pkl"
)

# Encode Disease
disease_encoder = LabelEncoder()
df["Disease"] = disease_encoder.fit_transform(df["Disease"])

# Features and Target
X = df.drop("Disease", axis=1)
y = df["Disease"]

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(disease_encoder, "model/label_encoder.pkl")

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
        3,
        activation="softmax"
    )
)

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
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

# Evaluate
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Save Accuracy
with open("model/accuracy.txt", "w") as f:
    f.write(str(round(accuracy * 100, 2)))

# Save Model
model.save("model/disease_model.h5")

print("Model Saved Successfully")
print("Accuracy Saved Successfully")