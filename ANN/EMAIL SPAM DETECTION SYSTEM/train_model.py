import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("spam_emails.csv")

# Features and target
X = data["email_text"]
y = data["label"]

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english'
)

X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Perceptron Classifier
model = Perceptron()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(
    model,
    open("spam_model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl", "wb")
)

print("\nModel Saved Successfully!")