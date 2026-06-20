import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from utils.preprocess import preprocess_email
from utils.pos_tagger import get_pos_tags

# ====================================
# LOAD DATASET
# ====================================

df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)

df = df[["label", "email"]]

print("Dataset Shape:", df.shape)

# ====================================
# PREPROCESS EMAILS
# ====================================

emails = []

for text in df["email"]:

    cleaned = preprocess_email(
        text
    )

    # POS Tagging Requirement
    tags = get_pos_tags(
        cleaned
    )

    emails.append(
        cleaned
    )

labels = df["label"]

# ====================================
# TF-IDF
# ====================================

vectorizer = TfidfVectorizer(
    max_features=5000
)

X = vectorizer.fit_transform(
    emails
)

# ====================================
# TRAIN TEST SPLIT
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42
)

# ====================================
# MODEL
# ====================================

model = MultinomialNB()

model.fit(
    X_train,
    y_train
)

# ====================================
# EVALUATE
# ====================================

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    "\nAccuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)

# ====================================
# SAVE FILES
# ====================================

os.makedirs(
    "model",
    exist_ok=True
)

with open(
    "model/spam_model.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )

with open(
    "model/vectorizer.pkl",
    "wb"
) as f:

    pickle.dump(
        vectorizer,
        f
    )

print(
    "\nModel Saved Successfully"
)