import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("resume_dataset.csv")

X = data["skills"]
y = data["selected"]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

model = LogisticRegression()

model.fit(X,y)

pickle.dump(
    model,
    open("resume_model.pkl","wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl","wb")
)

print("Model Trained Successfully")