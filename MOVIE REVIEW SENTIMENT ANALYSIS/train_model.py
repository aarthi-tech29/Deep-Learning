import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

data = pd.read_csv("movie_reviews.csv")

X = data["review"]
y = data["sentiment"]

vectorizer = TfidfVectorizer(
    stop_words="english"
)

X = vectorizer.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test,predictions)

print("\nAccuracy:",accuracy)

print("\nClassification Report")
print(classification_report(y_test,predictions))

pickle.dump(
    model,
    open("sentiment_model.pkl","wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl","wb")
)

print("Model Saved Successfully")