import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(
    stopwords.words("english")
)


def preprocess_email(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove Email IDs
    text = re.sub(
        r"\S+@\S+",
        "",
        text
    )

    # Remove Numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove Special Characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Tokenization
    tokens = word_tokenize(
        text
    )

    # Stopword Removal
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)