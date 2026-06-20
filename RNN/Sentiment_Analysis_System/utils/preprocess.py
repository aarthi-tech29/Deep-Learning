import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================
# STOPWORDS
# ==========================

stop_words = set(
    stopwords.words("english")
)

# Keep important sentiment words

custom_stopwords = stop_words - {
    "not",
    "no",
    "never",
    "hate",
    "love",
    "good",
    "bad",
    "best",
    "worst"
}

# ==========================
# LEMMATIZER
# ==========================

lemmatizer = WordNetLemmatizer()

# ==========================
# PREPROCESS FUNCTION
# ==========================

def preprocess_text(text):

    # Convert to lowercase

    text = str(text).lower()

    # Remove HTML tags

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove special characters

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization

    words = text.split()

    # Stopword Removal

    words = [
        word
        for word in words
        if word not in custom_stopwords
    ]

    # Lemmatization

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


# ==========================
# TEST
# ==========================

if __name__ == "__main__":

    sample = "I hate this movie. It was the worst movie ever!"

    print(
        preprocess_text(sample)
    )