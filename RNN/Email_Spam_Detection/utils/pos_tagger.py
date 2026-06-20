from nltk import pos_tag
from nltk.tokenize import word_tokenize


def get_pos_tags(text):

    tokens = word_tokenize(
        text
    )

    tags = pos_tag(
        tokens
    )

    return tags