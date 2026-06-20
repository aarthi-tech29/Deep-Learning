import nltk
from nltk.corpus import gutenberg

nltk.download('gutenberg')

text = ""

for fileid in gutenberg.fileids():
    text += " ".join(gutenberg.words(fileid))
    text += "\n"

with open(
    "data/corpus.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(text)

print("Corpus created successfully")