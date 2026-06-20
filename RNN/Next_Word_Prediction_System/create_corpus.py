import random

sentences = [
    "i love artificial intelligence",
    "i love machine learning",
    "i love deep learning",
    "artificial intelligence is the future",
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks",
    "python is a popular programming language",
    "data science uses machine learning",
    "artificial intelligence can solve complex problems",
    "machine learning models learn from data",
    "deep learning improves image recognition",
    "natural language processing understands text",
    "artificial intelligence powers modern applications",
    "machine learning helps make predictions",
    "python supports artificial intelligence development",
    "data science combines statistics and programming",
    "deep learning can recognize speech",
    "artificial intelligence improves automation",
    "machine learning improves accuracy",
    "deep learning improves computer vision",
    "students learn artificial intelligence",
    "engineers build machine learning models",
    "python is easy to learn",
    "data science is an exciting field",
    "artificial intelligence enhances productivity",
    "machine learning finds patterns in data",
    "deep learning uses multiple hidden layers",
    "artificial intelligence assists decision making",
    "machine learning is useful in healthcare",
    "deep learning powers autonomous vehicles",
    "artificial intelligence is transforming industries",
    "machine learning supports recommendation systems",
    "deep learning helps language translation",
    "python libraries simplify machine learning",
    "data science helps organizations grow",
    "artificial intelligence creates smart systems",
    "machine learning improves business decisions",
    "deep learning helps medical diagnosis",
    "natural language processing powers chatbots",
    "artificial intelligence changes the world"
]

with open(
    "data/corpus.txt",
    "w",
    encoding="utf-8"
) as f:

    for _ in range(1000):

        sentence = random.choice(sentences)

        f.write(sentence + "\n")

print("1000 lines generated successfully!")