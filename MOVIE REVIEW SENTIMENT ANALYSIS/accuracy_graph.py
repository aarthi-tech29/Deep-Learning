import matplotlib.pyplot as plt

accuracy = 0.92

plt.bar(
    ["Model Accuracy"],
    [accuracy]
)

plt.ylim(0,1)

plt.title(
    "Movie Sentiment Analysis Accuracy"
)

plt.show()