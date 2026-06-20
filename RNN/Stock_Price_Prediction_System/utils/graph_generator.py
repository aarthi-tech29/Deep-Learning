import matplotlib.pyplot as plt

def generate_graph(actual,predicted):

    plt.figure(figsize=(10,5))

    plt.plot(
        actual,
        color='blue',
        label='Actual Price'
    )

    plt.plot(
        predicted,
        color='red',
        label='Predicted Price'
    )

    plt.title(
        'Stock Price Prediction'
    )

    plt.xlabel('Time')

    plt.ylabel('Price')

    plt.legend()

    plt.show()