import yfinance as yf
import pandas as pd
import numpy as np
import pickle

from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from tensorflow.keras.models import load_model

app = Flask(__name__)

# ==========================
# LOAD MODEL
# ==========================

model = load_model(
    "model/stock_lstm.h5"
)

scaler = pickle.load(
    open(
        "model/scaler.pkl",
        "rb"
    )
)

SEQUENCE_LENGTH = 60


# ==========================
# HOME PAGE
# ==========================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================
# PREDICTION API
# ==========================

@app.route("/predict", methods=["POST"])
def predict():


    symbol = request.json.get(
        "symbol",
        "AAPL"
    ).upper()

    try:

        df = yf.download(
            symbol,
            start="2020-01-01",
            progress=False,
            auto_adjust=True
        )

        if len(df) < 100:
            return jsonify({
                "error": "Not enough stock data."
            })

        # Fix yfinance multi-index issue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        features = df[["Close", "Volume"]].copy()

        scaled_data = scaler.transform(features)

        last_60_days = scaled_data[-60:]

        batch = np.array([last_60_days])

        future_predictions = []

        for _ in range(7):

            pred = model.predict(
                batch,
                verbose=0
            )[0][0]

            last_volume = batch[0][-1][1]

            inverse = scaler.inverse_transform(
                [[pred, last_volume]]
            )

            actual_price = round(
                float(inverse[0][0]),
                2
            )

            future_predictions.append(
                actual_price
            )

            new_row = np.array(
                [[[pred, last_volume]]]
            )

            batch = np.concatenate(
                [
                    batch[:, 1:, :],
                    new_row
                ],
                axis=1
            )

        # Safe conversion
        close_prices = (
            df["Close"]
            .tail(60)
            .values
            .flatten()
            .astype(float)
            .tolist()
        )

        current_price = round(
            float(
                df["Close"]
                .iloc[-1]
            ),
            2
        )

        trend = (
            "Bullish"
            if future_predictions[-1] > current_price
            else "Bearish"
        )

        return jsonify({

            "symbol": symbol,

            "current_price":
            current_price,

            "predicted_price":
            future_predictions[-1],

            "trend":
            trend,

            "actual":
            close_prices,

            "predicted":
            future_predictions
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        })




# ==========================
# RUN APP
# ==========================

if __name__ == "__main__":

    app.run(
        debug=True
    )
# AAPL  → Apple 
# TSLA   → Tesla
# MSFT   → Microsoft
# GOOGL  → Alphabet (Google)
# AMZN   → Amazon
# META   → Meta (Facebook)
# NVDA   → NVIDIA