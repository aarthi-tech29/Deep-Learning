import pickle
import numpy as np

from tensorflow.keras.models import load_model

model = load_model(
    "model/stock_lstm.h5"
)

scaler = pickle.load(
    open("model/scaler.pkl","rb")
)

def predict_price(last_60_days):

    data=np.array(last_60_days)

    data=data.reshape(1,60,1)

    prediction=model.predict(data)

    prediction=scaler.inverse_transform(
        prediction
    )

    return prediction[0][0]