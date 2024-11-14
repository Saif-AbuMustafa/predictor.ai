# scripts/predict.py
import numpy as np
from tensorflow.keras.models import load_model
from data_loader import load_data, feature_engineering
from train_model import train_model

def predict_future_price(data, model, scaler, days=30):
    scaled_data = scaler.transform(data['Close'].values.reshape(-1, 1))
    last_data = scaled_data[-60:]  # Last 60 days

    predicted_prices = []
    for _ in range(days):
        last_data = np.reshape(last_data, (1, last_data.shape[0], 1))
        predicted_price = model.predict(last_data)
        predicted_prices.append(scaler.inverse_transform(predicted_price)[0, 0])
        last_data = np.append(last_data[0, 1:], predicted_price).reshape(-1, 1)

    return predicted_prices

if __name__ == "__main__":
    data = load_data('data/crypto_data.csv')
    data = feature_engineering(data)
    model, scaler = train_model(data)
    future_prices = predict_future_price(data, model, scaler, days=30)
    print("Predicted prices for the next 30 days:", future_prices)
