# app.py
from flask import Flask, jsonify
from scripts.data_loader import load_data, feature_engineering
from scripts.train_model import train_model
from scripts.predict import predict_future_price
import os

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    data = load_data('data/crypto_data.csv')
    data = feature_engineering(data)
    
    model_path = 'models/crypto_lstm.h5'
    if not os.path.exists(model_path):
        model, scaler = train_model(data)
    else:
        model = load_model(model_path)
        _, scaler = train_model(data)
    
    predictions = predict_future_price(data, model, scaler)
    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(debug=True)
