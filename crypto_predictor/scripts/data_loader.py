# scripts/data_loader.py
import pandas as pd

def load_data(file_path):
    # Load historical crypto data
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    data = data.sort_index()
    # Selecting relevant features
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    # Optionally add any further preprocessing steps
    return data

if __name__ == "__main__":
    # Sample run for testing
    file_path = 'data/crypto_data.csv'
    data = load_data(file_path)
    print(data.head())


def feature_engineering(data):
    # Adding moving average as a feature
    data['MA_7'] = data['Close'].rolling(window=7).mean()
    data['MA_21'] = data['Close'].rolling(window=21).mean()
    # Adding a simple price change as a feature
    data['Price_Change'] = data['Close'].pct_change()
    return data.dropna()
