import os
import requests
import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import pickle
from config import data_base_path, model_file_path, training_price_data_path, supported_tokens
from arch import arch_model
import time
import random

def get_coingecko_url(token):
            # Simulate a delay
    time.sleep(random.uniform(1, 5))  # Random delay between 1 and 5 seconds

    # Simulate a failure (e.g., 20% of the time)
    if random.random() < 0.2:
        raise Exception("Simulated API failure")
    base_url = "https://api.coingecko.com/api/v3/coins/"
    token_map = {
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BTC': 'bitcoin',
        'BNB': 'binancecoin',
        'ARB': 'arbitrum'
    }
    
    token = token.upper()
    if token in token_map:
        url = f"{base_url}{token_map[token]}/market_chart?vs_currency=usd&days=30&interval=daily"
        return url
    else:
        raise ValueError("Unsupported token")

def download_data():
    os.makedirs(training_price_data_path, exist_ok=True)

    for token in supported_tokens:
        try:
            headers = {
                "accept": "application/json",
                "x-cg-demo-api-key": "" # replace with your API key
            }
            url = get_coingecko_url(token)
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for bad responses
            data = response.json()

            # Extract price data
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('date')
            df = df.drop('timestamp', axis=1)
            df = df.rename(columns={'price': 'close'})

            # Resample to ensure daily frequency and forward fill any missing data
            df = df.resample('D').ffill()

            # Save to CSV
            output_file = os.path.join(training_price_data_path, f"{token.lower()}usdt_1d.csv")
            df.to_csv(output_file)
            print(f"Downloaded and saved {token} data to {output_file}")

        except Exception as e:
            print(f"Error downloading data for {token}: {str(e)}")

def train_model(token, timeframe):
    # Load and preprocess data
    price_data = pd.read_csv(os.path.join(training_price_data_path, f"{token.lower()}usdt_1d.csv"), index_col='date', parse_dates=True)
    price_series = price_data['close']
    price_series = price_series.sort_index().asfreq('D')

    # Adjust for timeframe
    if timeframe == '10m':
        price_series = price_series.resample('10T').interpolate(method='linear')
    elif timeframe == '20m':
        price_series = price_series.resample('20T').interpolate(method='linear')

    # Step 1: Fit ARIMA model
    arima_model = ARIMA(price_series, order=(1,1,1))  # You may need to adjust the order
    arima_results = arima_model.fit()

    # Step 2: Get residuals from ARIMA model
    residuals = arima_results.resid

    # Step 3: Fit GARCH model to the residuals
    garch_model = arch_model(residuals, vol='GARCH', p=1, q=1)
    garch_results = garch_model.fit()

    # Create a dictionary to store both models
    combined_model = {
        'arima': arima_results,
        'garch': garch_results
    }

    # Save the combined model
    model_file = f"{model_file_path}_{token.lower()}_{timeframe}.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(combined_model, f)

    return combined_model

if __name__ == "__main__":
    download_data()
    train_model()