import os  # Add this import for environment variables
import logging
import requests  # Add this import for making HTTP requests
import asyncio
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, Response, current_app
from model import download_data, train_model
from pmdarima import auto_arima
from config import model_file_path, training_price_data_path, supported_tokens, supported_timeframes
from sklearn.model_selection import GridSearchCV
from collections import deque
from datetime import datetime, timedelta
from price_history import PriceHistory
from utils import get_current_price
from threading import Timer
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from arch.__future__ import reindexing

app = Flask(__name__)
prediction_history = {}
#price_history = PriceHistory()
accuracy_results = {}

log_dir = '/app/logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=f'{log_dir}/inference.log', level=logging.INFO)
logging.basicConfig(filename=f'{log_dir}/auto_improve.log', level=logging.INFO)

def update_data():
    """Download price data and train models for all tokens and timeframes."""
    download_data()
    for token in supported_tokens:
        for timeframe in supported_timeframes:
            train_model(token, timeframe)
    

def get_inference(token, timeframe):
    """Load model and predict next price for given token and timeframe."""
    model_file = f"{model_file_path}_{token.lower()}_{timeframe}.pkl"
    with open(model_file, "rb") as f:
        loaded_model = pickle.load(f)

    # Predict the next step
    forecast = loaded_model.forecast(steps=1)
    predicted_price = forecast.values[0]

    # Get the current price
    current_price = get_current_price(token)

    # Calculate MA and Bollinger Bands
    if timeframe in ['10m', '20m', '60m']:
        spread = np.random.uniform(-0.0035, 0.0035)  # Random spread between -2% and +0.5%
    elif timeframe == '1d':
        spread = np.random.uniform(-0.025, 0.025)  # Random spread between -3% and +3%
    else:
        spread = 0  # No spread for other timeframes

    adjusted_price = current_price * (1 + spread)  # Use current price for adjustment
    predicted = predicted_price
    #prediction = (adjusted_price + predicted) / 2
    prediction = predicted_price

    # Store the prediction
    key = f"{token}_{timeframe}"
    prediction_history[key] = {
        "prediction": prediction,
        "time": datetime.now()
    }
    
    return prediction

def get_model(token, timeframe):
    model_file = f"{model_file_path}_{token.lower()}_{timeframe}.pkl"
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            return pickle.load(f)
    else:
        return train_model(token, timeframe)


timers = {}  
last_timer_set_time = {}

@app.route("/inference/<string:token>/<string:timeframe>")
def generate_inference(token, timeframe):
    model = get_model(token, timeframe)
    price_history = PriceHistory(token, timeframe)
    if token not in supported_tokens or timeframe not in supported_timeframes:
        return Response(json.dumps({"error": "Unsupported token or timeframe"}), status=400, mimetype='application/json')

    try:
        # Get the last known price
        price_data = pd.read_csv(os.path.join(training_price_data_path, f"{token.lower()}usdt_1d.csv"), index_col='date', parse_dates=True)
        last_known_price = price_data['close'].iloc[-1]

        # Predict with ARIMA
        arima_prediction = model['arima'].forecast(steps=1).iloc[-1]

        # Predict volatility with GARCH
        garch_forecast = model['garch'].forecast(horizon=1)
        volatility_prediction = garch_forecast.variance.iloc[-1].values[0] ** 0.5

        # Combine predictions
        prediction = arima_prediction + np.random.normal(0, volatility_prediction)

        # Ensure prediction is positive
        prediction = max(prediction, 0)
        price_history.save_prediction(token, timeframe, prediction, datetime.now().isoformat())

        latest_prediction = price_history.get_latest_prediction(token, timeframe)['predicted_price']
        if latest_prediction is None:
            logging.error(f"No previous prediction found for {token}/{timeframe}.")
        else :
            with app.app_context():
                check_accuracy(token, timeframe)

        return Response(str(prediction), status=200)
    except Exception as e:
        logging.error(f"Error during inference generation for {token}/{timeframe}: {str(e)}")
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

def run_check_accuracy(token, timeframe):
    with app.app_context():
        try:
            check_accuracy(token, timeframe)
        except Exception as e:
            logging.error(f"Error during accuracy check for {token}/{timeframe}: {str(e)}")
    
@app.route("/check_accuracy/<string:token>/<string:timeframe>")
def check_accuracy(token, timeframe):
    price_history = PriceHistory(token, timeframe)
    logging.info(f"check_accuracy called for {token}/{timeframe}")
    try:
        prediction = price_history.get_latest_prediction_with_accuracy(token, timeframe)
        logging.info(f"Prediction: {prediction}")
        if prediction is None:
            raise ValueError(f"No prediction found for {token}/{timeframe}")

        accuracy = prediction.get('accuracy')

        if accuracy is None:
            raise ValueError(f"Accuracy is None for {token}/{timeframe}")

        logging.info(f"Accuracy for {token}/{timeframe}: {accuracy}")
        if accuracy < 97:
            logging.info(f"Accuracy for {token}/{timeframe} is below threshold: {accuracy}")
            auto_improve(token, timeframe, accuracy)

        return jsonify({"accuracy": accuracy}), 200

    except ValueError as ve:
        logging.error(f"ValueError in check_accuracy for {token}/{timeframe}: {str(ve)}")
        return jsonify({"error": str(ve)}), 500

    except Exception as e:
        logging.error(f"Error in check_accuracy for {token}/{timeframe}: {str(e)}")
        return jsonify({"error": str(e)}), 500

def auto_improve(token, timeframe, accuracy):
    model_file = f"{model_file_path}_{token.lower()}_{timeframe}.pkl"
    
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    
    new_learning_rate = max(0.0001, min(0.01 / (1 + accuracy), 0.01))
    
    # Update model parameters if GARCH model exists
    if isinstance(model, dict) and 'garch' in model:
        garch_model = model['garch']
        if hasattr(garch_model, 'volatility'):
            garch_model.volatility.rescale = new_learning_rate
      
    price_series = combine_price_data(token, timeframe)
    if price_series is None:
        return
    
    if len(price_series) < 2:
        logging.error("Not enough data: price_series has fewer than 2 entries")
        return

    # Step 1: Fit ARIMA model
    price_series = pd.Series(price_series)
    arima_model = ARIMA(price_series, order=(1,1,1))  # Adjust the order if needed
    arima_results = arima_model.fit()

    # Step 2: Get residuals from ARIMA model
    residuals = arima_results.resid

    # Step 3: Fit GARCH model to the residuals
    garch_model = arch_model(residuals, vol='GARCH', p=1, q=1)
    garch_results = garch_model.fit()

    # Update the combined model
    combined_model = {
        'arima': arima_results,
        'garch': garch_results
    }

    # Save the updated combined model
    with open(model_file, "wb") as f:
        pickle.dump(combined_model, f)
    #train_model(token, timeframe)
    logging.info(f"Model for {token}/{timeframe} updated and saved.")

def combine_price_data(token, timeframe):
    # Load the historical price data from CSV
    price_history = PriceHistory(token, timeframe)
    price_data = pd.read_csv(os.path.join(training_price_data_path, f"{token.lower()}usdt_1d.csv"), index_col='date', parse_dates=True)
    csv_price_series = price_data['close']
    csv_price_series = csv_price_series.sort_index().asfreq('D')

    # Adjust for timeframe
    if timeframe == '10m':
        csv_price_series = csv_price_series.resample('10T').interpolate(method='linear')
    elif timeframe == '20m':
        csv_price_series = csv_price_series.resample('20T').interpolate(method='linear')
    elif timeframe == '60m':
        csv_price_series = csv_price_series.resample('60T').interpolate(method='linear')

    # Extract the latest actual prices from history
    history = price_history.get_history(token, timeframe)
    history_price_series = pd.Series(
        [entry['actual_price'] for entry in history if entry['actual_price'] is not None],
        index=[entry['timestamp'] for entry in history if entry['actual_price'] is not None]
    )
    history_price_series.index = pd.to_datetime(history_price_series.index)

    # Combine the two data sources
    combined_price_series = pd.concat([csv_price_series, history_price_series])
    combined_price_series = combined_price_series.sort_index().asfreq('D').interpolate(method='linear')

    return combined_price_series

@app.route("/history/<string:token>/<string:timeframe>")
def get_price_history(token, timeframe):
    price_history = PriceHistory(token, timeframe)
    history = price_history.get_history(token, timeframe)
    return jsonify(history)
    

@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return "0"
    except Exception:
        return "1"

if __name__ == "__main__":
   # update_data()
    app.run(host="0.0.0.0", port=8000)
