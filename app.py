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
from config import model_file_path, training_price_data_path, supported_tokens, supported_timeframes
from sklearn.metrics import mean_absolute_percentage_error
from collections import deque
from datetime import datetime, timedelta
from price_history import PriceHistory
from utils import get_current_price
from threading import Timer
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

price_history = PriceHistory()


app = Flask(__name__)
prediction_history = {}
price_history = PriceHistory()
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
    if timeframe in ['10m', '20m']:
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
        
        check_accuracy(token, timeframe)
        def run_check_accuracy():
            with app.app_context():
                check_accuracy(token, timeframe)
    
        buffer_time = 120
        delay = pd.Timedelta(timeframe).total_seconds() + buffer_time
        
        current_time = datetime.now()
        
        if token in last_timer_set_time:
            time_since_last_set = (current_time - last_timer_set_time[token]).total_seconds()
            if time_since_last_set < 10:
                logging.info(f"Not resetting timer for {token} as it is within the spare time.")
                return Response(str(prediction), status=200)
        
        if token in timers:
            timers[token].cancel()
        
        timers[token] = Timer(delay, run_check_accuracy)
        timers[token].start()
        
        last_timer_set_time[token] = current_time
        
        logging.info(f"Setting Timer for check_accuracy with delay: {delay} seconds")
        
        return Response(str(prediction), status=200)
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')
    
@app.route("/check_accuracy/<string:token>/<string:timeframe>")
def check_accuracy(token, timeframe):
    logging.info(f"check_accuracy called for {token}/{timeframe}")  # Log when this function is called
    try:
               # Convert timeframe to seconds for processing
        if timeframe.endswith('m'):
            numeric_timeframe = int(timeframe[:-1]) * 60  # Convert minutes to seconds
        elif timeframe.endswith('d'):
            numeric_timeframe = int(timeframe[:-1]) * 86400  # Convert days to seconds
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}. Expected format like '10m', '20m', or '1d'.")

        current_price = get_current_price(token)
        if current_price is None:
            logging.error(f"Current price for {token} is None.")
            return jsonify({"error": "Current price not available"}), 500
        
        initial_price = price_history.get_latest_prediction(token, timeframe)['predicted_price']
        
        # Ensure initial_price is valid
        if initial_price is None:
            logging.error(f"No previous prediction found for {token}/{timeframe}.")
            return jsonify({"error": "No previous prediction available"}), 500
        
        price_history.update_actual_price(token, timeframe, current_price, datetime.now().isoformat())
        
        accuracy = 1 - abs(initial_price - current_price) / current_price
        
        logging.info(f"Accuracy for {token}/{timeframe}: {accuracy}")
        if accuracy < 0.98:
            logging.info(f"Accuracy for {token}/{timeframe} is below threshold: {accuracy}")
            auto_improve(token, timeframe)

        return jsonify({
            "token": token,
            "timeframe": timeframe,
            "predicted_price": initial_price,
            "actual_price": current_price,
            "accuracy": accuracy
        })
    except Exception as e:
        logging.error(f"Error in check_accuracy for {token}/{timeframe}: {str(e)}")
        return jsonify({"error": str(e)}), 500

def calculate_margin_error(predicted_price, actual_price):
    return abs(predicted_price - actual_price) / actual_price * 100

error_history = deque(maxlen=100)  # Store last 100 errors for each token/timeframe

learning_rate = 0.01
#ACCURACY_THRESHOLD = 0.95

def auto_improve(token, timeframe):
    history = price_history.get_history(token, timeframe)
    model_file = f"{model_file_path}_{token.lower()}_{timeframe}.pkl"
    
    with open(model_file, "rb") as f:
        combined_model = pickle.load(f)
    
    # Extract the latest actual and predicted prices
    latest_prediction = history[-1]
    actual_price = latest_prediction['actual_price']
    predicted_price = latest_prediction['predicted_price']
    
    # Calculate error
    error = actual_price - predicted_price
    
    # Prepare data for re-fitting
    prices = [h['actual_price'] for h in history]
    returns = np.diff(np.log(prices))
    
    # Re-fit ARIMA model
    arima_order = combined_model['arima'].order
    arima_model = ARIMA(prices, order=arima_order)
    arima_results = arima_model.fit()
    
    # Re-fit GARCH model
    garch_order = combined_model['garch'].order
    garch_model = arch_model(returns, vol='Garch', p=garch_order[0], q=garch_order[1])
    garch_results = garch_model.fit(disp='off')
    
    # Update combined model
    combined_model['arima'] = arima_results
    combined_model['garch'] = garch_results
    
    # Save the updated model
    with open(model_file, "wb") as f:
        pickle.dump(combined_model, f)
    
    logging.info(f"Model re-fitted for {token}_{timeframe}. New ARIMA order: {arima_order}, GARCH order: {garch_order}")
    logging.info(f"New ARIMA parameters: {arima_results.params}")
    logging.info(f"New GARCH parameters: {garch_results.params}")
    
    return abs(error)


@app.route("/history/<string:token>/<string:timeframe>")
def get_price_history(token, timeframe):
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
    update_data()
    app.run(host="0.0.0.0", port=8000)