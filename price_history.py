import json
import os
from config import data_base_path
from datetime import datetime, timedelta
from utils import get_current_price
import logging

class PriceHistory:
    def __init__(self):
        self.history_file = os.path.join(data_base_path, 'price_history.json')
        self.history = self.load_history()

    def load_history(self):
        if os.path.exists(self.history_file) and os.path.getsize(self.history_file) > 0:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {}

    def save_prediction(self, token, timeframe, predicted_price, timestamp):
        self.update_actual_price(token, timeframe)  # Update actual prices for past predictions
        key = f"{token}_{timeframe}"
        if key not in self.history:
            self.history[key] = []
        initial_price = get_current_price(token)
        self.history[key].append({
            'timestamp': timestamp,
            'predicted_price': predicted_price,
            'initial_price': initial_price,
            'actual_price': None,
            'actual_timestamp': None
        })
        self._save_to_file()


    def update_actual_price(self, token, timeframe, current_price=None, current_time=None):
        if current_price is None:
            # You might need to implement this method or use an external service
            current_price = get_current_price(token)
        
        if current_time is None:
            current_time = datetime.now()
        elif isinstance(current_time, str):
            try:
                current_time = datetime.strptime(current_time, '%Y-%m-%dT%H:%M:%S.%f')
            except ValueError:
                current_time = datetime.strptime(current_time, '%Y-%m-%dT%H:%M:%S')

        key = f"{token}_{timeframe}"
        if key not in self.history:
            return

        predictions = self.history[key]
        seconds = self.timeframe_to_seconds(timeframe)

        for prediction in predictions:
            if prediction['actual_price'] is None:
                prediction_time = datetime.strptime(prediction['timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
                if current_time >= prediction_time + timedelta(seconds=seconds):
                    prediction['actual_price'] = current_price
                    prediction['accuracy'] = self.calculate_accuracy(prediction['predicted_price'], current_price)

        self._save_to_file()

    def calculate_accuracy(self, prediction, actual_price):
        predicted_price = prediction['predicted_price']
        prediction['accuracy'] = 100 - abs((actual_price - predicted_price) / actual_price * 100)

    def timeframe_to_seconds(self, timeframe):
        if timeframe.endswith('m'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 3600
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 86400
        else:
            raise ValueError(f"Invalid timeframe format: {timeframe}")

    def _save_to_file(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_history(self, token, timeframe, limit=100):
        key = f"{token}_{timeframe}"
        return self.history.get(key, [])[-limit:]

    def get_latest_prediction(self, token, timeframe):
        key = f"{token}_{timeframe}"
        if key in self.history and self.history[key]:
            return self.history[key][-1]
        return None
