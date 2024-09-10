import json
import os
from config import data_base_path
from datetime import datetime, timedelta
from utils import get_current_price
import logging

class PriceHistory:
    def __init__(self, token, timeframe):
        self.history_file = os.path.join(data_base_path, f'price_history_{token}_{timeframe}.json')
        self.history = self.load_history(token, timeframe)

    def load_history(self, token, timeframe):
        if os.path.exists(self.history_file) and os.path.getsize(self.history_file) > 0:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {}

    def save_prediction(self, token, timeframe, predicted_price, timestamp):
        self.update_actual_price(token, timeframe)  # Update actual prices for past predictions
        key = f"{token}_{timeframe}"
        if key not in self.history:
            self.history[key] = []
        self.history[key].append({
            'timestamp': timestamp,
            'predicted_price': predicted_price,  # Save as a simple numeric value
            'actual_price': None,  # Placeholder for actual price
            'accuracy': None  # Placeholder for accuracy
        })
        # Save the full history to the JSON file
        with open(self.history_file, 'w') as file:
            json.dump(self.history, file, indent=2)
    
        # Update in-memory cache to keep only the last 4 predictions
        self.history[key] = self.history[key][-4:]
        

    def update_actual_price(self, token, timeframe, current_price=None, current_time=None):
        if current_price is None:
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

        # Get the most recent prediction
        prediction = self.history[key][-1]  # Retrieve the last prediction
        if prediction['actual_price'] is None:
            prediction_time = datetime.strptime(prediction['timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
            seconds = self.timeframe_to_seconds(timeframe)
            if current_time >= prediction_time:
                prediction['actual_price'] = current_price  # Update actual price
                # Calculate and update accuracy
                self.calculate_accuracy(prediction, current_price)

        self._save_to_file()

    def calculate_accuracy(self, prediction, actual_price):
        predicted_price = prediction['predicted_price']
        if actual_price is not None and predicted_price is not None:
            prediction['accuracy'] = 100 - abs((actual_price - predicted_price) / actual_price * 100)
        else:
            prediction['accuracy'] = None  # Set to None if prices are not available

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
        if not os.path.exists(self.history_file):
            return []
        
        with open(self.history_file, 'r') as file:
            data = json.load(file)

        key = f"{token}_{timeframe}"
        return data.get(key, [])[-limit:]

    def get_latest_prediction(self, token, timeframe):
        with open(self.history_file, 'r') as file:
            data = json.load(file)

        key = f"{token}_{timeframe}"
        if key in data and data[key]:
            return data[key][-1]

        return None
    
    def get_latest_prediction_with_accuracy(self, token, timeframe):
        key = f"{token}_{timeframe}"
        if key in self.history:
            for entry in reversed(self.history[key]):
                if entry.get('accuracy') is not None:
                    return entry
        return None
