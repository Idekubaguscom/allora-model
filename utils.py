import os
import requests
import time
import random

def get_current_price(token):
        # Simulate a delay
    time.sleep(random.uniform(1, 5))  # Random delay between 1 and 5 seconds

    # Simulate a failure (e.g., 20% of the time)
    if random.random() < 0.2:
        raise Exception("Simulated API failure")
    token_map = {
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BTC': 'bitcoin',
        'BNB': 'binancecoin',
        'ARB': 'arbitrum'
    }
    token = token.upper()
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_map[token]}&vs_currencies=usd"
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": os.environ.get("")
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data[token_map[token]]['usd']
    else:
        raise Exception(f"Failed to retrieve current price for {token}")