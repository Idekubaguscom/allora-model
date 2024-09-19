import os
import requests
import time
import random
from datetime import datetime, timedelta

# Cache to store prices
price_cache = {}
CACHE_DURATION = timedelta(seconds=20)

def get_current_price(token, retries=3, backoff_factor=1):
    token = token.upper()
    current_time = datetime.now()

    # Check if we have a cached price and if it's still valid
    if token in price_cache:
        cached_price, cache_time = price_cache[token]
        if current_time - cache_time < CACHE_DURATION:
            return cached_price

    token_map = {
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BTC': 'bitcoin',
        'BNB': 'binancecoin',
        'ARB': 'arbitrum'
    }

    url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_map[token]}&vs_currencies=usd"
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": os.environ.get("")
    }

    for attempt in range(retries):
        try:
            # Simulate a delay
            time.sleep(random.uniform(1, 5))  # Random delay between 1 and 5 seconds

            # Simulate a failure (e.g., 20% of the time)
            if random.random() < 0.2:
                raise Exception("Simulated API failure in utils")

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                price = data[token_map[token]]['usd']
                
                # Update the cache
                price_cache[token] = (price, current_time)
                
                return price
            else:
                raise Exception(f"Failed to retrieve current price for {token}")
        except Exception as e:
            if attempt < retries - 1:
                wait_time = backoff_factor * (2 ** attempt)
                time.sleep(wait_time)
            else:
                raise e
