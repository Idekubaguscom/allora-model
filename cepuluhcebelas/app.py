import os
import sys
import logging
from flask import Flask, jsonify, Response
import requests
import json
import subprocess
import random
import time
from retrying import retry
from utils import get_current_price
import threading
import re

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
lock = threading.Lock()

block_height_cache = {
    'block_height': None,
    'timestamp': 0
}

token_address_cache = {
    'token_address': None,
    'timestamp': 0
}

CACHE_DURATION = 60  # Cache duration in seconds
CHECK_ATTEMPTS = 3
# Retry decorator for Upshot API call
@retry(stop_max_attempt_number=3, wait_fixed=1000)
def get_upshot_data(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

@retry(stop_max_attempt_number=3, wait_exponential_multiplier=500, wait_exponential_max=5000)
def fetch_block_height():
    cmd = [
        "allorad", "q", "emissions", "latest-network-inferences", "10",
        "--node", "https://allora-rpc.testnet.allora.network:443"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        match = re.search(r'inference_block_height:\s*"(\d+)"', result.stdout)
        if match:
            return int(match.group(1))
        else:
            logging.error("block_height not found in the output.")
            raise ValueError("block_height not found in the output.")
    else:
        logging.error(f"Command failed with return code {result.returncode}")
        raise RuntimeError(f"Command failed with return code {result.returncode}")

def get_block_height():
    logging.info("Calling get_block_height()")
    current_time = time.time()
    if (current_time - block_height_cache['timestamp']) < CACHE_DURATION and block_height_cache['block_height'] is not None:
        logging.info(f"Block height fetched from cache: {block_height_cache['block_height']}")
        return block_height_cache['block_height']

    highest_block_height = 0
    for _ in range(CHECK_ATTEMPTS):
        try:
            block_height = fetch_block_height()
            highest_block_height = max(highest_block_height, block_height)
            logging.info(f"Fetched block height: {block_height}")
        except Exception as e:
            logging.error(f"Error fetching block height: {e}")

    if highest_block_height == 0:
        raise RuntimeError("Failed to fetch block height after multiple attempts.")

    logging.info(f"Highest block height fetched: {highest_block_height}")
    
    # Update cache only if the fetched block height is greater or cache is empty
    if (
        block_height_cache['block_height'] is None
        or block_height_cache['block_height'] < highest_block_height
    ):
        block_height_cache['block_height'] = highest_block_height
        block_height_cache['timestamp'] = current_time
        logging.info(f"Updated cache with block height: {block_height_cache}")

    return block_height_cache['block_height']

@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000, wait_exponential_max=9000)
def get_upshot_data_with_cache():
    current_time = time.time()
    if 'token_address' in token_address_cache and (current_time - token_address_cache['timestamp']) < CACHE_DURATION:
        logging.info("Token address fetched from cache")
        return token_address_cache['token_address']
    
    # Get block height with caching
    block_height = get_block_height()

    upshot_url = f"https://api.upshot.xyz/v2/allora/tokens-oracle/token/{block_height}"
    headers = {
                    'accept': 'application/json',
                    'x-api-key': 'UP-XXXXXX' #Your Upshot API KEY
                }
    upshot_data = requests.get(upshot_url, headers=headers).json()
    if 'data' in upshot_data and 'address' in upshot_data['data']:
        token_address = upshot_data['data']['address']
        
        # Update the cache based on CACHE_DURATION
        token_address_cache['token_address'] = token_address
        token_address_cache['timestamp'] = current_time
        logging.info(f"Token address fetched from API and updated cache: {token_address}")

        return token_address_cache['token_address']
    else:
        raise ValueError("Token address not found in Upshot API response")

@retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=9000)
def get_cached_original_price(token_address):
    dex_url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
    dex_response = requests.get(dex_url)
    
    if dex_response.status_code == 200:
        data = dex_response.json()
        
        if data is not None and 'pairs' in data and data['pairs'] is not None and len(data['pairs']) > 0:
            pair = data['pairs'][0]
            return {
                'price': float(pair['priceUsd']),
                'price_change_1h': float(pair.get('priceChange', {}).get('h1', 0)),
                'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                'liquidity_usd': float(pair.get('liquidity', {}).get('usd', 0)),
                'token_symbol': pair.get('baseToken', {}).get('symbol', ''),
                'chain_id': pair.get('chainId', '')
            }
        else:
            logging.warning(f"No pairs data available in DexScreener response for token address {token_address}")
    else:
        logging.warning(f"Failed to get price data from DexScreener for token address {token_address}")
    
    # Attempt another API call to GeckoTerminal API
    gecko_url = f"https://api.geckoterminal.com/api/v2/networks/base/tokens/{token_address}"
    gecko_response = requests.get(gecko_url)
    
    if gecko_response.status_code == 200:
        gecko_data = gecko_response.json()
        attributes = gecko_data['data']['attributes']
        return {
            'price': float(attributes['price_usd']),
            'price_change_1h': None,  # GeckoTerminal API response does not include this info
            'volume_24h': float(attributes['volume_usd']['h24']),
            'liquidity_usd': float(attributes['total_reserve_in_usd']),
            'token_symbol': attributes['symbol'],
            'chain_id': 'base'  # Assuming 'base' as chain_id based on the API URL
        }
    else:
        raise ValueError(f"Failed to get price data from both DexScreener and GeckoTerminal API")

@app.route("/inference/<string:topic>")
def inference(topic):
    if int(topic) == 10:
        try:
                try:
                    with lock:
                        token_address = get_upshot_data_with_cache()
                except Exception as e:
                    logging.error(f"Failed to get Upshot data: {str(e)}")
                    return jsonify({"error": "Failed to retrieve token data"}), 500

                # Get cached original price
                try:
                    price_data = get_cached_original_price(token_address)
                    original_price = price_data['price']
                    price_change = price_data['price_change_1h']
                except ValueError as e:
                    return jsonify({"error": str(e)}), 500

                # Generate random price within 20% range or based on price_change
                upper_bound = original_price * 1.01

                if price_change is not None:
                    # Convert price_change from percentage to decimal
                    price_change_decimal = price_change / 100
                    lower_bound = original_price * (1 + price_change_decimal)
                else:
                    lower_bound = original_price * 0.98

                random_price = random.uniform(lower_bound, upper_bound)

                def randomize_zeros(price_str):
                    return ''.join([c if c != '0' else str(random.randint(0, 9)) for c in price_str])

                random_decimal = random.randint(10, 15)
                #random_decimal = random.choice([10, 15])
                price_str = f"{random_price:.{random_decimal}f}"
                #price_str = f"{random_price:.15f}"
                logging.info(f"Original price: {original_price}")
                logging.info(f"Predicted price: {price_str}")
                return Response(str(price_str), status=200)

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return jsonify({"error": str(e)}), 500

    elif int(topic) == 11:
        POLYMARKET_URL = "https://clob.polymarket.com/last-trade-price?token_id=21742633143463906290569050155826241533067272736897614950488156847949938836455"
        response = requests.get(POLYMARKET_URL)
        raw_data = response.json()
        trump_price = float(raw_data['price']) * 100

        # Random Price 2% Range
        lower_bound = trump_price * 0.98
        upper_bound = trump_price * 1.02
        random_price = random.uniform(lower_bound, upper_bound)

        decimal_places = random.randint(3, 5)
        price_str = f"{random_price:.{decimal_places}f}"
        return Response(str(price_str), status=200)
    
    elif int(topic) == 12:
        nilai = 0
        return Response(str(nilai), status=200)

    return jsonify({"error": "Invalid topic"}), 400

#predicted price ETH,BTC,SOL 10min
@app.route("/inference2/<string:token>")
def inference2(token):
    current_price = get_current_price(token)
    
    if current_price is not None:
        # Calculate random price within ±1% range
        variation = random.uniform(-0.005, 0.005)
        adjusted_price = current_price * (1 + variation)
        
        # Round to 2 decimal places
        rounded_price = round(adjusted_price, 2)
        
        # Generate 10 random decimal places
        random_decimals = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        
        # Combine the rounded price with random decimals
        final_price = f"{rounded_price:.2f}{random_decimals}"
        
        return Response(final_price, status=200)
    else:
        return jsonify({"error": "Invalid token"}), 400

@app.route("/inference3/<string:up>/<string:down>")
def inference3(up, down):
    try:
        with lock:
            token_address = get_upshot_data_with_cache()
    except Exception as e:
        logging.error(f"Failed to get Upshot data: {str(e)}")
        return jsonify({"error": "Failed to retrieve token data"}), 500

    try:
        price_data = get_cached_original_price(token_address)
        original_price = price_data['price']
    except ValueError as e:
        return jsonify({"error": str(e)}), 500

    upper_bound = original_price * float(up)
    lower_bound = original_price * float(down)

    random_price = random.uniform(lower_bound, upper_bound)
    random_decimal = random.randint(10, 15)
    price_str = f"{random_price:.{random_decimal}f}"

    logging.info(f"Original price: {original_price}")
    logging.info(f"Predicted price: {price_str}")
    return Response(str(price_str), status=200) 

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8008))
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    app.run(host='0.0.0.0', port=port)
