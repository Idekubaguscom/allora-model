Allora Model (GARCH + ARIMA)

Topic : 1-9

## Installation
- Setting Config.json pasphrase and RPC
- `cd $HOME/allora-model`
- `chmod +x init.config && ./init.config`
- Fill your Coingecko APIKEY in `utils.py and model.py`
- Setting Accuracy in app.py default <97%. if accuracy bellow 97 will trigger auto improve model.
- `docker compose up -d --build`


## fitur
- auto improve
- logs accuracy check using coingecko. if <97% retrain model with saved data history., you can adjust 97% this in app.py

  
