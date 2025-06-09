# trade_fetcher.py
import requests
import pandas as pd
import logging

class LiveTrade:
    def __init__(self, symbol="BTCUSDT", maxlen=1000):
        self.symbol = symbol
        self.maxlen = maxlen
        self.prices = []
        self.timestamps = []
        self.volumes = []
        self.api_url = 'https://fapi.binance.com/fapi/v1/trades'

    def get_last_trade(self):
        try:
            response = requests.get(self.api_url, params={'symbol': self.symbol}, timeout=5)
            trade = response.json()[-1]
            self.timestamps.append(trade['time'])
            self.prices.append(float(trade['price']))
            self.volumes.append(float(trade['qty']))

            self.timestamps = self.timestamps[-self.maxlen:]
            self.prices = self.prices[-self.maxlen:]
            self.volumes = self.volumes[-self.maxlen:]
        except Exception as e:
            logging.warning(f"Trade fetch error: {e}")

    def get_dataframe(self):
        return pd.DataFrame({
            "timestamp": self.timestamps,
            "price": self.prices,
            "volume": self.volumes
        })
