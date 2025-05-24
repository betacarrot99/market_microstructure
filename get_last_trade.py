"""
Polling is a technique used to build real-time application where the client repeatedly sends requests to the server
to act on latest data. This is often done in a loop, where the program repeatedly checks for changes or updates.

Make a REST call every 2 seconds to Binance future exchange to print order book of BTCUSDT.

"""

import requests
import time
import logging
from datetime import datetime
import os
import csv

logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', level=logging.INFO)

# Base endpoint
URL = 'https://fapi.binance.com'

# https://binance-docs.github.io/apidocs/futures/en/#order-book
METHOD = '/fapi/v1/trades'


def get_last_trade(symbol):
    # GET request
    response = requests.get(URL + METHOD, params={'symbol': symbol})

    # convert to JSON object by response.json()
    trade_data = response.json()[0]

    # print best bid and offer price
    price = trade_data['price']
    qty = trade_data['qty']
    return [price, qty]

def append_to_csv(timestamp, price, qty):
    file_exists = os.path.isfile("result/live_data.csv")
    with open("result/live_data.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['timestamp', 'price', 'qty'])
        writer.writerow([timestamp, price, qty])

# use a while loop to query Binance once every 2 seconds
while True:
    # get best bid and offer
    prices = get_last_trade("BTCUSDT")
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    logging.info('Last trade:{}, qty: {}'.format(prices[0], prices[1]))
    append_to_csv(timestamp, prices[0], prices[1])

    # sleep
    time.sleep(2)
