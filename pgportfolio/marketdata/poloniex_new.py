import json
import time
import sys
from datetime import datetime

from urllib.request import Request, urlopen
from urllib.parse import urlencode

from pgportfolio.constants import *

minute = 60
hour = minute * 60
day = hour * 24
week = day * 7
month = day * 30
year = day * 365

# Possible Commands
PUBLIC_COMMANDS = ['ticker24h', 'candles']

"""
Poloniex new public API

Note: Symbol convention is <base_currency>_<quote_currency> e.g. BTC_USDT, which is opposite from legacy system, 
which follows <quote_currency>_<base_currency> format e.g. USDT_BTC.
"""


class PoloniexNew:
    def __init__(self, APIKey='', Secret=''):
        self.APIKey = APIKey.encode()
        self.Secret = Secret.encode()
        # Conversions
        self.timestamp_str = lambda timestamp=time.time(), format="%Y-%m-%d %H:%M:%S": datetime.fromtimestamp(
            timestamp).strftime(format)
        self.str_timestamp = lambda datestr=self.timestamp_str(), format="%Y-%m-%d %H:%M:%S": int(
            time.mktime(time.strptime(datestr, format)))
        self.float_roundPercent = lambda floatN, decimalP=2: str(round(float(floatN) * 100, decimalP)) + "%"

        # PUBLIC COMMANDS
        self.marketTicker24h = lambda x=0: self.api('ticker24h')
        self.marketChart = lambda symbol, interval=DAY_1, startTime=int(round(time.time() * 1000)) - (week * 1), endTime=int(round(time.time() * 1000)): self.api(
            symbol + '/candles', {'interval': interval, 'startTime': startTime, 'endTime': endTime, 'limit': 500})

    #####################
    # Main Api Function #
    #####################
    def api(self, command, args={}):
        base_url = 'https://api.poloniex.com/markets/'
        url = base_url + command + '?' + urlencode(args)
        req = Request(
            url=url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        ret = urlopen(req)
        return json.loads(ret.read().decode(encoding='UTF-8'))
