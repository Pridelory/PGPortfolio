from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from pgportfolio.marketdata.poloniex_new import PoloniexNew
from pgportfolio.tools.data import get_chart_until_success
import pandas as pd
from datetime import datetime
import logging
from pgportfolio.constants import *


class CoinList(object):
    def __init__(self, end, volume_average_days=1, volume_forward=0):
        self._polo = PoloniexNew()
        # connect the internet to accees volumes
        # vol = self._polo.marketVolume()
        ticker24h = self._polo.marketTicker24h()
        pairs = []
        coins = []
        volumes = []
        prices = []

        logging.info("select coin online from %s to %s" % (datetime.fromtimestamp(end - (DAY * volume_average_days) -
                                                                                  volume_forward).
                                                           strftime('%Y-%m-%d %H:%M'),
                                                           datetime.fromtimestamp(end - volume_forward).
                                                           strftime('%Y-%m-%d %H:%M')))
        for entry in ticker24h:
            symbol = entry['symbol']
            if symbol.startswith("BTC_") or symbol.endswith("_BTC"):
                if 'TUSD' in symbol or 'USDD' in symbol or 'USDC' in symbol or 'ADA' in symbol or 'TRU' in symbol:
                    continue
                pairs.append(symbol)
                if symbol.endswith('_BTC'):
                    coins.append(symbol.split('_')[0])
                    if entry['markPrice'] != '':
                        price = float(entry['markPrice'])
                    else:
                        price = 0
                    prices.append(price)
                else:
                    coins.append('reversed_' + symbol.split('_')[1])
                    prices.append(1.0 / float(entry['markPrice']))
                volumes.append(self.__get_total_volume(pair=symbol, global_end=end,
                                                       days=volume_average_days,
                                                       forward=volume_forward))
        self._df = pd.DataFrame({'coin': coins, 'pair': pairs, 'volume': volumes, 'price': prices})
        self._df = self._df.set_index('coin')

    @property
    def allActiveCoins(self):
        return self._df

    @property
    def polo(self):
        return self._polo

    def get_chart_until_success(self, pair, start, period, end):
        start = start * 1000
        end = end * 1000
        period = MINUTE_5
        return get_chart_until_success(self._polo, pair, start, period, end)

    # get several days volume
    def __get_total_volume(self, pair, global_end, days, forward):
        start = global_end - (DAY * days) - forward
        end = global_end - forward
        chart = self.get_chart_until_success(pair=pair, period=DAY_1, start=start, end=end)
        result = 0.0
        for one_day in chart:
            # get quote units
            if pair.startswith("BTC_"):
                result += float(one_day[4])
            else:
                result += float(one_day[5])
        return result

    def topNVolume(self, n=5, order=True, minVolume=0):
        if minVolume == 0:
            r = self._df.loc[self._df['price'] > 2e-6]
            r = r.sort_values(by='volume', ascending=False)[:n]
            # print(r)
            if order:
                return r
            else:
                return r.sort_index()
        else:
            return self._df[self._df.volume >= minVolume]
