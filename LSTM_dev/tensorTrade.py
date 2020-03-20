import os
import sys
import warnings
import numpy
from tensortrade.environments import TradingEnvironment
import ccxt
# from tensortrade.exchanges.live import CCXTExchange
import pandas as pd
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import PPO2

coinbase = ccxt.coinbasepro()
exchange = CCXTExchange(exchange=coinbase, base_instrument='EUR')

environment = TradingEnvironment(exchange=exchange,
                                 action_scheme=action_scheme,
                                 reward_scheme=reward_scheme,
                                 feature_pipeline=feature_pipeline)

def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter(action='ignore', category=FutureWarning)
numpy.seterr(divide='ignore')

sys.path.append(os.path.dirname(os.path.abspath('')))


def get_historical_data(exchange_name, instrument, timeframe):
    path = "https://www.cryptodatadownload.com/cdd/"
    filename = "{}_{}USD_{}.csv".format(exchange, symbol, timeframe)
    df = pd.read_csv(path + filename, skiprows=[0])
    df = df[::-1]
    df = df.drop(["Symbol", "Volume USD"], axis=1)
    df = df.rename({"Volume {}".format(symbol): "Volume"}, axis=1)
    df.columns = [name.lower() for name in df.columns]
    df = df.set_index("date")
    df.columns = [instrument + ":" + name for name in df.columns]
    return df


coinbase_btc = get_historical_data("Coinbase", "BTC", "d")
coinbase_eth = get_historical_data("Coinbase", "ETH", "d")
coinbase_ltc = get_historical_data("Coinbase", "LTC", "d")

exchange = Exchange("coinbase", service="simulated")(
    Stream("USD-BTC", list(df2['BTC:close'])),
    Stream("USD-ETH", list(df2['ETH:close'])),
    Stream("USD-LTC", list(df2['LTC:close']))
)
