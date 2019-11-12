import pandas as pd
import math as m
import numpy as np
from datetime import datetime as dt
import os
import yfinance as yf
from pandas_datareader import data as pdr
from forex_python.converter import CurrencyRates, CurrencyCodes
from forex_python.bitcoin import BtcConverter
import world_bank_data as wb 

# yf.pdr_override() #pandas datareader format

###### SET PARAMETERS ######
today = dt.today().strftime('%Y-%m-%d')
start_dt = "2015-01-01"
period = "1y" #1d, 5d, 1mo,3mo,6mo,1y,2y,5y,10,ytd,max
interval = "1d" #1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

###### DEFINE STOCKS ######
stocks = 'GOOG AAPL TSLA CL=F OPECX ^GSPC PE OAS GC=F WTI'
rates = 'GBPUSD=X BTC-GBP'
tickers = yf.Tickers(stocks)

###### DEFINE CURRENCY RATES ###### 
c = CurrencyRates()
b = BtcConverter()
c.get_rate("USD", "GBP")


Stocks = yf.download(stocks, period = period, interval = interval)
Rates = yf.download(rates, period = period, interval = interval)

###### DATAHUB URLS ######

###### WORLD BANK DATA ########
wb.get_topics()
wb.get_sources()
wb.get_countries()
# wb.get_series()
