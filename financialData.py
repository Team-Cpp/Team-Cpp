import pandas as pd
import math as m
import numpy as np
from datetime import datetime
import yfinance as yf
from pandas_datareader import data as pdr

# yf.pdr_override() #pandas datareader format

###### SET PARAMETERS ######
today = datetime.today().strftime('%Y-%m-%d')
start_dt = "2015-01-01"
period = "10y" #1d, 5d, 1mo,3mo,6mo,1y,2y,5y,10,ytd,max
interval = "1d" #1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

###### DEFINE STOCKS ######
labels = 'goog aapl'
tickers = yf.Tickers(labels)


stocks = yf.download(tickers, period = period, interval = interval)
stocks
