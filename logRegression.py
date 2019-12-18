import os.path
import time
import csv
import shutil
import pandas as pd
import math as m
import numpy as np
from datetime import datetime as dt
import os
import yfinance as yf
from pandas_datareader import data as pdr
from IPython.display import display
import requests
yf.pdr_override()
sys.path.insert(1, "/Users/qw19176/Documents/Courses/codingchallenge/")
import data_functions as dataFun


def SMA(period, data):
    sma = data.rolling(window=period).mean()
    return sma


combined.head(25)
len(combined["Prices"][combined["Prices"].isnull()])
combined["20dSMA"][combined["20dSMA"].isnull()]

test = combined["Prices"]
tes = test.dropna()


priceDF = dataFun.dataHub(url="https://datahub.io/core/oil-prices/r/wti-daily.csv", import_new_data = False)
oilDF = dataFun.oilProduction()
df = dataFun.combineFrames(priceDF,oilDF)


df["20dSMA"] = SMA(20, df["Prices"])
df["10dSMA"] = SMA(10, df["Prices"])
df["5dSMA"] = SMA(5, df["Prices"])
df["50dSMA"] = SMA(50, df["Prices"])
df["200dSMA"] = SMA(200, df["Prices"])


def bolli_bois(data):

    mid_band = SMA(20, data)
    std_dev = np.std(mid_band)
    up_band = mid_band + 2*std_dev
    low_band = mid_band - 2*std_dev
    return low_band, up_band


combined["bollinger_low"] = bolli_bois(combined["20dSMA"])[0]
combined["bollinger_high"] = bolli_bois(combined["20dSMA"])[1]


show_more(combined, 300)

dataFun.plot2axis(combined["Date"], combined["Prices"].astype(
    float), combined["Production of Crude Oil"], "Date", "Price (USD)",
    'Production of Crude Oil (Thousand Barrels per Day)', lineax1=True,
    lineax1y=combined["20dSMA"], lineax1name="20d SMA",
    fill_boll=True, bol_high=combined["bollinger_high"],
    bol_low=combined["bollinger_low"])


dataFun.plot2axis(combined["Date"], combined["Prices"].astype(
    float), combined["Production of Crude Oil"], "Date", "Price (USD)",
    'Production of Crude Oil (Thousand Barrels per Day)')
