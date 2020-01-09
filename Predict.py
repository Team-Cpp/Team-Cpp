import matplotlib.pyplot as plt
import data_functions as dataFun
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.externals import joblib
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
import sys
sys.path.insert(1, "/Users/qw19176/Documents/Courses/codingchallenge/")

priceDF = dataFun.dataHub(
    url="https://datahub.io/core/oil-prices/r/wti-daily.csv", import_new_data=True)
oilDF = dataFun.oilProduction()
df = dataFun.combineFrames(priceDF, oilDF)
# dataFun.show_more(df, 200)
df = df[np.isfinite(df['Prices'])]
#indices = np.arange(0, len(df))
#df['indx']
#df = pd.DataFrame(index = indices, data = df)
df = df.sort_values(by=['Date'])
df = df.reset_index().drop(["index"], axis=1)

today = dt.today().strftime('%Y-%m-%d')
# df2 = pd.DataFrame


print("Please enter the current WTI Oil Price")
newPrice = float(input())

print("Please enter the current WTI Oil Production Value")
newProd = float(input())
df = df.append({"Date": today, "Prices": newPrice,
                "Production of Crude Oil": newProd}, ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"])


def SMA(period, data):
    sma = data.rolling(window=period).mean()
    return sma


def ema(data, window=20):

    exp = data.ewm(span=window, adjust=False).mean()
    return exp


def bollinger(data, window=20):

    mid_band = SMA(window, data)
    std_dev = data.rolling(window=window).std()
    up_band = mid_band + 2*std_dev
    low_band = mid_band - 2*std_dev
    return low_band, up_band


def momentum(df, n):
    """
    :param df: pandas.DataFrame 
    :param n: 
    :return: pandas.DataFrame
    """
    M = pd.Series(df['Prices'].diff(n), name='Momentum_' + str(n))
    df = df.join(M)
    return df


def relative_strength_index(df, n=14):
    """Calculate Relative Strength Index(RSI) for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = df.index.min()
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        # UpMove = df.loc[i + 1, 'High'] - df.loc[i, 'High']
        # DoMove = df.loc[i, 'Low'] - df.loc[i + 1, 'Low']
        Move = df.loc[i, 'Prices'] - df.loc[i + 1, 'Prices']

        if Move > 0:
            UpD = Move
        else:
            UpD = 0
        UpI.append(UpD)
        if Move < 0:
            DoD = Move
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    df = df.join(RSI)
    return df


def rate_of_change(df, n):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    M = df['Prices'].diff(n - 1)
    N = df['Prices'].shift(n - 1)
    ROC = pd.Series(M / N, name='ROC_' + str(n))
    df = df.join(ROC)
    return df


def macd(df, n_fast, n_slow):
    """Calculate MACD, MACD Signal and MACD difference
    
    :param df: pandas.DataFrame
    :param n_fast: 
    :param n_slow: 
    :return: pandas.DataFrame
    """
    EMAfast = pd.Series(df['Prices'].ewm(
        span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df['Prices'].ewm(
        span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' +
                     str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(
    ), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' +
                         str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df


df["20dSMA"] = SMA(20, df["Prices"])
df["200dSMA"] = SMA(20, df["Prices"])
df["boll_lo"] = bollinger(df['Prices'])[0]
df["boll_hi"] = bollinger(df['Prices'])[1]

df = momentum(df, 14)
df = macd(df, 12, 26)
df = rate_of_change(df, 14)
df = relative_strength_index(df)

df["boll_hi"] = pd.to_numeric(df["boll_hi"])
df["boll_lo"] = pd.to_numeric(df["boll_lo"])
df["20dSMA"] = pd.to_numeric(df["20dSMA"])
df["200dSMA"] = pd.to_numeric(df["200dSMA"])

df["bollAmplitude"] = df["boll_hi"] - df["boll_lo"]
df["distFromTopBoll"] = df["boll_hi"] - df["Prices"]
df["distFromLowBoll"] = df["boll_lo"] - df["Prices"]
df["20d200dDist"] = np.abs(df["20dSMA"] - df["200dSMA"])

filename = 'finalized_model.sav'
model = joblib.load(filename)
#result = model.score(X_test, Y_test)
#print(result)
x_test = df.tail(1)
x_test = x_test.drop(["200dSMA", "Date",
             "Prices", "boll_lo", "boll_hi", "MACDsign_12_26"], axis=1)

pred = model.predict(x_test)
proba = model.predict_proba(x_test)

if pred[0] == 0:
    print("the price of oil is not likely to increase tomorrow with probability {:.1%}".format(proba[0][0]))
else:
    print("the price of oil is likely to increase tomorrow with probability {:.1%}".format(
        proba[0][1]))


