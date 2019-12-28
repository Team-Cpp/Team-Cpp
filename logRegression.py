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
import matplotlib.pyplot as plt


def SMA(period, data):
    sma = data.rolling(window=period).mean()
    return sma


# combined.head(25)
# len(combined["Prices"][combined["Prices"].isnull()])
# combined["20dSMA"][combined["20dSMA"].isnull()]

# test = combined["Prices"]
# tes = test.dropna()


priceDF = dataFun.dataHub(url="https://datahub.io/core/oil-prices/r/wti-daily.csv", import_new_data = False)
oilDF = dataFun.oilProduction()
df = dataFun.combineFrames(priceDF,oilDF)
# dataFun.show_more(df, 200)
df = df[np.isfinite(df['Prices'])]
df = df.sort_values(by =['Date'])

df["20dSMA"] = SMA(20, df["Prices"])
df["10dSMA"] = SMA(10, df["Prices"])
df["5dSMA"] = SMA(5, df["Prices"])
df["50dSMA"] = SMA(50, df["Prices"])
df["200dSMA"] = SMA(200, df["Prices"])


def bollinger(data, window = 20):

    mid_band = SMA(window, data)
    std_dev = data.rolling(window = window).std()
    up_band = mid_band + 2*std_dev
    low_band = mid_band - 2*std_dev
    return low_band, up_band


df["boll_lo"] = bollinger(df['Prices'])[0]
df["boll_hi"] = bollinger(df['Prices'])[1]

df["boll_hi"] = pd.to_numeric(df["boll_hi"])
df["boll_lo"] = pd.to_numeric(df["boll_lo"])
df["20dSMA"] = pd.to_numeric(df["20dSMA"])
df["10dSMA"] = pd.to_numeric(df["10dSMA"])
df["5dSMA"] = pd.to_numeric(df["5dSMA"])
df["50dSMA"] = pd.to_numeric(df["50dSMA"])
df["200dSMA"] = pd.to_numeric(df["200dSMA"])

df = df[np.isfinite(df['200dSMA'])]

def ema(data, window = 20):
    
    exp = data.ewm(span = window, adjust=False).mean()
    return exp

# def ppsr(df):
#     """Calculate Pivot Points, Supports and Resistances for given data
    
#     :param df: pandas.DataFrame
#     :return: pandas.DataFrame
#     """
#     PP = df["Prices"] #pd.Series((df['High'] + df['Low'] + df['Close']) 
#     R1 = pd.Series(2 * PP - df['Low'])
#     S1 = pd.Series(2 * PP - df['High'])
#     R2 = pd.Series(PP + df['High'] - df['Low'])
#     S2 = pd.Series(PP - df['High'] + df['Low'])
#     R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))
#     S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))
#     psr = {'PP': PP, 'R1': R1, 'S1': S1,
#            'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3}
#     PSR = pd.DataFrame(psr)
#     df = df.join(PSR)
#     return df

def momentum(df, n):
    """
    
    :param df: pandas.DataFrame 
    :param n: 
    :return: pandas.DataFrame
    """
    M = pd.Series(df['Prices'].diff(n), name='Momentum_' + str(n))
    df = df.join(M)
    return df

def relative_strength_index(df, n = 14):
    """Calculate Relative Strength Index(RSI) for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = df.Index.min()
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
            DoD = DoMove
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


dataFun.plot2axis(df["Date"], df["Prices"].astype( \
    float), df["Production of Crude Oil"], "Date", "Price (USD)", \
    'Production of Crude Oil (Thousand Barrels per Day)', lineax1=False, \
    lineax1y=df["20dSMA"], lineax1name="20d SMA", \
    fill_boll=True, bol_high=df["boll_hi"], \
    bol_low=df["boll_lo"])


dataFun.plot2axis(combined["Date"], combined["Prices"].astype(
    float), combined["Production of Crude Oil"], "Date", "Price (USD)",
    'Production of Crude Oil (Thousand Barrels per Day)')

# set style, empty figure and axes
plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)

# Get index values for the X axis for facebook DataFrame
test = df[df['Date'] > '2019-01-01']
x_axis = test['Date']

# Plot shaded 21 Day Bollinger Band for Facebook
ax.fill_between(x_axis, test['boll_hi'], test['boll_lo'], color='blue', alpha = 0.3)

# Plot Adjust Closing Price and Moving Averages
ax.plot(x_axis, test['Prices'], color='black', lw=2)
ax.plot(x_axis, test['20dSMA'], color='orange', lw=2)

# Set Title & Show the Image
ax.set_title('20 Day Bollinger Band For WTI Oil Price')
ax.set_xlabel('Date (Year/Month)')
ax.set_ylabel('Price(USD)')
ax.legend()
plt.show()
