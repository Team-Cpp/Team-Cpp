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
sys.path.insert(1, "/Users/qw19176/Documents/Courses/codingchallenge/")
import data_functions as dataFun
import matplotlib.pyplot as plt


def SMA(period, data):
    sma = data.rolling(window=period).mean()
    return sma


def ema(data, window=20):

    exp = data.ewm(span=window, adjust=False).mean()
    return exp

def bollinger(data, window = 20):
    
    mid_band = SMA(window, data)
    std_dev = data.rolling(window = window).std()
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

# combined.head(25)
# len(combined["Prices"][combined["Prices"].isnull()])
# combined["20dSMA"][combined["20dSMA"].isnull()]

# test = combined["Prices"]
# tes = test.dropna()


priceDF = dataFun.dataHub(url="https://datahub.io/core/oil-prices/r/wti-daily.csv", import_new_data = True)
oilDF = dataFun.oilProduction()
df = dataFun.combineFrames(priceDF,oilDF)
# dataFun.show_more(df, 200)
df = df[np.isfinite(df['Prices'])]


#indices = np.arange(0, len(df))
#df['indx']
#df = pd.DataFrame(index = indices, data = df)
df = df.sort_values(by=['Date'])
df = df.reset_index().drop(["index"], axis = 1)

df["20dSMA"] = SMA(20, df["Prices"])
df["10dSMA"] = SMA(10, df["Prices"])
df["5dSMA"] = SMA(5, df["Prices"])
df["50dSMA"] = SMA(50, df["Prices"])
df["200dSMA"] = SMA(200, df["Prices"])


df["boll_lo"] = bollinger(df['Prices'])[0]
df["boll_hi"] = bollinger(df['Prices'])[1]

df = momentum(df, 14)
df = macd(df, 12, 26)
df = rate_of_change(df, 14)
df = relative_strength_index(df)

df["boll_hi"] = pd.to_numeric(df["boll_hi"])
df["boll_lo"] = pd.to_numeric(df["boll_lo"])
df["20dSMA"] = pd.to_numeric(df["20dSMA"])
# df["10dSMA"] = pd.to_numeric(df["10dSMA"])
# df["5dSMA"] = pd.to_numeric(df["5dSMA"])
# df["50dSMA"] = pd.to_numeric(df["50dSMA"])
df["200dSMA"] = pd.to_numeric(df["200dSMA"])

df["bollAmplitude"] = df["boll_hi"] - df["boll_lo"]
df["distFromTopBoll"] = df["boll_hi"] - df["Prices"]
df["distFromLowBoll"] = df["boll_lo"] - df["Prices"]
df["20d200dDist"] = np.abs(df["20dSMA"] - df["200dSMA"])

barrels = 750000
costPerDay = 30000
days = 1

def objective(df, days, costPerDay, barrels):
    
    day = int(days)
    incName = "increaseIn"+str(day)+"d"
    changeName = "change"+str(day)+"d"
    profitName = "profitIn"+str(day)+"d"
    obj = "objective"+str(day)+"d"
    
    df[incName] = np.nan
    df[profitName] = np.nan
    df[changeName] = np.nan
    df[obj] = np.nan
    
    for i in range(len(df)-days):
        ind = df.iloc[i].name
        change = df.iloc[i+days]["Prices"] - df.iloc[i]["Prices"]
        df[changeName].loc[ind] = change
        if change > 0:
            df[incName].loc[ind] = 1
        else:
            df[incName].loc[ind] = 0
            
        profit = df.iloc[i+days]["Prices"]*barrels - df.iloc[i]["Prices"]*barrels - costPerDay*days
        df[profitName].loc[ind] = profit
        if profit > 0:
            df[obj].loc[ind] = 1
        else:
            df[obj].loc[ind] = 0
    return df
          
dftest = objective(df, days, costPerDay, barrels)
len(dftest)
dftest = dftest[np.isfinite(dftest['200dSMA'])]
len(dftest)
obj = "objective"+str(int(days))+"d"
dftest = dftest[np.isfinite(dftest[obj])]
len(dftest)

Features = dftest.iloc[:, :-1]
Features = Features.drop(["profitIn1d", "change1d", "increaseIn1d", "Date",
                          "Prices", "boll_lo", "boll_hi", "MACDsign_12_26"], axis=1)

import seaborn as sns
correl = Features.corr()
sns.heatmap(correl, xticklabels=correl.columns, yticklabels=correl.columns)
plt.tight_layout()
plt.savefig("Figure 1: Correlation Matrix.png")
plt.clf()

Features = Features.drop(["10dSMA", "5dSMA", "50dSMA"], axis=1)
correl = Features.corr()
sns.heatmap(correl, xticklabels=correl.columns, yticklabels=correl.columns)
plt.tight_layout()
plt.savefig("Figure 1: Correlation Matrix.png")
plt.clf()

Features = Features.drop(["200dSMA"], axis=1)
correl = Features.corr()
sns.heatmap(correl, xticklabels=correl.columns, yticklabels=correl.columns)
plt.tight_layout()
plt.savefig("Figure 1: Correlation Matrix.png")
plt.clf()

Y = dftest.iloc[:, -1]
Y = Y.astype('int32')


def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)

def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))

def cost_function(self, theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost

def gradient(self, theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)

def fit(self, x, y, theta):
    opt_weights = fmin_tnc(func=cost_function, x0=theta,
                           fprime=gradient, args=(x, y.flatten()))
    return opt_weights[0]


Xt = np.c_[np.ones((Features.shape[0], 1)), Features]
y = Y[:, np.newaxis]
theta = np.zeros((Xt.shape[1], 1))


x_train, x_test, y_train, y_test = train_test_split(Features, Y, test_size=0.2, random_state=1)

lr = LogisticRegression()
lr.fit (x_train, y_train)

y_pred = lr.predict(x_test)
confusion_matrix(y_test, y_pred)
lr.predict_proba(x_test)
score = lr.score(x_test, y_test)
accuracy = accuracy_score(y_test, y_pred)
parameters = lr.coef_

print(score)


filename = 'finalized_model.sav'
joblib.dump(lr, filename)


# fd = pd.DataFrame({'x': x_test, 'y': y_test})
# fd = fd.sort_values(by='x')
# from scipy.special import expit
# sigmoid_function = expit(fd['x'] * lr.coef_[0][0] + lr.intercept_[0]).ravel()
# plt.plot(fd['x'], sigmoid_function)
# plt.scatter(fd['x'], fd['y'], c=fd['y'], cmap='rainbow', edgecolors='b')

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



dataFun.plot2axis(df["Date"], df["Prices"].astype( \
    float), df["Production of Crude Oil"], "Date", "Price (USD)", \
    'Production of Crude Oil (Thousand Barrels per Day)', lineax1=False, \
    lineax1y=df["20dSMA"], lineax1name="20d SMA", \
    fill_boll=True, bol_high=df["boll_hi"], \
    bol_low=df["boll_lo"])

dfrecent = df[df["Date"]> "2019-01-01"]
dataFun.plot2axis(dfrecent["Date"], dfrecent["Prices"].astype(
    float), dfrecent["Production of Crude Oil"], "Date", "Price (USD)",
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
