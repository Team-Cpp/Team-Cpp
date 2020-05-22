import os
import sys

PATH = os.environ["DF_ROOT"]
sys.path.insert(1, PATH)

import warnings

warnings.filterwarnings("ignore")
import requests
import time
import copy
import warnings
import quandl
import pickle
import logging
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import itertools
import seaborn as sns

import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.externals import joblib

import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import plot_model
import multiprocessing



"""
Data Attributes
"""
dataAttr = {
    "wti": {
        "quandlCode": "FRED/DCOILWTICO",
        "yahooCode": "CL=F",
        "yahooPeriod": "1d",
        "dfName": "Prices",
    },
    # "oilProduction": {
    #     "quandlCode": "EIA/STEO_COPRPUS_M",
    #     "yahooCode": "",
    #     "yahooPeriod": "",
    #     "dfName": "OilProduction"
    # },
    "natGas": {
        "quandlCode": "EIA/NG_RNGWHHD_D",
        "yahooCode": "NG=F",
        "yahooPeriod": "1d",
        "dfName": "NatGasPrices",
    },
    # "brent": {
    #     "quandlCode": "FRED/DCOILBRENTEU",
    #     "yahooCode": "BZ=F",
    #     "yahooPeriod": "1d",
    #     "dfName": "BrentPrices",
    # },
    # "sp500": {
    #     "quandlCode": "",
    #     "yahooCode": "^GSPC",
    #     "yahooPeriod": "1d",
    #     "dfName": "SP500",
    # },
    "nasdaq": {
        "quandlCode": "",
        "yahooCode": "^IXIC",
        "yahooPeriod": "1d",
        "dfName": "Nasdaq",
    }
    # "dow": {
    #     "quandlCode": "",
    #     "yahooCode": "^DJI",
    #     "yahooPeriod": "1d",
    #     "dfName": "DowJones",
    # },
    # "gold": {
    #     "quandlCode": "",
    #     "yahooCode": "GC=F",
    #     "yahooPeriod": "1d",
    #     "dfName": "Gold",
    # },
    # "btc": {
    #     "quandlCode": "",
    #     "yahooCode": "BTC-USD",
    #     "yahooPeriod": "1d",
    #     "dfName": "Btc",
    # },
    # "bond": {
    #     "quandlCode": "",
    #     "yahooCode": "^TNX",
    #     "yahooPeriod": "1d",
    #     "dfName": "Bond10y",
    # },
}

QAPIKEY = "YpAydSEsKoSAfuQ9UKhu"
quandl.ApiConfig.api_key = QAPIKEY

# Cost parameters set by the task for running ship for one day
barrels = 750000
costPerDay = 30000
daysToPredict = 1

# Data split for training and testing.
trainDataDate = "2009-06-01"
testSplitDate = "2020-04-01"

# Parameters for the model.
params = {
    "batch_size": 16,  # 20<16<10, 25 was a bust
    "epochs": 200,
    "lr": 0.0010000,
    "time_steps": 50,
}

INPUT_PATH = PATH + "/LSTM_dev/inputs/"
OUTPUT_PATH = PATH + "/LSTM_dev/outputs/"
TIME_STEPS = params["time_steps"]
BATCH_SIZE = params["batch_size"]

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TZ"] = "Europe/London"
time.tzset()
stime = time.time()

modDate = dt.today().date()
dataDate = modDate - td(days=1)

dataFileName = "inputData.csv"
modFileName = "LSTM_model.h5"
histFileName = "LSTM_history.csv"
scalerFileName = "dataScaler.save"


train_cols = [
    "Prices",
    "OilProduction",
    "NatGasPrices",
    "Nasdaq",
    "daysAbove200dSMA",
    "MACD_12_26",
    "Momentum_14",
]

inp_size = len(train_cols)


# Creating directories
# check if directory already exists
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print("Directory created", OUTPUT_PATH)

else:
    print("Directory " + OUTPUT_PATH + " already exists.")

if not os.path.exists(INPUT_PATH):
    os.makedirs(INPUT_PATH)
    print("Directory created", INPUT_PATH)
else:
    print("Directory " + INPUT_PATH + " already exists.")


def yFinData(
    startDt, interval="1d", endDt=-1, stock="CL=F", onlyClose=True, name="Prices"
):
    today = dt.today().strftime("%Y-%m-%d")

    if endDt == -1:
        endDt = today

    stockInfo = yf.Ticker(stock).info

    ###### GET DATA #######
    Stocks = yf.download(stock, start=startDt, end=endDt, interval=interval)
    Stocks.reset_index(level=0, inplace=True)
    Stocks = Stocks.sort_values(by=["Date"])
    Stocks = Stocks.drop_duplicates(keep="first")

    if onlyClose:
        Stocks = Stocks.drop(["Open", "High", "Low", "Adj Close", "Volume"], axis=1)
        Stocks = Stocks.rename(columns={"Close": name})

    return Stocks, stockInfo


""" YAHOO FINANCE SECTION END """


# US Oil Production in 1000 barrels per day
def oilProduction(url="https://www.eia.gov/dnav/pet/hist_xls/WCRFPUS2w.xls"):
    # url = 'https://www.eia.gov/dnav/pet/hist_xls/WCRFPUS2w.xls'
    r2 = requests.get(url)
    data_xls = pd.read_excel(url, "Data 1", skiprows=2, comment="#")
    data_xls.columns = ["Date", "Production of Crude Oil"]
    # Oil_Production = data_xls
    return data_xls


def combineFrames(dfPrice, df2):
    start_dt = np.max([np.min(dfPrice["Date"]), np.min(df2["Date"])])

    newframe = dfPrice[dfPrice["Date"] >= start_dt]
    proddata = df2[df2["Date"] >= start_dt]

    newframe["Date"] = pd.to_datetime(newframe["Date"])

    proddata = proddata.set_index("Date").resample("B").ffill().reset_index()

    combined = pd.merge(proddata, newframe, how="outer", on="Date")
    # combined.columns

    nulls = combined[combined.isnull()]
    last_prod = combined["Production of Crude Oil"].last_valid_index()
    last_prod_val = combined["Production of Crude Oil"].iloc[last_prod]

    oil = combined.pop("Production of Crude Oil")
    oil = oil.fillna(last_prod_val)

    combined["Production of Crude Oil"] = oil
    return combined
    # combined["Prices"] = pd.to_numeric(combined["Prices"])


"""
########### CALCULATIONS #############
"""


def SMA(period, data):
    sma = data.rolling(window=period).mean()
    return sma


def ema(data, window=20):

    exp = data.ewm(span=window, adjust=False).mean()
    return exp


def bollinger(data, window=20):

    mid_band = SMA(window, data)
    std_dev = data.rolling(window=window).std()
    up_band = mid_band + 2 * std_dev
    low_band = mid_band - 2 * std_dev
    return low_band, up_band


def momentum(df, n):
    """
    :param df: pandas.DataFrame 
    :param n: 
    :return: pandas.DataFrame
    """
    M = pd.Series(df["Prices"].diff(n), name="Momentum_" + str(n))
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
        Move = df.loc[i, "Prices"] - df.loc[i + 1, "Prices"]

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
    RSI = pd.Series(PosDI / (PosDI + NegDI), name="RSI_" + str(n))
    df = df.join(RSI)
    return df


def rate_of_change(df, n):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    M = df["Prices"].diff(n - 1)
    N = df["Prices"].shift(n - 1)
    ROC = pd.Series(M / N, name="ROC_" + str(n))
    df = df.join(ROC)
    return df


def macd(df, n_fast, n_slow):
    """Calculate MACD, MACD Signal and MACD difference
    
    :param df: pandas.DataFrame
    :param n_fast: 
    :param n_slow: 
    :return: pandas.DataFrame
    """
    EMAfast = pd.Series(df["Prices"].ewm(span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df["Prices"].ewm(span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name="MACD_" + str(n_fast) + "_" + str(n_slow))
    MACDsign = pd.Series(
        MACD.ewm(span=9, min_periods=9).mean(),
        name="MACDsign_" + str(n_fast) + "_" + str(n_slow),
    )
    MACDdiff = pd.Series(
        MACD - MACDsign, name="MACDdiff_" + str(n_fast) + "_" + str(n_slow)
    )
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df


def profitNextDayObjective(df, days, costPerDay, barrels):

    day = int(days)
    incName = "increaseIn" + str(day) + "d"
    changeName = "change" + str(day) + "d"
    profitName = "profitIn" + str(day) + "d"
    obj = "objective" + str(day) + "d"

    df[incName] = np.nan
    df[profitName] = np.nan
    df[changeName] = np.nan
    df[obj] = np.nan

    for i in range(len(df) - days):
        ind = df.iloc[i].name
        change = df.iloc[i + days]["Prices"] - df.iloc[i]["Prices"]
        df[changeName].loc[ind] = change
        if change > 0:
            df[incName].loc[ind] = 1
        else:
            df[incName].loc[ind] = 0

        profit = (
            df.iloc[i + days]["Prices"] * barrels
            - df.iloc[i]["Prices"] * barrels
            - costPerDay * days
        )
        df[profitName].loc[ind] = profit
        if profit > 0:
            df[obj].loc[ind] = 1
        else:
            df[obj].loc[ind] = 0
    return df


def create_features(
    fd, features, indxCol="Date", label=None, shift=0, nonShiftFeatures=None
):
    feats = copy.copy(features)
    fd = fd.set_index(indxCol)
    X = fd[features]
    if shift > 0:

        newFeatures = copy.copy(features)
        for f in nonShiftFeatures:
            newFeatures.remove(f)

        X[newFeatures] = X[newFeatures].shift(shift)

    if label:
        y = fd[label]
        return X, y

    return X


def getData(key, quand=False, yahoo=False):

    if quand:
        data = quandl.get(key["quandlCode"])
        data.reset_index(level=0, inplace=True)
        data = data.rename(columns={"Value": key["dfName"]})

    if yahoo:
        stocks = key["yahooCode"]
        period = key["yahooPeriod"]

        if quand:
            yfStartDate = data["Date"].iloc[-1].strftime("%Y-%m-%d")
            Stocks, yfInfo = yFinData(yfStartDate, stock=stocks, name=key["dfName"])
            data = data.append(Stocks, ignore_index=True)

        else:
            yfStartDate = trainDataDate
            Stocks, yfInfo = yFinData(yfStartDate, stock=stocks, name=key["dfName"])
            data = Stocks

    if not (yahoo or quand):
        dates = pd.date_range(start=trainDataDate, end=dataDate, freq="D")
        data = pd.DataFrame(dates, columns=["Date"])

    data = data.sort_values(by=["Date"])
    return data


def build_timeseries(mat, y_col_index):
    """
    Converts ndarray into timeseries format and supervised data format. Takes first TIME_STEPS
    number of rows as input and sets the TIME_STEPS+1th data as corresponding output and so on.
    :param mat: ndarray which holds the dataset
    :param y_col_index: index of column which acts as output
    :return: returns two ndarrays-- input and output in format suitable to feed
    to LSTM.
    """
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    print("dim_0", dim_0)
    for i in range(dim_0):
        x[i] = mat[i : TIME_STEPS + i]
        y[i] = mat[TIME_STEPS + i, y_col_index]
    #         if i < 10:
    #           print(i,"-->", x[i,-1,:], y[i])
    print("length of time-series i/o", x.shape, y.shape)
    return x, y


def plot2axis(
    x,
    y1,
    x_name="Date",
    y_name="WTI Price (USD per Barrel)",
    color1="tab:black",
    axis2=False,
    y2=[],
    y2_name="",
    color2="tab:blue",
    lineax1=False,
    lineax1y=0,
    lineax1name="",
    colorax1y="tab:gray",
    lineax2=False,
    lineax2y=0,
    lineax2name="",
    colorax2y="tab:orange",
    lineax3=False,
    lineax3y=0,
    lineax3name="",
    colorax3y="tab:green",
    fill_boll=False,
    bol_low=[],
    bol_high=[],
    bol_name=""
):
    def flip(items, ncol):
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])

    y1Ampl = (np.max(y1) - np.min(y1)) * 0.1
    y2Ampl = (np.max(y2) - np.min(y2)) * 0.1

    y1limtop = np.max(y1) + y1Ampl * 3
    y1limbot = np.min(y1) - y1Ampl
    y2limtop = np.max(y2) + y2Ampl * 3
    y2limbot = np.min(y2) - y2Ampl
    datemin = np.datetime64(x.iloc[0], "W")
    datemax = np.datetime64(x.iloc[-1], "W") + np.timedelta64(1, "W")

    plt.style.use("fivethirtyeight")

    fig, ax = plt.subplots(figsize=(16, 10))
    # fig.figure(figsize=(20,15))

    color = color1
    ax.set_xlabel(x_name, size=28, labelpad=20)
    ax.set_ylabel(y_name, size=28, labelpad=20, color=color)
    (a,) = ax.plot(x, y1, lw=2, color=color, label=y_name)
    # ax.plot(x, df["20dSMA"][-90:], "--", lw = 2, color = "tab:gray", label="20 Day SMA")
    ax.tick_params(axis="y", labelsize=18, labelcolor=color)
    ax.tick_params(axis="x", labelsize=18, labelcolor=color)
    ax.set_ylim(y1limbot, y1limtop)
    ax.set_xlim(datemin, datemax)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
    ax.fmt_xdata = mdates.DateFormatter("%Y-%m-%d")
    if fill_boll is True:
        # ax.axhspan(bol_low, bol_high, facecolor='blue', label=bol_name, alpha=0.75)
        bol = ax.fill_between(
            x, bol_high, bol_low, color="tab:blue", alpha=0.25, label=bol_name
        )

    if axis2:
        ax2 = ax.twinx()
        color = color2

        # we already handled the x-label with ax
        ax2.set_ylabel(y2_name, size=28, labelpad=20, color=color)
        (b,) = ax2.plot(x, y2, color=color, lw=2, label=y2_name)
        ax2.tick_params(axis="y", labelsize=18, labelcolor=color)
        ax2.set_ylim(y2limbot, y2limtop)

        # ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))
        ax2.grid(None)

    if lineax1:
        (l1,) = ax.plot(
            x,
            lineax1y,
            "--",
            linewidth=2,
            alpha=0.75,
            color=colorax1y,
            label=lineax1name,
        )
        lines = [a, l1, bol]
        if axis2:
            lines = [a, b, l1, bol]

    elif lineax2:
        (l1,) = ax.plot(
            x,
            lineax1y,
            "--",
            linewidth=2,
            alpha=0.75,
            color=colorax1y,
            label=lineax1name,
        )
        (l2,) = ax.plot(
            x,
            lineax2y,
            "--",
            linewidth=2,
            alpha=0.75,
            color=colorax2y,
            label=lineax2name,
        )
        lines = [a, l1, bol, l2]
        if axis2:
            lines = [a, b, l1, bol, l2]

    elif lineax3:
        (l1,) = ax.plot(
            x,
            lineax1y,
            "--",
            linewidth=2,
            alpha=0.75,
            color=colorax1y,
            label=lineax1name,
        )
        (l2,) = ax.plot(
            x,
            lineax2y,
            "--",
            linewidth=2,
            alpha=0.75,
            color=colorax2y,
            label=lineax2name,
        )
        (l3,) = ax.plot(
            x,
            lineax3y,
            "--",
            linewidth=2,
            alpha=0.75,
            color=colorax3y,
            label=lineax3name,
        )
        lines = [a, l1, bol, l2, l3]
        if axis2:
            lines = [a, b, l1, bol, l2, l3]

    else:
        lines = [a, bol]
        if axis2:
            lines.append(b)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    handles = handles1
    for hand in handles2:
        handles.append(hand)
    labels = labels1
    for lab in labels2:
        labels.append(lab)

    ax.legend(
        flip(handles, 2),
        flip(labels, 2),
        loc=9,
        ncol=3,
        prop={"size": 18},
        frameon=False,
    )

    # ax.legend(lines, [line.get_label() for line in lines], loc='upper left', prop={'size': 18}, frameon = False)
    # plt.subplots_adjust(left=0.1, right=0.5)
    fig.autofmt_xdate()
    # fig.title(title, size = 36)
    fig.tight_layout()

    #cols = df.columns


def plot_data(df, variables=train_cols, days=60):
    variables.remove("Prices")
    days = days
    dates = df["Date"][-days:].astype("O")
    prices = df["Prices"][-days:]
    upBand = df["boll_hi"][-days:]
    lowBand = df["boll_lo"][-days:]
    bollName = "20d Bollinger Bands"
    for var in variables:
        print(var)

        plot2axis(
            x=pd.to_datetime(dates),
            y1=prices,
            y_name="WTI Price (USD per Barrel)",
            color1="black",
            lineax1=True,
            lineax1y=df["20dSMA"][-days:],
            lineax1name="20 Day SMA",
            axis2=True,
            y2=df[var][-days:],
            y2_name=var,
            fill_boll=True,
            bol_low=lowBand,
            bol_high=upBand,
            bol_name=bollName,
        )
        plt.title("WTI Price & " + var, size=36)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_PATH + "featurePlots/", var + ".png"),
            dpi=50,
            bbox_inches="tight",
        )


focus_features = [
    "Prices",
    "OilProduction",
    "NatGasPrices",
    # "BrentPrices",
    # "SP500",
    "Nasdaq",
    # "DowJones",
    # "Gold",
    # "Bond10y",
    "10dSMA",
    "20dSMA",
    "200dSMA",
    "daysAbove200dSMA",
    "distFromLowBoll",
    "20d200dDist",
    "daysAbove20dSMA",
    "distFromTopBoll",
    "RSI_14",
    "Momentum_14",
    "MACD_12_26",
    "month",
]
nonShiftFeat = ["Prices", "month"]


def plot_correlations(df, features=focus_features, nonShiftFeats=nonShiftFeat):
    sns.set(style="white")
    dfCorr = copy.copy(df)

    focus_cols = create_features(
        dfCorr, features=focus_features, shift=1, nonShiftFeatures=nonShiftFeat
    )
    corr = focus_cols.corr()

    heat_fig, (ax) = plt.subplots(1, 1, figsize=(11, 9))

    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    heat = sns.heatmap(
        corr,
        ax=(ax),
        mask=mask,
        vmax=0.5,
        square=True,
        linewidths=0.33,
        cbar_kws={"shrink": 0.33},
        cmap=cmap,
    )

    heat_fig.subplots_adjust(top=0.93)

    heat_fig.suptitle(
        "Correlation between main focus variables", fontsize=20, fontweight="bold"
    )

    plt.savefig(
        os.path.join(OUTPUT_PATH, "mainFeatureCorrelations.png"), dpi=50, format="png"
    )
    print(corr)
    return corr


def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    print (mat.shape)
    no_of_rows_drop = mat.shape[0] % batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat


def create_model(batch_size=BATCH_SIZE):
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    lstm_model.add(
        LSTM(
            128,
            batch_input_shape=(batch_size, TIME_STEPS, inp_size),
            dropout=0.0,
            recurrent_dropout=0.0,
            stateful=False,
            return_sequences=True,
            kernel_initializer="random_uniform",
        )
    )
    lstm_model.add(Dropout(0.4))
    lstm_model.add(LSTM(100, return_sequences=True, dropout=0.0))
    lstm_model.add(LSTM(100, return_sequences=False))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(Dense(20, activation="relu"))
    lstm_model.add(Dense(1, activation="sigmoid"))
    optimizer = optimizers.RMSprop(lr=params["lr"])
    # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    lstm_model.compile(loss="mean_squared_error", optimizer=optimizer)
    return lstm_model


def train_model(df, val_split=0.1):

    from keras import backend as K

    idx_col = df[train_cols].columns.get_loc("Prices")

    modeldf = copy.copy(df)
    modeldf = modeldf[modeldf["Date"] > trainDataDate]

    df_train, df_test = train_test_split(
        modeldf, train_size=(1 - val_split), test_size=val_split, shuffle=False
    )

    tooManyTests = len(df_test) % (BATCH_SIZE )#+ TIME_STEPS)
    df_test = df_test[tooManyTests:]

    tooManyTrains = len(df_train) % (BATCH_SIZE )#+ TIME_STEPS)
    df_train = df_train[tooManyTrains:]

    x = df_train.loc[:, train_cols].values
    sc = MinMaxScaler()
    x_train = sc.fit_transform(x)
    joblib.dump(sc, os.path.join(OUTPUT_PATH, scalerFileName))

    print(
        "Are any NaNs present in train matrix?",
        np.isnan(x_train).any(),
        # np.isnan(x_test).any(),
    )

    print("Are any NaNs present in train matrix?", np.isnan(x_train).any())
    x_t, y_t = build_timeseries(x_train, idx_col)
    x_t = trim_dataset(x_t, BATCH_SIZE)
    y_t = trim_dataset(y_t, BATCH_SIZE)
    print("Batch trimmed size", x_t.shape, y_t.shape)

    x_test = sc.transform(df_test.loc[:, train_cols].values)
    x_temp, y_temp = build_timeseries(x_test, idx_col)
    x_val = trim_dataset(x_temp, BATCH_SIZE)
    y_val = trim_dataset(y_temp, BATCH_SIZE)
    print("Validation size", x_val.shape, y_val.shape)

    print("Building model...")
    print("Train--Test size", len(df_train), len(df_test))
    model = create_model()

    es = EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=30, min_delta=0.0001
    )

    mcp = ModelCheckpoint(
        os.path.join(OUTPUT_PATH, "best_model.h5"),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        period=1,
    )

    # Not used here. But leaving it here as a reminder for future
    r_lr_plat = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=30,
        verbose=0,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
    )

    csv_logger = CSVLogger(
        os.path.join(
            OUTPUT_PATH, "training_log_" + time.ctime().replace(" ", "_") + ".log"
        ),
        append=True,
    )

    history = model.fit(
        x_t,
        y_t,
        epochs=params["epochs"],
        verbose=2,
        batch_size=BATCH_SIZE,
        shuffle=False,
        use_multiprocessing=True, 
        # workers=multiprocessing.cpu_count() - 1,
        # validation_split=val_split,
        validation_data=(
            trim_dataset(x_val, BATCH_SIZE),
            trim_dataset(y_val, BATCH_SIZE),
        ),
        callbacks=[es, mcp, csv_logger],
    )

    hist_df = pd.DataFrame(history.history)
    model.reset_states()
    weights = model.get_weights()
    print("saving model: ", modFileName)
    print("saving history: " + "")

    hist_df.to_csv(OUTPUT_PATH + "model_history.csv")
    model.save(os.path.join(OUTPUT_PATH, modFileName))
    # pickle.dump(model, open(, "wb"))

    return model, hist_df, weights


def test_train_loss(history):
    plt.style.use("fivethirtyeight")
    #plt.figure()
    fig, ax = plt.subplots(figsize=(7,5))
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    plt.savefig(os.path.join(OUTPUT_PATH, "train_vis_BS_" + ".png"))

def visualise_prediction(df, numDaysAgo, numDaysUntil,weights):
    pred_df = copy.copy(df)
    # df_test = df_test[train_cols]
    pred_df = pred_df[-(numDaysAgo + TIME_STEPS) : -numDaysUntil]
    preds = []
    for i in range(numDaysAgo - numDaysUntil + 1):
        pred_val = predict_new(weights, pred_df[i : (TIME_STEPS + i)])
        preds.append(pred_val[0][0])
        
    real_vals = pred_df["Prices"].values
    real_vals = real_vals[-numDaysAgo:-numDaysUntil]

    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(preds)
    plt.plot(real_vals)
    plt.title("Prediction vs Real Stock Price")
    plt.ylabel("Price")
    plt.xlabel("Days")
    plt.legend(["Prediction", "Real"], loc="upper left")

    plt.savefig(os.path.join(OUTPUT_PATH, "pred_vs_real_BS" + ".png"))

def predict_new(weights, df, days=1):

    sc = joblib.load(os.path.join(OUTPUT_PATH, scalerFileName))
    idx_col = df[train_cols].columns.get_loc("Prices")

    if len(df) < TIME_STEPS:
        raise Exception(
            "To make a prediciton for any following day, {} previous datapoints are required".format(
                TIME_STEPS
            )
        )
    new_model = create_model(batch_size=days)
    new_model.set_weights(weights)
    data_for_pred = copy.copy(df)
    data_for_pred = data_for_pred[train_cols][-TIME_STEPS:]
    x_for_pred = sc.transform(data_for_pred.loc[:, train_cols].values)

    pred = new_model.predict(
        x_for_pred.reshape(days, TIME_STEPS, inp_size), batch_size=days
    )
    pred_org = (pred * sc.data_range_[idx_col]) + sc.data_min_[idx_col]
    if days == 1:
        string = ("The price prediction for tomorrow is {:.3f}".format(pred_org[0][0]))
    else:
        string = (
            "The price predictions for the following {:.0f} days are {}".format(
                days, pred_org
            )
        )
    return string


def getLstmData():
    print("Querying data and building dataframe...")
    df = getData(dataAttr["wti"], quand=True, yahoo=True)

    # # Getting Oil production data and combining dataframes
    oilDF = oilProduction()
    df = combineFrames(df, oilDF)
    df = df[np.isfinite(df["Prices"])]
    df = df.reset_index().drop(["index"], axis=1)

    for i, attr in enumerate(dataAttr):
        if i == 0:
            continue

        else:
            q = False
            y = False

            if dataAttr[attr]["quandlCode"]:
                q = True

            if dataAttr[attr]["yahooCode"]:
                y = True

            newData = getData(dataAttr[attr], quand=q, yahoo=y)
            df = pd.merge(df, newData, on=["Date"], how="left")
            df[dataAttr[attr]["dfName"]] = df[dataAttr[attr]["dfName"]].interpolate(
                method="nearest"
            )

    if (df["Prices"] <= 0.1).any():
        for i, row in df.tail(100).iterrows():
            price = row["Prices"]
            if price == 0 or price < 0.01:
                key = dataAttr["wti"]
                yfStartDate = df["Date"][i]
                stocks = key["yahooCode"]
                Stocks, yfInfo = yFinData(yfStartDate, stock=stocks, name=key["dfName"])
                missingData = Stocks
                df["Prices"][i] = missingData["Prices"][0]

    smas = {"5dSMA": 5, "10dSMA": 10, "20dSMA": 20, "50dSMA": 50, "200dSMA": 200}

    # Calculating the technical indicators for price data
    df = df.reset_index().drop(["index"], axis=1)
    df = df.sort_values(by=["Date"])

    for sma in smas:
        df[sma] = SMA(smas[sma], df["Prices"])
        df[sma] = pd.to_numeric(df[sma])

    df["boll_lo"] = bollinger(df["Prices"])[0]
    df["boll_hi"] = bollinger(df["Prices"])[1]

    df = momentum(df, 14)
    df = macd(df, 12, 26)
    df = rate_of_change(df, 14)
    df = relative_strength_index(df)

    df["boll_hi"] = pd.to_numeric(df["boll_hi"])
    df["boll_lo"] = pd.to_numeric(df["boll_lo"])

    i = 0
    j = 0
    for sma in smas:
        title = "daysAbove" + sma
        df[title] = float("NaN")
        for price, val, pos in zip(df["Prices"], df[sma], range(len(df))):
            if price > val:
                j = 0
                i += 1
                df[title].iloc[pos] = i

            elif val > price:
                i = 0
                j -= 1
                df[title].iloc[pos] = j

            else:
                i = 0
                j = 0
                df[title].iloc[pos] = 0

    df["bollAmplitude"] = df["boll_hi"] - df["boll_lo"]
    df["distFromTopBoll"] = df["boll_hi"] - df["Prices"]
    df["distFromLowBoll"] = df["boll_lo"] - df["Prices"]
    df["20d200dDist"] = np.abs(df["20dSMA"] - df["200dSMA"])

    df = df[np.isfinite(df["200dSMA"])]
    df = df.rename(columns={"Production of Crude Oil": "OilProduction"})
    df = df.drop_duplicates("Date", keep="first")
    df = df[np.isfinite(df["Prices"])]
    df = df.reset_index().drop(["index"], axis=1)

    """
    Creating time series features from datetime index
    """

    df["dayofweek"] = df["Date"].dt.dayofweek
    df["quarter"] = df["Date"].dt.quarter
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["dayofyear"] = df["Date"].dt.dayofyear
    df["dayofmonth"] = df["Date"].dt.day
    df["weekofyear"] = df["Date"].dt.weekofyear
    df = df.sort_values(by="Date")
    df = df[df["Date"] > trainDataDate]
    df = df.reset_index().drop(["index"], axis=1)

    print("Saving dataframe to file ", dataFileName, "at ", INPUT_PATH)
    df.to_csv(INPUT_PATH + dataFileName)
    return df

if __name__ == 'main':
    while True:
        quit = False
        trained = False
        predicc = False
        correls = False
        trainvars = False
        traintestloss = False
        pricevs = False
        archi = False
        updateData = False
    
        while True:
            try:
                option = int(
                    input(
                        "Choose your destiny:\n 0: Quit \n 1: Download data \n 2: Train or load model \n 3: Predict price for tomorrow \n 4: Plot correlations of variables \n 5: Plot variables used for training \n 6: Plot model training/test loss \n 7: Plot predicted vs real price \n 8: Plot model architecture \n :"
                    )
                )
                break
            except (ValueError):
                print("Invalid input, please select one of the possible integers!")
    
        if option == 0:
            break
        elif option == 1:
            updateData = True
        elif option == 2:
            trained = True
        elif option == 3:
            predicc = True
        elif option == 4:
            correls = True
        elif option == 5:
            trainvars = True
        elif option == 6:
            traintestloss = True
        elif option == 7:
            pricevs = True
        elif option == 8:
            archi = True
        else:
            continue
    
        if updateData is False:
            try:
                df = pd.read_csv(INPUT_PATH + dataFileName)
                print("Loaded data file " + dataFileName + " ...")
                updateData = False
    
            except FileNotFoundError:
                print("Data file not found")
                updateData = True
    
        if updateData is True:
            df = getLstmData()
    
        if trained is True:
    
            choice = None
            print("\n 0: Load existing model \n 1: Train new model \n")
            while True:
                try:
                    choice = int(input(":"))
                    if choice != 0 and choice != 1:
                        raise (ValueError)
                    else:
                        break
    
                except (ValueError):
                    print("Choose an acceptable option!")
    
            if choice == 0:
                try:
    
                    model = load_model(os.path.join(OUTPUT_PATH, modFileName))
                    history = pd.read_csv(os.path.join(OUTPUT_PATH, histFileName))
                    sc = joblib.load(os.path.join(OUTPUT_PATH, scalerFileName))
                    weights = model.get_weights()
                    print("Loaded saved model...")
    
                except (FileNotFoundError, OSError):
                    print("Model not found")
                    choice = 1
    
            if choice == 1:
    
                model, history, weights = train_model(df)
                model_trained = True
            else:
                continue
    
        if predicc:
            predict_new(weights, df)
    
        if correls:
            plot_correlations(df)
    
        if trainvars:
            plot_data(df)
    
        if traintestloss:
            test_train_loss(history)
    
        if pricevs:
            numDaysAgo = 0
            numDaysUntil = 0
            print(
                "Please enter the timeframe for which you wish to compare prediction and real price"
            )
            while True:
                try:
                    numDaysAgo = int(
                        input("Enter how many days ago you want the prediction to start: ")
                    )
                    if numDaysAgo < 0:
                        raise (ValueError)
                    else:
                        break
                except (ValueError):
                    print("Please enter a positive integer for number of days!")
    
            while True:
                try:
                    numDaysUntil = int(
                        input(
                            "Enter until how many days ago you want the prediction to end: "
                        )
                    )
                    numDaysUntil += 1
                    if numDaysUntil > numDaysAgo:
                        raise (ValueError)
                    else:
                        break
    
                except (ValueError):
                    print("This must be later than the prediction start date!")
    
            visualise_prediction(df, numDaysAgo, numDaysUntil)
    
        if archi:
            plot_model(model.model, to_file=os.path.join(OUTPUT_PATH, "model.png"))
    
        if quit:
            break
    
        else:
            continue
