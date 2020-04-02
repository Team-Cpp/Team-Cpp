import os
import sys
PATH = os.environ['DF_ROOT']
sys.path.insert(1, PATH)

import quandl
import pickle
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
import matplotlib.pyplot as plt

import keras

QAPIKEY                     = "YpAydSEsKoSAfuQ9UKhu"
quandl.ApiConfig.api_key    = QAPIKEY
# Cost parameters set by the task for running ship for one day
barrels         = 750000
costPerDay      = 30000
daysToPredict   = 1
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'
os.environ['TZ']                    = 'Europe/London'
time.tzset()
stime                               = time.time()

modDate         = dt.today().date()
dataDate        = modDate - td(days=1)
updateData      = False
is_update_model = True

dataFileName    = "inputData_" + str(dataDate) + ".csv"
modFileName     = "LSTM_" + str(modDate) + ".sav"

# check if directory already exists
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print("Directory created", OUTPUT_PATH)
else:
    print("Directory "+OUTPUT_PATH+" already exists.")

if not os.path.exists(INPUT_PATH):
    os.makedirs(INPUT_PATH)
    print("Directory created", INPUT_PATH)
else:
    print("Directory "+INPUT_PATH+" already exists.")
    

def yFinData(startDt, interval = "1d", endDt = -1, stock="CL=F", onlyClose = True, name="Prices"):
    today = dt.today().strftime('%Y-%m-%d')
    if endDt == -1:
        endDt = today
        
    stockInfo = yf.Ticker(stock).info
    
    ###### GET DATA #######
    Stocks = yf.download(stock, start = startDt, end = endDt, interval = interval)
    Stocks.reset_index(level=0, inplace=True)
    Stocks = Stocks.sort_values(by =["Date"])
    Stocks = Stocks.drop_duplicates(keep="first")
   
    if onlyClose:
        Stocks = Stocks.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
        Stocks = Stocks.rename(columns={"Close": name})

    return Stocks, stockInfo

""" YAHOO FINANCE SECTION END """


#US Oil Production in 1000 barrels per day
def oilProduction(url='https://www.eia.gov/dnav/pet/hist_xls/WCRFPUS2w.xls'):
    #url = 'https://www.eia.gov/dnav/pet/hist_xls/WCRFPUS2w.xls'
    r2 = requests.get(url)
    data_xls = pd.read_excel(url, 'Data 1', skiprows=2, comment='#')
    data_xls.columns = ['Date', 'Production of Crude Oil']
    #Oil_Production = data_xls
    return data_xls


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


def profitNextDayObjective(df, days, costPerDay, barrels):

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

        profit = df.iloc[i+days]["Prices"]*barrels - \
            df.iloc[i]["Prices"]*barrels - costPerDay*days
        df[profitName].loc[ind] = profit
        if profit > 0:
            df[obj].loc[ind] = 1
        else:
            df[obj].loc[ind] = 0
    return df


def create_features(fd, features, indxCol = 'Date', label = None, shift = 0, nonShiftFeatures = None):
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

"""
Getting WTI price data 
"""
dataAttr = {
    "wti": {
        "quandlCode": "FRED/DCOILWTICO",
        "yahooCode": "CL=F",
        "yahooPeriod": "1d",
        "dfName": "Prices"
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
        "dfName": "NatGasPrices"
    },
    "brent": {
        "quandlCode": "FRED/DCOILBRENTEU",
        "yahooCode": "BZ=F",
        "yahooPeriod": "1d",
        "dfName": "BrentPrices"
    },
    "sp500": {
        "quandlCode": "",
        "yahooCode": "^GSPC",
        "yahooPeriod": "1d",
        "dfName": "SP500"
    },
    "nasdaq": {
        "quandlCode": "",
        "yahooCode": "^IXIC",
        "yahooPeriod": "1d",
        "dfName": "Nasdaq"
    }, 
    "dow": {
        "quandlCode": "",
        "yahooCode": "^DJI",
        "yahooPeriod": "1d",
        "dfName": "DowJones"
    }, 
    "gold": {
        "quandlCode": "",
        "yahooCode": "GC=F",
        "yahooPeriod": "1d",
        "dfName": "Gold"
    },
    "btc": {
        "quandlCode": "",
        "yahooCode": "BTC-USD",
        "yahooPeriod": "1d",
        "dfName": "Btc"
    }, 
    "bond": {
        "quandlCode": "",
        "yahooCode": "^TNX",
        "yahooPeriod": "1d",
        "dfName": "Bond10y"
    }
}

def getData(key, quand = False, yahoo = False):

    if quand:
        data = quandl.get(key['quandlCode'])
        data.reset_index(level = 0, inplace = True)
        data = data.rename(columns = {"Value": key["dfName"]})
    
    if yahoo:
        stocks          = key['yahooCode']
        period          = key['yahooPeriod']
        
        if quand:
            yfStartDate     = data['Date'].iloc[-1].strftime('%Y-%m-%d')
            Stocks, yfInfo  = yFinData(yfStartDate, stock = stocks, name = key['dfName'])
            data            = data.append(Stocks, ignore_index = True)

        else:
            yfStartDate     = trainDataDate
            Stocks, yfInfo  = yFinData(yfStartDate, stock = stocks, name = key['dfName'])
            data            = Stocks

    if not (yahoo or quand):
        dates   = pd.date_range(start = trainDataDate, end = dataDate, freq = 'D')
        data    = pd.DataFrame(dates, columns = ["Date"])

    data    = data.sort_values(by = ["Date"])
    return data


updateData = True
if updateData is False:
    df  = pd.read_csv(INPUT_PATH+dataFileName)
    print("Dataframe already exists, reading from file...")


else:
    print("Datafile not found, querying data and building dataframe...")
    df = getData(dataAttr['wti'], quand = True, yahoo = True)

    # # Getting Oil production data and combining dataframes
    oilDF   = oilProduction()
    df      = combineFrames(df,oilDF)
    df      = df[np.isfinite(df['Prices'])]
    df      = df.reset_index().drop(["index"], axis = 1)

    for i,attr in enumerate(dataAttr):
        if i == 0:
            continue

        else:
            q = False
            y = False

            if dataAttr[attr]["quandlCode"]:
                q = True

            if dataAttr[attr]["yahooCode"]:
                y = True

            newData = getData(dataAttr[attr], quand = q, yahoo = y)
            df      = pd.merge(df, newData, on = ['Date'], how = "left")
            df[dataAttr[attr]["dfName"]] = df[dataAttr[attr]["dfName"]].interpolate(method = 'nearest')

    smas = {
        "5dSMA": 5,
        "10dSMA": 10,
        "20dSMA": 20,
        "50dSMA": 50,
        "200dSMA": 200
    }

    # Calculating the technical indicators for price data
    df = df.reset_index().drop(["index"], axis = 1)
    df = df.sort_values(by = ["Date"])

    for sma in smas:
        df[sma] = SMA(smas[sma], df["Prices"])
        df[sma] = pd.to_numeric(df[sma])

    df["boll_lo"] = bollinger(df['Prices'])[0]
    df["boll_hi"] = bollinger(df['Prices'])[1]

    df = momentum(df, 14)
    df = macd(df, 12, 26)
    df = rate_of_change(df, 14)
    df = relative_strength_index(df)

    df["boll_hi"] = pd.to_numeric(df["boll_hi"])
    df["boll_lo"] = pd.to_numeric(df["boll_lo"])

    i = 0 
    j = 0
    for sma in smas:
        title = "daysAbove"+sma
        df[title] = float("NaN")
        for price,val,pos in zip(df["Prices"],df[sma],range(len(df))):
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


    df["bollAmplitude"]     = df["boll_hi"] - df["boll_lo"]
    df["distFromTopBoll"]   = df["boll_hi"] - df["Prices"]
    df["distFromLowBoll"]   = df["boll_lo"] - df["Prices"]
    df["20d200dDist"]       = np.abs(df["20dSMA"] - df["200dSMA"])

    df = df[np.isfinite(df['200dSMA'])]
    df = df.rename(columns={"Production of Crude Oil": "OilProduction"})
    df = df.drop_duplicates("Date",keep="first")
    df = df[np.isfinite(df['Prices'])]
    df = df.reset_index().drop(["index"], axis = 1)

    """
    Creating time series features from datetime index
    """

    df['dayofweek']     = df['Date'].dt.dayofweek
    df['quarter']       = df['Date'].dt.quarter
    df['month']         = df['Date'].dt.month
    df['year']          = df['Date'].dt.year
    df['dayofyear']     = df['Date'].dt.dayofyear
    df['dayofmonth']    = df['Date'].dt.day
    df['weekofyear']    = df['Date'].dt.weekofyear
    df                  = df.sort_values(by='Date')
    df = df[df["Date"] > trainDataDate]
    df = df.reset_index().drop(["index"], axis = 1)
    
    print("Saving dataframe to file ", dataFileName, "at ", INPUT_PATH)
    df.to_csv(INPUT_PATH+dataFileName)
    
# load the saved best model from above
saved_model = load_model(os.path.join(OUTPUT_PATH, 'best_model.h5')) # , "lstm_best_7-3-19_12AM",
print(saved_model)

train_cols = [
            "Prices",
            "NatGasPrices",
            "DowJones",
            "Gold"
        ]

to_pred = df["Prices"][-139:].values
to_pred = np.append(to_pred,50)
# trim_dataset(x_test_t, BATCH_SIZE)
to_pred = sc.transform(to_pred.reshape(-1,1))
x_new, y_new = build_timeseries(to_pred, target_idx)
x_newval, x_new_t = np.split(trim_dataset(x_new,BATCH_SIZE),2)
y_newval, y_new_t = np.split(trim_dataset(y_new,BATCH_SIZE),2)
to_pred.shape
# print("Test size", x_new_t.shape, y_new_t.shape, x_newval.shape, y_newval.shape)