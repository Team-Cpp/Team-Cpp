import dataFunctionsDeps
from datetime import datetime as dt
import matplotlib.pyplot as plt
import requests
from pandas_datareader import data as pdr
import yfinance as yf
import numpy as np
import math as m
import pandas as pd
import urllib.request as ur
import urllib.error as ue
import shutil
import csv
import time
import os
import os.path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def plot2axis(x, y1, y2, x_name, y_name, y2_name, lineax1=False,
              lineax1y=0, lineax1name="lost", fill_boll=False,
              bol_low=[], bol_high=[], bol_name="easy"):
    import matplotlib.pyplot as plt

    plt.style.use("seaborn")
    fig, ax = plt.subplots()
    color = 'tab:green'
    ax.set_xlabel(x_name, size=16)
    ax.set_ylabel(y_name, size=16, color=color)
    ax.plot(x, y1, lw=1, color=color, label=y_name)
    ax.tick_params(axis='y', labelcolor=color)
    if lineax1 == True:
        ax.plot(x, lineax1y, label=lineax1name,
                linewidth=2, alpha=0.75, color="orange")
    if fill_boll == True:
        # ax.axhspan(bol_low, bol_high, facecolor='blue', label=bol_name, alpha=0.75)
        ax.fill_between(x, bol_high,
                        bol_low, color='blue', alpha=0.75)

    ax.legend(loc='upper left', prop={'size': 16})

    ax2 = ax.twinx()
    color = 'tab:blue'
    
    # we already handled the x-label with ax
    ax2.set_ylabel(y2_name, size=14, color=color)
    ax2.plot(x, y2, color=color, lw=1.5, label=y2_name)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.show()


# yf.pdr_override() #pandas datareader format

""" YAHOO FINANCE SECTION START """
###### SET PARAMETERS ######


def yFinData(startDt, interval = "1d", endDt = -1, stock="CL=F", onlyClose = 1,name="Prices"):
    today = dt.today().strftime('%Y-%m-%d')
    if endDt == -1:
        endDt = today
        
    #start_dt = "2015-01-01"
    #period = "5y" #1d, 5d, 1mo,3mo,6mo,1y,2y,5y,10,ytd,max
    # interval = "5d" #1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

    ###### DEFINE STOCKS ######
    # Google, Apple, Tesla, Crude oil, S&P 500,Parsley Energy Inc.,Oasis Petroleum Inc., Gold, W&T Offshore Inc.,
    # NASDAQ 100, Goodrich Petroleum Corporation
    # stocks = 'BZ=F GOOG AAPL TSLA CL = F ^ GSPC PE OAS GC = F WTI NQ = F GDP'

    #stocks = "CL=F"  # CL=F ^GSPC OAS GDP ^DJI NQ=F"

    # GBP/USD, BTC/GBP, USD/JPY, EUR/GBP, ETH/USD
    # rates = 'GBPUSD=X CNY=X EURUSD=X'
    stockInfo = yf.Ticker(stock).info
    
    ###### GET DATA #######
    Stocks = yf.download(stock, start = startDt, end = endDt, interval = interval)
    Stocks.reset_index(level=0, inplace=True)
    Stocks = Stocks.sort_values(by =["Date"])
    Stocks = Stocks.drop_duplicates(keep="first")
   
    if onlyClose == 1:
        Stocks = Stocks.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
        Stocks = Stocks.rename(columns={"Close": name})
    # Stocks.columns
    # len(Stocks)
    # Rates = yf.download(rates, period = period, interval = interval)
    # show_more(Stocks, 300)
    return Stocks, stockInfo

""" YAHOO FINANCE SECTION END """

###### DATAHUB URLS ######


#US Oil Production in 1000 barrels per day
def oilProduction(url='https://www.eia.gov/dnav/pet/hist_xls/WCRFPUS2w.xls'):
    #url = 'https://www.eia.gov/dnav/pet/hist_xls/WCRFPUS2w.xls'
    r2 = requests.get(url)
    data_xls = pd.read_excel(url, 'Data 1', skiprows=2, comment='#')
    data_xls.columns = ['Date', 'Production of Crude Oil']
    #Oil_Production = data_xls
    return data_xls


""" DATAHUB SECTION START """


def dataHub(url, import_new_data=True, fname="standard", error=False):

    # import_new_data = False
    today = dt.today().strftime('%Y-%m-%d')
    # error = False  # you can use this flag to tell the programme to basically skip everything else and go to the end so you don't get a crash when there's a problem
    if fname == 'standard':
        fname = 'WTI_oil_prices'+str(today)+'.csv'

    #Allows the user to choose to use a previously downloaded datafile, or to download a new one. Mostly so I don't have to keep downloading the same datafile over and over again
    # if os.path.isfile(fname) == True:
    #     while True:
    #         user_selection = "Y" #input("Download new data (Y/N)? \n")
    #         if user_selection == 'Y':
    #             import_new_data = True
    #             break
    #         if user_selection == 'N':
    #             import_new_data == False
    #             break
    #         else:
    #             print("Input not recognised, please select either 'Y' or 'N'")
    # else:
    #     print("No local file detected, defaulting to downloading data")
    #     import_new_data = True
    if os.path.isfile(fname):
        print("File already exists")
        import_new_data = False

    if import_new_data == False:
        print("Skipping download")
    else:
        print("Downloading csv file...")

    if import_new_data == True:
        try:
            response = ur.urlopen(url)
            print("Connection ok")
            data = response.read()      # a `bytes` object
            # a `str`; this step can't be used if data is binary
            text = data.decode('utf-8')
            with ur.urlopen(url) as response, open(fname, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except (ue.HTTPError, ue.URLError) as e:
                requrl = url
                print(
                    "Error, something went wrong with the download Error code:", e.code)
                print("If the error code is '404', try checking the url and/or your internet connection - 404 means that it can't find the address")
                error = True
        if error == False:
            print("Download Complete")

    if error == False:

        Dates_List = []
        Prices_List = []

        print("Reading downloaded csv file...")

        with open(fname) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                Dates_List.append(row[0])
                Prices_List.append(row[1])

            print("Reading complete, creating dataframe...")
        p = np.asarray(Prices_List[1:]).astype(float)
        d = np.asarray(Dates_List[1:]).astype('datetime64')
        frame = pd.DataFrame({"Date": d, "Prices": p})
        # frame = frame.drop([0])
        #frame["Prices"] = pd.to_numeric(frame["Prices"], errors='coerce')

        print("Dataframe created!")

    #print("Programme complete!")
    return frame

""" DATAHUB SECTION END """

""" WORLD BANK DATA SECTION START """
###### EXPLORE WORLD BANK DATA ########
# import world_bank_data as wb
# wb.get_topics()
# wb.get_sources()
# countries = wb.get_countries()
# inds = wb.get_indicators(topic = 7)
# show_more(inds,300)

###### GET THE DATA #######
# ffConsumption = wb.get_series(
#     "EG.USE.COMM.FO.ZS", date="2018", id_or_value='id', simplify_index=True)
# gdp = wb.get_series(
#     "NY.GDP.MKTP.CD", date="2018", id_or_value='id', simplify_index=True)
# gdpPerCapita = wb.get_series(
#     "NY.GDP.PCAP.CD", date="2018", id_or_value='id', simplify_index=True)
# gni = wb.get_series(
#     "NY.GNP.MKTP.CD", date="2018", id_or_value='id', simplify_index=True)
# costOfDamageDueToCarbonEmissions = wb.get_series(
#     "NY.ADJ.DCO2.CD", date="2018", id_or_value='id', simplify_index=True)

# ####### CREATE PANDAS DATAFRAME #######
# wbdf = countries[['region', 'name']].rename(columns={'name': 'country'}).loc[countries.region != 'Aggregates']
# wbdf["year"] =
# wbdf['ffConsumption'] = ffConsumption
# wbdf["gdp"] = gdp
# wbdf["gdpPerCapita"] = gdpPerCapita
# wbdf["gni"] = gni
# wbdf["costOfDamageDueToCarbonEmissions"] = costOfDamageDueToCarbonEmissions
# show_more(wbdf, 218)
""" WORLD BANK DATA SECTION END """
#Stocks
#frame.columns

""" COMBINING DATA FRAMES """


def combineFrames(dfPrice, df2):
    start_dt = np.max([np.min(dfPrice["Date"]), np.min(df2["Date"])])

    newframe = dfPrice[dfPrice["Date"] >= start_dt]
    proddata = df2[df2["Date"] >= start_dt]


# newframe = newframe.drop([0])
# newframe = newframe.rename(columns={"Dates": "Date"})
    newframe["Date"] = pd.to_datetime(newframe["Date"])

# test = proddata
# test["Date"] = pd.to_datetime(test.Date, format='%d/%m/%Y')
# test.set_index('Date').resample('B').ffill().reset_index()

    proddata = proddata.set_index('Date').resample('B').ffill().reset_index()

    combined = pd.merge(proddata, newframe, how='outer', on='Date')
    # combined.columns

    nulls = combined[combined.isnull()]
    last_prod = combined['Production of Crude Oil'].last_valid_index()
    last_prod_val = combined['Production of Crude Oil'].iloc[last_prod]

    oil = combined.pop("Production of Crude Oil")
    oil = oil.fillna(last_prod_val)

    combined["Production of Crude Oil"] = oil
    return combined
    # combined["Prices"] = pd.to_numeric(combined["Prices"])


# p = np.asarray(final_frame["Prices"])
# d = np.asarray(final_frame["Date"])

# from datetime import date, timedelta
# today = date.today()
# gap = (today - date(2015, 1, 1)).days
# from scipy.optimize import curve_fit
# from scipy.interpolate import make_interp_spline, BSpline

# x_new = [today - timedelta(days=x) for x in range(gap)]
# spl = make_interp_spline(T, power, k=3)  # type: BSpline
# power_smooth = spl(xnew)
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
