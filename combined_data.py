import os.path
import time
import csv
import shutil
import urllib.error
import urllib.request
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

def show_more(df, lines):
    with pd.option_context("display.max_rows", lines):
        display(df)
# yf.pdr_override() #pandas datareader format

""" YAHOO FINANCE SECTION START """
###### SET PARAMETERS ######
today = dt.today().strftime('%Y-%m-%d')
start_dt = "2015-01-01"
period = "5y" #1d, 5d, 1mo,3mo,6mo,1y,2y,5y,10,ytd,max
interval = "1d" #1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

###### DEFINE STOCKS ######
# Google, Apple, Tesla, Crude oil, S&P 500,Parsley Energy Inc.,Oasis Petroleum Inc., Gold, W&T Offshore Inc.,
# NASDAQ 100, Goodrich Petroleum Corporation
#stocks = 'BZ=F GOOG AAPL TSLA CL = F ^ GSPC PE OAS GC = F WTI NQ = F GDP'

stocks = "CL=F"  # CL=F ^GSPC OAS GDP ^DJI NQ=F"

# GBP/USD, BTC/GBP, USD/JPY, EUR/GBP, ETH/USD
# rates = 'GBPUSD=X CNY=X EURUSD=X'
tickers = yf.Ticker(stocks)
tickers.info
###### GET DATA #######
Stocks = yf.download(stocks, period=period, interval="1d")
Stocks.columns
Stocks
# Rates = yf.download(rates, period = period, interval = interval)
# show_more(Stocks, 300)

""" YAHOO FINANCE SECTION END """

###### DATAHUB URLS ######


#US Oil Production in 1000 barrels per day
url2 = 'https://www.eia.gov/dnav/pet/hist_xls/WCRFPUS2w.xls'
r2 = requests.get(url2)
data_xls = pd.read_excel(url2, 'Data 1', skiprows=2, comment='#')
data_xls.columns = ['Date', 'Production of Crude Oil']
Oil_Production = data_xls


""" DATAHUB SECTION START """

fname = 'Testdata.csv'
import_new_data = False

error = False  # you can use this flag to tell the programme to basically skip everything else and go to the end so you don't get a crash when there's a problem

#Allows the user to choose to use a previously downloaded datafile, or to download a new one. Mostly so I don't have to keep downloading the same datafile over and over again
if os.path.isfile(fname) == True:
    while True:
        user_selection = "Y"#input("Download new data (Y/N)? \n")
        if user_selection == 'Y':
            import_new_data = True
            break
        if user_selection == 'N':
            import_new_data == False
            break
        else:
            print("Input not recognised, please select either 'Y' or 'N'")
else:
    print("No local file detected, defaulting to downloading data")
    import_new_data = True

if import_new_data == False:
    print("Skipping download")
else:
    print("Downloading csv file...")

if import_new_data == True:
    url = 'https://datahub.io/core/oil-prices/r/wti-daily.csv'
    try:
        response = urllib.request.urlopen(url)
        print("Connection ok")
        data = response.read()      # a `bytes` object
        # a `str`; this step can't be used if data is binary
        text = data.decode('utf-8')
        with urllib.request.urlopen(url) as response, open(fname, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
            requrl = url
            print("Error, something went wrong with the download Error code:", e.code)
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

    frame = pd.DataFrame({'Dates': Dates_List, 'Prices': Prices_List})

    print("Dataframe created!")

print("Programme complete!")

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

newframe = frame[frame["Dates"] >= start_dt]
proddata = Oil_Production[Oil_Production["Date"] > start_dt]


newframe = newframe.drop([0])
newframe = newframe.rename(columns={"Dates": "Date"})
newframe["Date"] = pd.to_datetime(newframe["Date"])

test = proddata
test["Date"] = pd.to_datetime(test.Date, format='%d/%m/%Y')
test.set_index('Date').resample('B').ffill().reset_index()

proddata = proddata.set_index('Date').resample('B').ffill().reset_index()

combined = pd.merge(proddata, newframe, how='outer', on='Date')
combined.columns
nulls = combined[combined['Production of Crude Oil'].isnull()]
last_prod = combined['Production of Crude Oil'].last_valid_index()
last_prod_val = combined['Production of Crude Oil'].iloc[last_prod]

final_frame = combined.fillna(last_prod_val)




