import pandas as pd
import math as m
import numpy as np
from datetime import datetime as dt
import os
import yfinance as yf
from pandas_datareader import data as pdr
import world_bank_data as wb 
from IPython.display import display

def show_more(df, lines):
    with pd.option_context("display.max_rows", lines):
        display(df)
# yf.pdr_override() #pandas datareader format

###### SET PARAMETERS ######
today = dt.today().strftime('%Y-%m-%d')
start_dt = "2015-01-01"
period = "1y" #1d, 5d, 1mo,3mo,6mo,1y,2y,5y,10,ytd,max
interval = "1d" #1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

###### DEFINE STOCKS ######
# Google, Apple, Tesla, Crude oil, S&P 500,Parsley Energy Inc.,Oasis Petroleum Inc., Gold, W&T Offshore Inc.,
# NASDAQ 100, Goodrich Petroleum Corporation
stocks = 'GOOG AAPL TSLA CL=F ^GSPC PE OAS GC=F WTI NQ=F GDP'

# GBP/USD, BTC/GBP, USD/JPY, EUR/GBP, ETH/USD
rates = 'GBPUSD=X BTC-GBP USDJPY=X EURGBP = X ETHUSD=X'
tickers = yf.Tickers(stocks)

###### GET DATA #######
Stocks = yf.download(stocks, period = period, interval = interval)
Rates = yf.download(rates, period = period, interval = interval)

###### DATAHUB URLS ######

###### EXPLORE WORLD BANK DATA ########
wb.get_topics()
wb.get_sources()
countries = wb.get_countries()
inds = wb.get_indicators(topic = 7)
show_more(inds,300)

###### GET THE DATA #######
ffConsumption = wb.get_series(
    "EG.USE.COMM.FO.ZS", date="2018", id_or_value='id', simplify_index=True)
gdp = wb.get_series(
    "NY.GDP.MKTP.CD", date="2018", id_or_value='id', simplify_index=True)
gdpPerCapita = wb.get_series(
    "NY.GDP.PCAP.CD", date="2018", id_or_value='id', simplify_index=True)
gni = wb.get_series(
    "NY.GNP.MKTP.CD", date="2018", id_or_value='id', simplify_index=True)
costOfDamageDueToCarbonEmissions = wb.get_series(
    "NY.ADJ.DCO2.CD", date="2018", id_or_value='id', simplify_index=True)

####### CREATE PANDAS DATAFRAME #######
wbdf = countries[['region', 'name']].rename(columns={'name': 'country'}).loc[countries.region != 'Aggregates']
wbdf["year"] =
wbdf['ffConsumption'] = ffConsumption
wbdf["gdp"] = gdp
wbdf["gdpPerCapita"] = gdpPerCapita
wbdf["gni"] = gni
wbdf["costOfDamageDueToCarbonEmissions"] = costOfDamageDueToCarbonEmissions
show_more(wbdf, 218)

