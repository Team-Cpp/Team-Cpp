#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:08:49 2020

@author: nj18237
"""

import os
import os.path
#abspath = os.path.abspath(__file__)
#dname = os.path.dirname(abspath)
#os.chdir(dname)
import rrsBdtDevDependencies
import dataFunctions as dataFun
from datetime import datetime as dt
import quandl
import yfinance as yf
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree

def show_more(df, lines):
    with pd.option_context("display.max_rows", lines):
        display(df)

def create_features(df, label=None, shift = 0):
    """
    Creates time series features from datetime index
    """
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['dayofmonth'] = df['Date'].dt.day
    df['weekofyear'] = df['Date'].dt.weekofyear
    df = df.set_index('Date')
    #X = df[['OilProduction', 'NatGasPrices', 'BrentPrices', '20dSMA', 'Momentum_14', 'MACD_12_26', 'MACDdiff_12_26', 'ROC_14', 'RSI_14', 'bollAmplitude', 'distFromTopBoll', 'distFromLowBoll', '20d200dDist','dayofyear','dayofmonth','weekofyear']]

    X = df[['OilProduction', '20dSMA', 'Momentum_14', 'MACD_12_26', 'MACDdiff_12_26', 'ROC_14', 'RSI_14', 'bollAmplitude', 'distFromTopBoll', 'distFromLowBoll', '20d200dDist','dayofyear','dayofmonth','weekofyear']]
    if shift > 0:
        tiems = X[['dayofyear','dayofmonth','weekofyear']]
        #X = X[['OilProduction', 'NatGasPrices', 'BrentPrices', '20dSMA', 'Momentum_14', 'MACD_12_26', 'MACDdiff_12_26','ROC_14', 'RSI_14', 'bollAmplitude', 'distFromTopBoll', 'distFromLowBoll', '20d200dDist']].shift(shift)
        X = X[['OilProduction', '20dSMA', 'Momentum_14', 'MACD_12_26', 'MACDdiff_12_26','ROC_14', 'RSI_14', 'bollAmplitude', 'distFromTopBoll', 'distFromLowBoll', '20d200dDist']].shift(shift)
        X = X.merge(tiems, how='inner', left_index=True, right_index=True)

    if label:
        y = df[label]
        return X, y
    return X

def Intellidock_Test_Accuracy(df):
    df['WTI_Prediction_iterative'] = pd.Series(np.zeros(len(df.index)))
    df['WTI_Prediction_iterative_delta'] = pd.Series(np.zeros(len(df.index)))
    df['Prices_iterative_delta'] = pd.Series(np.zeros(len(df.index)))
    df['Correct Prediction?'] = pd.Series(np.zeros(len(df.index)))
    preset_early_stopping_rounds = 100
    
    for i in range(preset_early_stopping_rounds,len(df.index)):
        print(i)
        #df_iter = df[0:][0:i+1]
        
        #testSplitDate = df_iter.Date[i].strftime("%Y-%m-%d")
        #testSplitDate = df_iter.Date[i]
        
        #iterative_df_train = df[0:][0:i+1][df[0:][0:i+1]["Date"] <= testSplitDate].copy()
        #iterative_df_test = df[0:][0:i+1][df[0:][0:i+1]["Date"] > testSplitDate].copy()
        
        
        #X_train, y_train = create_features(iterative_df_train, label='Prices')
        X_train, y_train = create_features(df[0:i], label='Prices',shift = 1)
        X_test, y_test = create_features(df[i::i], label='Prices',shift = 1)
        
        model = XGBRegressor(
                n_estimators=1000,
                #max_depth=8,
                #min_child_weight=300, 
                #colsample_bytree=0.8, 
                #subsample=0.8, 
                #eta=0.3,    
                #seed=42
                )
        
        model.fit(
                X_train, 
                y_train, 
                eval_metric="rmse", 
                eval_set=[(X_train, y_train), (X_test, y_test)], 
                verbose=False, 
                early_stopping_rounds = preset_early_stopping_rounds
                )
        df['WTI_Prediction_iterative'][i] = model.predict(X_test)[0]
        
        delta_predicted = (df['WTI_Prediction_iterative'][i]-df['Prices'][i-1])*barrels - costPerDay
        
        if(delta_predicted>0):
            df['WTI_Prediction_iterative_delta'][i]=1
        else:
            df['WTI_Prediction_iterative_delta'][i]=-1
                
        delta_truth = (df['Prices'][i]-df['Prices'][i-1])*barrels - costPerDay
                
        if(delta_truth>0):
            df['Prices_iterative_delta'][i]=1
        else:
            df['Prices_iterative_delta'][i]=-1
                        
        if(df['Prices_iterative_delta'][i]==df['WTI_Prediction_iterative_delta'][i]):
            df['Correct Prediction?'][i]=1
            
    print ("Testing complete, accuracy percentage = ",df['Correct Prediction?'].sum()/(len(df.index)-preset_early_stopping_rounds),"using data from", df["Date"][0], "to", df["Date"][len(df.index)-1],".")
    return

def Intellidock_Predict_Next_Day(df):
    
    #print(df[len(df.index)-1])
    preset_early_stopping_rounds = 100   
    X_train, y_train = create_features(df, label='Prices',shift = 0)
    X_test, y_test = create_features(df[len(df.index)-1::len(df.index)-1], label='Prices',shift = 0)
        
    model = XGBRegressor(
            n_estimators=1000,
            #max_depth=8,
            #min_child_weight=300, 
            #colsample_bytree=0.8, 
            #subsample=0.8, 
            #eta=0.3,    
            #seed=42
            )
        
    model.fit(
            X_train, 
            y_train, 
            eval_metric="rmse", 
            eval_set=[(X_train, y_train), (X_test, y_test)], 
            verbose=False, 
            early_stopping_rounds = preset_early_stopping_rounds
            )
    WTI_Prediction_tomorrow = model.predict(X_test)[0]
        
    delta_predicted = (WTI_Prediction_tomorrow - df['Prices'][len(df.index)-1])*barrels - costPerDay
        
    output_string = ""
    print("\n \n \n")    
    if (delta_predicted > 0):
        output_string = "Predicted price increase is sufficient to warrant waiting until tomorrow."
        print("\033[92m" + output_string + "\033[0m")
    else:  
        output_string = "Predicted price change does not warrant waiting until tomorrow."
        print("\033[91m" + output_string + "\033[0m")
    
    print("Details:")
    print("Price Today: ",df['Prices'][len(df.index)-1])
    print("Price Predicted Tomorrow: ",WTI_Prediction_tomorrow)
    print("Anticipated price change:", WTI_Prediction_tomorrow - df['Prices'][len(df.index)-1])
    print("Assumed Costs = ",costPerDay)
    print("Barrels Contained on Ship: = ", barrels)
    print("\n \n \n") 
    return

barrels = 750000
costPerDay = 30000
days = 1
trainDataDate = '2018-01-01'

print('Running...')
print("Acquiring Data...")
QAPIKEY = 'YpAydSEsKoSAfuQ9UKhu'
quandl.ApiConfig.api_key = QAPIKEY
wtiData = quandl.get("FRED/DCOILWTICO")

wtiData.reset_index(level=0, inplace=True)
wtiData = wtiData.rename(columns={"Value": "Prices"})
#wtiData.head()


yfStartDate = wtiData['Date'].iloc[-1].strftime('%Y-%m-%d')
stocks = "CL=F"
period = "1d"

Stocks, yfInfo = dataFun.yFinData(yfStartDate)

wtiData = wtiData.append(Stocks, ignore_index =True)
wtiData = wtiData.sort_values(by = ["Date"])


oilDF = dataFun.oilProduction()

df = dataFun.combineFrames(wtiData,oilDF)
df = df[np.isfinite(df['Prices'])]
df = df.reset_index().drop(["index"], axis = 1)

natGasData = quandl.get("EIA/NG_RNGWHHD_D")
natGasData.reset_index(level=0, inplace=True)
natGasData = natGasData.rename(columns={"Value": "NatGasPrices"})

yfStartDate = natGasData['Date'].iloc[-1].strftime('%Y-%m-%d')
stocks = "NG=F"
period = "1d"

NGStocks, yfInfo = dataFun.yFinData(yfStartDate,stock=stocks,name ="NatGasPrices")
natGasData = natGasData.append(NGStocks, ignore_index =True)
natGasData = natGasData.sort_values(by = ["Date"])


newdf = pd.merge(df, natGasData, on=['Date'], how ="left")
#newdf.head()

brentData = quandl.get("FRED/DCOILBRENTEU")
brentData.reset_index(level=0, inplace=True)
name = "BrentPrices"
brentData = brentData.rename(columns={"Value": name})

yfStartDate = brentData['Date'].iloc[-1].strftime('%Y-%m-%d')
stocks = "BZ=F"
period = "1d"
BStocks, yfInfo = dataFun.yFinData(yfStartDate,stock=stocks,name = name)
brentData = brentData.append(BStocks, ignore_index =True)
brentData = brentData.sort_values(by = ["Date"])
#brentData

print("Complete!")
print("Constructing Data Frame...")
df = pd.merge(newdf, brentData, on=['Date'], how ="left")
df = df[df["Date"] > trainDataDate]
df = df.rename(columns={"Production of Crude Oil": "OilProduction"})
#df.isna().sum()

df["BrentPrices"] = df["BrentPrices"].interpolate(method='nearest')
df["NatGasPrices"] = df["NatGasPrices"].interpolate(method='nearest')
#df.isna().sum()

df = df.reset_index().drop(["index"], axis = 1)
df["20dSMA"] = dataFun.SMA(20, df["Prices"])
df["10dSMA"] = dataFun.SMA(10, df["Prices"])
df["5dSMA"] = dataFun.SMA(5, df["Prices"])
df["50dSMA"] = dataFun.SMA(50, df["Prices"])
df["200dSMA"] = dataFun.SMA(200, df["Prices"])


df["boll_lo"] = dataFun.bollinger(df['Prices'])[0]
df["boll_hi"] = dataFun.bollinger(df['Prices'])[1]

df = dataFun.momentum(df, 14)
df = dataFun.macd(df, 12, 26)
df = dataFun.rate_of_change(df, 14)
df = dataFun.relative_strength_index(df)

df["boll_hi"] = pd.to_numeric(df["boll_hi"])
df["boll_lo"] = pd.to_numeric(df["boll_lo"])
df["20dSMA"] = pd.to_numeric(df["20dSMA"])
df["10dSMA"] = pd.to_numeric(df["10dSMA"])
df["5dSMA"] = pd.to_numeric(df["5dSMA"])
df["50dSMA"] = pd.to_numeric(df["50dSMA"])
df["200dSMA"] = pd.to_numeric(df["200dSMA"])

df["bollAmplitude"] = df["boll_hi"] - df["boll_lo"]
df["distFromTopBoll"] = df["boll_hi"] - df["Prices"]
df["distFromLowBoll"] = df["boll_lo"] - df["Prices"]
df["20d200dDist"] = np.abs(df["20dSMA"] - df["200dSMA"])
#df

df = df[np.isfinite(df['200dSMA'])]
#df.isna().sum()

df = df.drop_duplicates("Date",keep="first")
#df

df = df[df["Date"] > trainDataDate]
df = df.reset_index().drop(["index"], axis = 1)
#df



#testSplitDate = '2020-01-01'
#df_train = df[df["Date"] <= testSplitDate].copy()
#df_test = df[df["Date"] > testSplitDate].copy()

#X_train, y_train = create_features(df_train, label='Prices', shift =1)
#X_test, y_test = create_features(df_test, label='Prices', shift =1)
#X_train = X_train.iloc[1:]
#X_test = X_test.iloc[1:]
#y_train = y_train.iloc[1:]
#y_test = y_test.iloc[1:]

print("Complete!")

option = "0"

print("Please enter one of the following options: \n 1: Test the accuracy of the system \n 2: Predict if staying undocked is worthwhile \n q: quit")

while (option != "1" and option != "2" and option != "q"):

    option = input()
    print("Invalid input, please select either 1,2 or q")

if(option == "1"):
    print("Testing Prediction Accuracy")
    Intellidock_Test_Accuracy(df)
elif(option == "2"):
    Intellidock_Predict_Next_Day(df)
elif(option == "q"):
    print("Exiting")
    
#---------------------------------------------

