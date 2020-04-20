#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:08:49 2020

@author: nj18237
"""

###ToDo List: 
#Impliment an "error" as a deviation from the mean value and hence a confidence limit
#Done!

#Run a test to see if it makes a profit/loss overall and see by how much
#Done!

#Display how much each parameter actually impacts the result
#Partial...

#Separate the training from the prediction
#Done!

#"Serverless"

#Find a way to host the model stuff online somewhere
#
#
#import os
#import os.path
#abspath = os.path.abspath(__file__)
#dname = os.path.dirname(abspath)
#os.chdir(dname)

import os
import sys
#sys.path.insert(1,os.environ['DF_ROOT'])

import pandas as pd
print(pd.__file__)
import requests
import numpy as np
import matplotlib.pyplot as plt

import commonFunctions.dataFunctions as dataFun

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image, display
from sklearn import tree

from datetime import datetime as dt
import quandl
import yfinance as yf
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree

from tkinter import *


##--------------------------------------------------------------------
#Start of Function Definitions112

def show_more(df, lines):
    with pd.option_context("display.max_rows", lines):
        display(df)

def create_features(df, label=None, shift = 0):
    """
    Creates time series features from datetime index
    """
    df['Day of the week'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Day of the year'] = df['Date'].dt.dayofyear
    df['Day of the month'] = df['Date'].dt.day
    df['Week of the year'] = df['Date'].dt.weekofyear
    df = df.set_index('Date')
    #X = df[['OilProduction', 'NatGasPrices', 'BrentPrices', '20dSMA', 'Momentum_14', 'MACD_12_26', 'MACDdiff_12_26', 'ROC_14', 'RSI_14', 'bollAmplitude', 'distFromTopBoll', 'distFromLowBoll', '20d200dDist','dayofyear','dayofmonth','weekofyear']]

    X = df[['OilProduction', '20 day Simple Moving Average', 'Momentum_14', 'Moving average convergence/divergence 12 26', 'Moving average convergence/divergence diff 12 26', 'ROC_14', 'RSI_14', 'Bollinger Amplitude', 'Distance from high Bollinger band', 'Distance from low Bollinger band', '20 and 200 day SMA difference','Day of the year','Day of the month','Week of the year']]
    if shift > 0:
        tiems = X[['Day of the year','Day of the month','Week of the year']]
        #X = X[['OilProduction', 'NatGasPrices', 'BrentPrices', '20dSMA', 'Momentum_14', 'MACD_12_26', 'MACDdiff_12_26','ROC_14', 'RSI_14', 'bollAmplitude', 'distFromTopBoll', 'distFromLowBoll', '20d200dDist']].shift(shift)
        X = X[['OilProduction', '20 day Simple Moving Average', 'Momentum_14', 'Moving average convergence/divergence 12 26', 'Moving average convergence/divergence diff 12 26','ROC_14', 'RSI_14', 'Bollinger Amplitude', 'Distance from high Bollinger band', 'Distance from low Bollinger band', '20 and 200 day SMA difference']].shift(shift)
        X = X.merge(tiems, how='inner', left_index=True, right_index=True)

    if label:
        y = df[label]
        return X, y
    return X

#This Function had been retired - its functionality is duplicated in the profitability test, so for now it just calls that function to avoid breaking the GUI until that can eb updated
def Intellidock_Test_Accuracy(df,window,barrels,costPerDay):
    Intellidock_Test_Profitability(df,window,barrels,costPerDay)
    """
    df['WTI_Prediction_iterative'] = pd.Series(np.zeros(len(df.index)))
    df['WTI_Prediction_iterative_delta'] = pd.Series(np.zeros(len(df.index)))
    df['Prices_iterative_delta'] = pd.Series(np.zeros(len(df.index)))
    df['Correct Prediction?'] = pd.Series(np.zeros(len(df.index)))
    df['Deviation'] = pd.Series(np.zeros(len(df.index)))
    preset_early_stopping_rounds = 100
    sum_of_squares = 0
    n = len(df.index)-preset_early_stopping_rounds
    
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
        
            
        df["Deviation"][i] = abs(df['WTI_Prediction_iterative'][i]-df['Prices'][i])
        sum_of_squares += df["Deviation"][i]
        
    print ("Testing complete, accuracy percentage = ",df['Correct Prediction?'].sum()/(len(df.index)-preset_early_stopping_rounds),"using data from", df["Date"][0], "to", df["Date"][len(df.index)-1],".")
    
    standard_deviation = (sum_of_squares/n - df["Deviation"].mean()**2)**0.5
    
    print("Mean error as absolute deviation from truth value: ", standard_deviation)
    
    string1 = "Testing complete, accuracy percentage = ",df['Correct Prediction?'].sum()/(len(df.index)-preset_early_stopping_rounds),"using data from", df["Date"][0], "to", df["Date"][len(df.index)-1],"."
    string2 = "Mean error as absolute deviation from truth value: ", standard_deviation
    lbl = Label(window, text=string1,font = ('Arial',30))
    lbl.grid(column=0, row=0)  
    lbl = Label(window, text=string2,font = ('Arial',30))
    lbl.grid(column=0, row=1)  
    """
    return()

def Intellidock_Train(df):
    
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
    print("\n Training Complete!")
    return df, model, X_test, y_test
    
def Intellidock_Predict_Next_Day(df,model,X_test, y_test,window,barrels,costPerDay):
    WTI_Prediction_tomorrow = model.predict(X_test)[0]
    #WTI_Prediction_tomorrow_confidence = model.predict_proba(X_test)
        
    delta_predicted = (WTI_Prediction_tomorrow - df['Prices'][len(df.index)-1])*barrels - costPerDay
        
    output_string = ""
    print("\n \n \n")    
    if (delta_predicted > 0):
        output_string = "Predicted price increase is sufficient to warrant waiting until tomorrow."
        print("\033[92m" + output_string + "\033[0m")
        
        lbl = Label(window, text= output_string,font = ('Arial',30))
        lbl.grid(column=0, row=0)    
    else:  
        output_string = "Predicted price change does not warrant waiting until tomorrow."
        print("\033[91m" + output_string + "\033[0m")
        
        lbl = Label(window, text= output_string,font = ('Arial',30))
        lbl.grid(column=0, row=0)    
    
    print("Details:")
    print("Price Today: ",df['Prices'][len(df.index)-1])
    print("Price Predicted Tomorrow: ",WTI_Prediction_tomorrow)
    print("Anticipated price change:", WTI_Prediction_tomorrow - df['Prices'][len(df.index)-1])
    print("Assumed Costs = ",costPerDay)
    print("Barrels Contained on Ship: = ", barrels)
    print("Gain if Sold Today:", barrels*df['Prices'][len(df.index)-1])
    print("Anticipated Gain Change (including operating costs): ", barrels*(WTI_Prediction_tomorrow-df['Prices'][len(df.index)-1])-costPerDay)
    print("Gain if Sold Tomorrow Minus Operating Costs:", barrels*WTI_Prediction_tomorrow-costPerDay)
    print("\n \n \n") 
    
    string1 = "Price Today: ",df['Prices'][len(df.index)-1]
    string2 = "Price Predicted Tomorrow: ",WTI_Prediction_tomorrow
    string3 = "Anticipated price change:", WTI_Prediction_tomorrow - df['Prices'][len(df.index)-1]
    string4 = "Assumed Costs = ",costPerDay
    string5 = "Barrels Contained on Ship: = ", barrels
    string6 = "Gain if Sold Today:", barrels*df['Prices'][len(df.index)-1]
    string7 = "Anticipated Gain Change (including operating costs): ", barrels*(WTI_Prediction_tomorrow-df['Prices'][len(df.index)-1])-costPerDay
    string8 = "Gain if Sold Tomorrow Minus Operating Costs:", barrels*WTI_Prediction_tomorrow-costPerDay
    
    
    lbl = Label(window, text="Details:",font = ('Arial',30))
    lbl.grid(column=0, row=1)    
    lbl = Label(window, text=string1,font = ('Arial',30))
    lbl.grid(column=0, row=2)    
    lbl = Label(window, text=string2,font = ('Arial',30))
    lbl.grid(column=0, row=3)    
    lbl = Label(window, text=string3,font = ('Arial',30))
    lbl.grid(column=0, row=4)    
    lbl = Label(window, text=string4,font = ('Arial',30))
    lbl.grid(column=0, row=5)    
    lbl = Label(window, text=string5,font = ('Arial',30))
    lbl.grid(column=0, row=6)    
    lbl = Label(window, text=string6,font = ('Arial',30))
    lbl.grid(column=0, row=7)    
    lbl = Label(window, text=string7,font = ('Arial',30))
    lbl.grid(column=0, row=8)    
    lbl = Label(window, text=string8,font = ('Arial',30))
    lbl.grid(column=0, row=9)    
    
    return 

def Intellidock_Display_Feature_Importance(df,model,X_test, y_test):
    fig, ax = plt.subplots(figsize=(12,18))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()
    return

def Intellidock_Get_Data():

    trainDataDate = '2018-01-01'
    
    print('Running...')
    print("Acquiring Data...")
    QAPIKEY = 'YpAydSEsKoSAfuQ9UKhu'
    quandl.ApiConfig.api_key = QAPIKEY
    wtiData = quandl.get("FRED/DCOILWTICO")#Crude Oil Prices: Brent - Europe

    wtiData.reset_index(level=0, inplace=True)
    wtiData = wtiData.rename(columns={"Value": "Prices"})
    #wtiData.head()


    yfStartDate = wtiData['Date'].iloc[-1].strftime('%Y-%m-%d')
    stocks = "CL=F"
    period = "1d"

    Stocks, yfInfo = dataFun.yFinData(yfStartDate)

    wtiData = wtiData.append(Stocks, ignore_index =True)
    wtiData = wtiData.sort_values(by = ["Date"])


    oilDF = dataFun.oilProduction() #US Field Production of Crude Oil

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
    df["20 day Simple Moving Average"] = dataFun.SMA(20, df["Prices"])
    df["10 day Simple Moving Average"] = dataFun.SMA(10, df["Prices"])
    df["5 day Simple Moving Average"] = dataFun.SMA(5, df["Prices"])
    df["50 day Simple Moving Average"] = dataFun.SMA(50, df["Prices"])
    df["200 day Simple Moving Average"] = dataFun.SMA(200, df["Prices"])


    df["Bollinger band low"] = dataFun.bollinger(df['Prices'])[0]
    df["Bollinger band high"] = dataFun.bollinger(df['Prices'])[1]

    df = dataFun.momentum(df, 14)#difference between the price now and beforefor the last 14 days
    df = dataFun.macd(df, 12, 26)#Moving Average Convergence Divergence (a momentum indicator) based on the exponential moving average
    df = dataFun.rate_of_change(df, 14)
    df = dataFun.relative_strength_index(df)

    df["Bollinger band high"] = pd.to_numeric(df["Bollinger band high"])
    df["Bollinger band low"] = pd.to_numeric(df["Bollinger band low"])
    df["20 day Simple Moving Average"] = pd.to_numeric(df["20 day Simple Moving Average"])
    df["10 day Simple Moving Average"] = pd.to_numeric(df["10 day Simple Moving Average"])
    df["5 day Simple Moving Average"] = pd.to_numeric(df["5 day Simple Moving Average"])
    df["50 day Simple Moving Average"] = pd.to_numeric(df["50 day Simple Moving Average"])
    df["200 day Simple Moving Average"] = pd.to_numeric(df["200 day Simple Moving Average"])
    
    df["Bollinger Amplitude"] = df["Bollinger band high"] - df["Bollinger band low"]
    df["Distance from high Bollinger band"] = df["Bollinger band high"] - df["Prices"]
    df["Distance from low Bollinger band"] = df["Bollinger band low"] - df["Prices"]
    df["20 and 200 day SMA difference"] = np.abs(df["20 day Simple Moving Average"] - df["200 day Simple Moving Average"])
    #df
    
    df = df[np.isfinite(df['200 day Simple Moving Average'])]
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
    return df

def Intellidock_Test_Profitability(df,window,barrels,costPerDay):
    df['WTI_Prediction_iterative'] = pd.Series(np.zeros(len(df.index)))
    df['WTI_Prediction_iterative_delta'] = pd.Series(np.zeros(len(df.index)))
    df['Prices_iterative_delta'] = pd.Series(np.zeros(len(df.index)))
    df['Correct Prediction?'] = pd.Series(np.zeros(len(df.index)))
    df['Deviation'] = pd.Series(np.zeros(len(df.index)))
    df['Relative Profit'] = pd.Series(np.zeros(len(df.index)))
    df['Confidence Values'] = pd.Series(np.zeros(len(df.index)))
    preset_early_stopping_rounds = 100
    sum_of_squares = 0
    n = len(df.index)-preset_early_stopping_rounds
    
    run_start = 0
    run_active = False
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
        #df["Confidence Values"][i] = model.predict_proba(X_test)
        
        delta_predicted = (df['WTI_Prediction_iterative'][i]-df['Prices'][i-1])*barrels - costPerDay
        
        if(delta_predicted>0):
            df['WTI_Prediction_iterative_delta'][i]=1
            if(run_active == False):
                run_start = i
            run_active = True

        else:
            df['WTI_Prediction_iterative_delta'][i]=-1
            if(run_active == True):
                for j in range(run_start - 1, i-1):
                    df["Relative Profit"][j] = barrels*(df["Prices"][i-1]-df["Prices"][j])-costPerDay*(i-1-j)
            else:
                df["Relative Profit"][i-1] = 0
            run_active = False
                
        delta_truth = (df['Prices'][i]-df['Prices'][i-1])*barrels - costPerDay
                
        if(delta_truth>0):
            df['Prices_iterative_delta'][i]=1
        else:
            df['Prices_iterative_delta'][i]=-1
                        
        if(df['Prices_iterative_delta'][i]==df['WTI_Prediction_iterative_delta'][i]):
            df['Correct Prediction?'][i]=1
            
        df["Deviation"][i] = (df['WTI_Prediction_iterative'][i]-df['Prices'][i])
        sum_of_squares += (df["Deviation"][i])**2
                  
    print ("Testing complete, accuracy percentage = ",df['Correct Prediction?'].sum()/(len(df.index)-preset_early_stopping_rounds),"using data from", df["Date"][0], "to", df["Date"][len(df.index)-1],".")
    
    standard_deviation = (sum_of_squares/n - df["Deviation"].mean()**2)**0.5
    
    print("Mean error as standard deviation from truth value: ", standard_deviation)
    
    print("\n\n Profitability testing completed, estimated profit PER DAY relative to immediate sale:")
    print (df["Relative Profit"].sum()/len(df.index))
    
    fraction_included = 0.0
    
    for i in range(0,len(df.index)):
        if df['Deviation'][i]<1.645*standard_deviation and df['Deviation'][i]>-1.645*standard_deviation:
            fraction_included += 1
    fraction_included = fraction_included/len(df.index)
        
    
    print("Theoretical 90%CL:", df["Deviation"].mean()," +-", 1.645*standard_deviation)
    
    print("Actual amount enclosed in this interval:", fraction_included)
    
    Truth_CL_upper = 0.0
    Truth_CL_lower = 0.0
    Fraction_enclosed = 0.0
    Number_enclosed = 0
    
    while Fraction_enclosed <0.45:
        Truth_CL_upper += 0.01
        for i in range(preset_early_stopping_rounds,len(df.index)):
            if df['Deviation'][i]>df["Deviation"].mean() and df['Deviation'][i]<Truth_CL_upper:
                Number_enclosed +=1
            Fraction_enclosed = Number_enclosed/len(df.index)
        print("upper_CL = ",Truth_CL_upper, " Fraction enclosed =", Fraction_enclosed)
    
    
    Fraction_enclosed = 0.0
    Number_enclosed = 0.0
    
    while Fraction_enclosed <0.45:
        Truth_CL_lower -= 0.01
        for i in range(preset_early_stopping_rounds,len(df.index)):
            if df['Deviation'][i]<df["Deviation"].mean() and df['Deviation'][i] > Truth_CL_lower:
                Number_enclosed +=1
            Fraction_enclosed = Number_enclosed/len(df.index)
        print("lower_CL = ",Truth_CL_lower, " Fraction enclosed =", Fraction_enclosed, "number enclosed =", Number_enclosed)
    
    print("CL = ",df['Deviation'].mean(),"+",Truth_CL_upper, "-" , Truth_CL_lower)
    print("Confidence range = ",df['Deviation'].mean()+Truth_CL_lower,"to", df['Deviation'].mean()+Truth_CL_upper)
    
    
    string1 = "Testing complete, accuracy percentage = ",df['Correct Prediction?'].sum()/(len(df.index)-preset_early_stopping_rounds),"using data from", df["Date"][0], "to", df["Date"][len(df.index)-1],"."
    string2 = "Mean error as standard deviation from truth value: ", standard_deviation
    string3 = "\n\n Profitability testing completed, estimated profit PER DAY relative to immediate sale:"
    string4 = df["Relative Profit"].sum()/len(df.index)
   
    string5 = "Theoretical gaussian 90%CL:", df["Deviation"].mean(), "+-", 1.645*standard_deviation
    string6 = "Actual amount enclosed in this interval:", fraction_included
    string7 = "Truth CL = ",df['Deviation'].mean(),"+",Truth_CL_upper, "-" , -Truth_CL_lower
    string8 = "Confidence range = ",df['Deviation'].mean()+Truth_CL_lower,"to", df['Deviation'].mean()+Truth_CL_upper
    
    plt.hist(df['Deviation'],bins = 26)
    plt.savefig('Deviation_Histogram.png')
    plt.show()

    
    lbl = Label(window, text=string1,font = ('Arial',30))
    lbl.grid(column=0, row=0)    
    
    lbl = Label(window, text=string2,font = ('Arial',30))
    lbl.grid(column=0, row=1)    
    
    lbl = Label(window, text=string3,font = ('Arial',30))
    lbl.grid(column=0, row=2)    
    
    lbl = Label(window, text=string4,font = ('Arial',30))
    lbl.grid(column=0, row=3)
    
    lbl = Label(window, text=string5,font = ('Arial',30))
    lbl.grid(column=0, row=4)
    
    lbl = Label(window, text=string6,font = ('Arial',30))
    lbl.grid(column=0, row=5)    
    
    lbl = Label(window, text=string7,font = ('Arial',30))
    lbl.grid(column=0, row=6)    
    
    lbl = Label(window, text=string8,font = ('Arial',30))
    lbl.grid(column=0, row=7)    
    return

##-----------------------------------------------------------------------------    
#End of function Definitions

#Default Parameters  
    

#---------------------------------------------

