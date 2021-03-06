#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:08:49 2020
@author: nj18237
"""
import warnings
warnings.filterwarnings("ignore")
import os
from os import path

import sys

basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, ".."))
print (filepath)
try:
    sys.path.insert(1,os.environ['DF_ROOT'])
except:
    sys.path.insert(0,filepath)
    sys.path.insert(1,filepath+"/commonFunctions")
    sys.path.insert(2,basepath+"/commonFunctions")


#sys.path.insert(1,os.environ['DF_ROOT'])

import pandas as pd
print(pd.__file__)
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import string

import commonFunctions.dataFunctions as dataFun
import commonFunctions.Covid_19_Data_Scrapers as Covid

import csv

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
#from IPython.display import Image, display
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

import click

#from tkinter import *

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
    
    X = df[['Oil Production', '20 day Simple Moving Average', 'Momentum_14', 'Moving average convergence/divergence 12 26', 'Moving average convergence/divergence diff 12 26', 'ROC_14', 'RSI_14', 'Bollinger Amplitude', 'Distance from high Bollinger band', 'Distance from low Bollinger band', '20 and 200 day SMA difference','Day of the year','Day of the month','Week of the year',"COVID-19 Cases (ECDC data)"]]
    if shift > 0:
        tiems = X[['Day of the year','Day of the month','Week of the year']]
        X = X[['Oil Production', '20 day Simple Moving Average', 'Momentum_14', 'Moving average convergence/divergence 12 26', 'Moving average convergence/divergence diff 12 26','ROC_14', 'RSI_14', 'Bollinger Amplitude', 'Distance from high Bollinger band', 'Distance from low Bollinger band', '20 and 200 day SMA difference',"COVID-19 Cases (ECDC data)"]].shift(shift)
        X = X.merge(tiems, how='inner', left_index=True, right_index=True)

    if label:
        y = df[label]
        return X, y
    return X

#This Function had been retired - its functionality is duplicated in the profitability test, so for now it just calls that function to avoid breaking the GUI until that can eb updated
def Intellidock_Test_Accuracy(df,window,barrels,costPerDay):
    string1,string2,string3,string4 = Intellidock_Test_Profitability(df,window,barrels,costPerDay)
    return(string1,string2,string3,string4)

def Intellidock_Train(df):
    
    #print(df[len(df.index)-1])
    preset_early_stopping_rounds = 1   
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
    
def Intellidock_Predict_Next_Day(df,model,X_test, y_test,barrels,costPerDay):
    WTI_Prediction_tomorrow = model.predict(X_test)[0]
    #WTI_Prediction_tomorrow_confidence = model.predict_proba(X_test)
        
    delta_predicted = (WTI_Prediction_tomorrow - df['Prices'][len(df.index)-1])*barrels - costPerDay
        
    output_string = ""
    print("\n \n \n")    
    if (delta_predicted > 0):
        output_string = "Predicted price increase is sufficient to warrant waiting until tomorrow."
        print("\033[92m" + output_string + "\033[0m")
          
    else:  
        output_string = "Predicted price change does not warrant waiting until tomorrow."
        print("\033[91m" + output_string + "\033[0m")
    row_index = 0
    with open('CL_Limits.csv') as CL_lims:       
        CL_Lims_Reader = csv.reader(CL_lims)
        for row in CL_Lims_Reader:
            if row_index == 0:
                CL_Low = float(row[0])
            elif row_index == 1:
                CL_High = float(row[0])
            row_index += 1
            
    
    print("Details:")
    print("Price Today: {:.2f}".format(df['Prices'][len(df.index)-1]))
    print("Price Predicted Tomorrow: {:.2f} (90% Confidence Interval of {:.2f} to {:.2f})".format(WTI_Prediction_tomorrow, WTI_Prediction_tomorrow + CL_Low,WTI_Prediction_tomorrow + CL_High))
    print("Anticipated price change: {:.2f}".format(WTI_Prediction_tomorrow - df['Prices'][len(df.index)-1]))
    print("Assumed Costs = {:.2f}".format(costPerDay))
    print("Barrels Contained on Ship: = {:.0f}".format(barrels))
    print("Gain if Sold Today: {:.2f}".format(barrels*df['Prices'][len(df.index)-1]))
    print("Anticipated Gain Change (including operating costs): {:.2f}".format(barrels*(WTI_Prediction_tomorrow-df['Prices'][len(df.index)-1])-costPerDay))
    print("Gain if Sold Tomorrow Minus Operating Costs: {:.2f}".format(barrels*WTI_Prediction_tomorrow-costPerDay))
    print("\n \n \n") 
    
    string1 = ' '.join(["Price Today: {:.2f}".format(df['Prices'][len(df.index)-1])])
    string2 = ' '.join(["Price Predicted Tomorrow: {:.2f}".format(WTI_Prediction_tomorrow)])
    string3 = ' '.join(["Anticipated price change: {:.2f}".format(WTI_Prediction_tomorrow - df['Prices'][len(df.index)-1])])
    string4 = ' '.join(["Assumed Costs = {:.2f}".format(costPerDay)])
    string5 = ' '.join(["Barrels Contained on Ship: = {:.2f}".format(barrels)])
    string6 = ' '.join(["Gain if Sold Today: {:.2f}".format(barrels*df['Prices'][len(df.index)-1])])
    string7 = ' '.join(["Anticipated Gain Change (including operating costs): {:.2f}".format(barrels*(WTI_Prediction_tomorrow-df['Prices'][len(df.index)-1])-costPerDay)])
    string8 = ' '.join(["Gain if Sold Tomorrow Minus Operating Costs: {:.2f}".format(barrels*WTI_Prediction_tomorrow-costPerDay)])
    
    print(string1)
    print(string2)
    print(string3)
    print(string4)
    print(string5)
    print(string6)
    print(string7)
    print(string8)
    
    
    return(output_string,string1,string2,string3,string4,string5,string6,string7,string8)

def Intellidock_Display_Feature_Importance(df,model,X_test, y_test):
    fig, ax = plt.subplots(figsize=(15,10))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()
    return fig

def Intellidock_Get_Data():

    trainDataDate = '2018-01-01'
    Covid_Data_Start = '2019-12-31'
    
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
    
    ECDC_Covid_Data_Series = Covid.ECDC_Covid_Data()
    ECDC_Covid_Data_Frame = ECDC_Covid_Data_Series.to_frame()
    #print("Series:")
    #print(ECDC_Covid_Data_Series)
    #print(ECDC_Covid_Data_Frame.columns())
    #ECDC_Covid_Data_Frame.set_index('Date')

    print("Complete!")
    print("Constructing Data Frame...")
    df = pd.merge(newdf, brentData, on=['Date'], how ="left")
    df = df[df["Date"] > trainDataDate]
    df = df.rename(columns={"Production of Crude Oil": "Oil Production"})
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
    
    
    #df=pd.concat([df,ECDC_Covid_Data.to_frame()])
    #df.rename(columns = {"cases":"ECDC Covid Total Cases"})
    
    df = df[np.isfinite(df['200 day Simple Moving Average'])]
    #df.isna().sum()
        
    df = df.set_index('Date').join(ECDC_Covid_Data_Frame)
    
    df = df.reset_index()
    
    #The remainder of this should just be formatting, making sure everything is easily readable.
    df = df.drop_duplicates("Date",keep="first")
    #df
    
    df = df[df["Date"] > trainDataDate]
    df = df.reset_index().drop(["index"], axis = 1)
    df = df.fillna(0)
    #print(df.columns)
    
    #df = df.rename({'cases':'COVID-19 Cases (ECDC data)'}) #this line should work but got broken for no apparent reason when I tried to push it to github
    
    df.columns = ['Date', 'Prices', 'Oil Production', 'NatGasPrices', 'BrentPrices',  '20 day Simple Moving Average', '10 day Simple Moving Average',  '5 day Simple Moving Average', '50 day Simple Moving Average',  '200 day Simple Moving Average', 'Bollinger band low',  'Bollinger band high', 'Momentum_14',  'Moving average convergence/divergence 12 26',  'Moving average convergence/divergence sign 12 26',  'Moving average convergence/divergence diff 12 26', 'ROC_14', 'RSI_14',  'Bollinger Amplitude', 'Distance from high Bollinger band',  'Distance from low Bollinger band', '20 and 200 day SMA difference',  'COVID-19 Cases (ECDC data)']
    
    print(df.columns)    
    
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

def Intellidock_Test_Profitability(df,barrels,costPerDay):
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
                  
    print ("Testing complete, accuracy percentage = {:.1%} using data from {} to {}".format(df['Correct Prediction?'].sum()/(len(df.index)-preset_early_stopping_rounds), df["Date"][0], df["Date"][len(df.index)-1]))
    
    standard_deviation = (sum_of_squares/n - df["Deviation"].mean()**2)**0.5
    
    print("Mean error as standard deviation from truth value: {:.3f}".format(standard_deviation))
    
    print("\n\n Profitability testing completed, estimated profit PER DAY relative to immediate sale:")
    print ("{:.2f}".format(df["Relative Profit"].sum()/len(df.index)))
    
    fraction_included = 0.0
    Truth_CL_upper = 0.0
    Truth_CL_lower = 0.0
    Fraction_enclosed = 0.0
    Number_enclosed = 0
    
    
    for i in range(0,len(df.index)):
        if df['Deviation'][i]<1.645*standard_deviation and df['Deviation'][i]>-1.645*standard_deviation:
            Number_enclosed += 1
    fraction_included = Number_enclosed/len(df.index)
        
    
    print("Theoretical 90%CL: {:.1%} +/- {:.1%}".format(df["Deviation"].mean(), 1.645*standard_deviation))
    
    print("Actual amount enclosed in this interval: {:.2f}".format(fraction_included))
    
    median_deviation = df['Deviation'].sort_values().median()
    Truth_CL_upper = 0.0
    Truth_CL_lower = 0.0
    Fraction_enclosed = 0.0
    Number_enclosed = 0
   
    while Fraction_enclosed <0.45:
        Truth_CL_upper += 0.1
        for i in range(preset_early_stopping_rounds,len(df.index)):
            if df['Deviation'][i]>=median_deviation and df['Deviation'][i] < median_deviation + Truth_CL_upper:
                Number_enclosed += 1
            Fraction_enclosed = Number_enclosed/(len(df.index)-preset_early_stopping_rounds)
            
        Number_enclosed = 0
        
        print("{:.2f}, upper_CL = {:.2f}, fraction enclosed = {:.2f}, number enclosed = {:.2f}".format(median_deviation, Truth_CL_upper, Fraction_enclosed, Number_enclosed))
        if Truth_CL_upper>40:
            break
    
    ll = 0.45+(0.45-Fraction_enclosed)
    Fraction_enclosed = 0.0
    Number_enclosed = 0.0
    
    
    while Fraction_enclosed <ll:
        Truth_CL_lower -= 0.1
        for i in range(preset_early_stopping_rounds,len(df.index)):
            if df['Deviation'][i]<median_deviation and df['Deviation'][i] > median_deviation + Truth_CL_lower:
                Number_enclosed +=1
            Fraction_enclosed = Number_enclosed/(len(df.index)-preset_early_stopping_rounds)
            
        Number_enclosed = 0
        
        print("lower_CL = {:.2f},  {:.2f},  {:.2f}".format(Truth_CL_lower, Fraction_enclosed, Number_enclosed))
       
    Truth_CL_lower = median_deviation + Truth_CL_lower
        
    Truth_CL_upper = median_deviation + Truth_CL_upper
    
    
    #print("CL = ",0,"+",Truth_CL_upper, "-" , Truth_CL_lower)
    print("Confidence range = {:.2f} to {:.2f}".format(Truth_CL_lower, Truth_CL_upper))
    
    
    string1 = ' '.join(["Testing complete, accuracy percentage = {:.2f} using data from {} to {}.".format(df['Correct Prediction?'].sum()/(len(df.index)-preset_early_stopping_rounds), (df["Date"][0]), df["Date"][len(df.index)-1])])
    string2 = ' '.join(["Mean error as standard deviation from truth value: {:.2f}".format(standard_deviation)])
    string3 = ' '.join(["\n\n Profitability testing completed, estimated profit PER DAY relative to immediate sale:"])
    string4 = ' '.join(["{:.2f}".format(df["Relative Profit"].sum()/len(df.index))])
   
    string5 = ' '.join(["Theoretical gaussian 90%CL: {:.2f} +/- {:.2f}".format(df["Deviation"].mean(), 1.645*standard_deviation)])
    string6 = ' '.join(["Actual amount enclosed in this interval: {:.2f}".format(fraction_included)])
    string7 = ' '.join(["Empirical 90% Confidence Limit Range = + {:.2f} - {:.2f} relative to predicted price".format(Truth_CL_upper, -Truth_CL_lower)])

    string8 = ' '.join(["90% Confidence range = {:.2f} to {:.2f}".format(Truth_CL_lower, Truth_CL_upper)])
    
    with open('CL_Limits.csv','w',newline = '') as file_CL_Limits:
            writer = csv.writer(file_CL_Limits)
    
            writer.writerow([Truth_CL_lower])
            
            writer.writerow([Truth_CL_upper])
    
    print(string1)
    print(string2)
    print(string3)
    print(string4)
    print(string5)
    print(string6)
    print(string7)
    print(string8)

    plt.figure()

    plt.hist(df['Deviation'],bins = 100, range = [-25,25])
    #plt.suptitle('Histogram of Deviation from the Truth Price')
    plt.axvspan(Truth_CL_lower,Truth_CL_upper,alpha = 0.5,color='yellow')
    plt.xlabel('Deviation (Dollars)')
    plt.ylabel('Frequency')
    plt.savefig('Deviation_Histogram.png')
    plt.show()

    return(string1,string2,string3,string4,string7,string8)

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

#Click Commands

@click.command("BDT_Predictor")
@click.option("--predict/--no-predict", "predict", help="Make price predictions", default=False)
@click.option("--testProfit/--no-testProfit", "testProfit", help="Test profitability of using the model from 01/01/2018", default=False)

def run(predict, testProfit):
    rcParams.update({'figure.autolayout':True})
    barrels = 750000
    costPerDay = 30000
    days = 1
    option = -1
    df = Intellidock_Get_Data()
    if predict is True:
        df,model,X_test,y_test = Intellidock_Train(df)
        os1,os2,os3,os4,os5,os6,os7,os8,os9 = Intellidock_Predict_Next_Day(df,model,X_test,y_test,barrels,costPerDay)

        with open('BDT_Predicted_Data.csv','w',newline = '') as file:
            writer = csv.writer(file)
    
            writer.writerow([os1])
    
            writer.writerow([os2])
    
            writer.writerow([os3])
    
            writer.writerow([os4])
    
            writer.writerow([os5])
    
            writer.writerow([os6])
    
            writer.writerow([os7])
    
            writer.writerow([os8])
    
            writer.writerow([os9])
            
            fig = Intellidock_Display_Feature_Importance(df,model,X_test,y_test)
            
            fig.savefig('Feature_Importance.png',bbox_inces='tight')
        
    if testProfit is True:
        os1,os2,os3,os4,os7,os8 = Intellidock_Test_Profitability(df, barrels, costPerDay)
        
        with open('BDT_Profitability_Test.csv','w',newline = '') as file:
            writer = csv.writer(file)
    
            writer.writerow([os1])
    
            writer.writerow([os2])
    
            writer.writerow([os3])
    
            writer.writerow([os4])
    
            writer.writerow([os7])
    
            writer.writerow([os8])

    return

if __name__ == "__main__":
    run()
#---------------------------------------------------
#Debugging section - lets the code be run locally to check for bugs before pushing to git/VM

'''
barrels = 750000
costPerDay = 30000

df = Intellidock_Get_Data()

os1,os2,os3,os4,os7,os8 = Intellidock_Test_Profitability(df, barrels, costPerDay)

df,model,X_test,y_test = Intellidock_Train(df)
os1,os2,os3,os4,os5,os6,os7,os8,os9 = Intellidock_Predict_Next_Day(df,model,X_test,y_test,barrels,costPerDay)
'''

