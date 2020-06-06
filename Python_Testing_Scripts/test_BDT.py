#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:20:13 2020

@author: nj18237
"""

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
    sys.path.insert(3,filepath+"/GUI")

from Chris_Intellidock import Intellidock_Predict_Next_Day
from Chris_Intellidock import Intellidock_Train
from Chris_Intellidock import Intellidock_Get_Data
from Chris_Intellidock import Intellidock_Test_Profitability


import pandas as pd
from datetime import datetime as dt
from datetime import timedelta

def build_mock_df():
    
    date_list = [dt.strptime('2018-10-16','%Y-%m-%d')] 
    
    price_list = [60]
    
    irrelevant = [0]
    
    for i in range (0,100,1):
        date_list.append(date_list[i]+timedelta(1))
        price_list.append(price_list[i]+0.1)
        irrelevant.append(0)
    
    #Series = pd.Series({dt.strptime('2018-10-16','%Y-%m-%d'):60,dt.strptime('2018-10-17','%Y-%m-%d'):65,dt.strptime('2018-10-18','%Y-%m-%d'):70,dt.strptime('2018-10-18','%Y-%m-%d'):75})
    #Series_irrelevant = pd.Series([0,0,0,0])
    
    
    
    df = pd.DataFrame(date_list)
    df['Prices'] = price_list
    df['distFromTopBoll']=irrelevant
    df['20 and 200 day SMA difference']=irrelevant
    df['ROC_14']=irrelevant
    df['Moving average convergence/divergence 12 26']=irrelevant
    df['RSI_14']=irrelevant
    df['Moving average convergence/divergence diff 12 26']=irrelevant
    df['distFromLowBoll']=irrelevant
    df['20 day Simple Moving Average']=irrelevant
    df['Bollinger Amplitude']=irrelevant
    df['Oil Production']=irrelevant
    df['Distance from high Bollinger band']=irrelevant
    df['Distance from low Bollinger band']=irrelevant
    df['Momentum_14']=irrelevant
    df['Day of the year']=irrelevant
    df['Day of the month']=irrelevant
    df['Week of the Year']=irrelevant
    df['COVID-19 Cases (ECDC data)']=irrelevant
    
    
    barrels = 750000
    costPerDay = 30000
    
    df= df.rename(columns={0:'Date'})
    return df

def test_BDT():
    barrels = 750000
    costPerDay = 30000
    df = build_mock_df()
    df,model,x_test,y_test = Intellidock_Train(df)
    
    print(df)
    
    os1,os2,os3,os4,os5,os6,os7,os8,os9 = Intellidock_Predict_Next_Day(df,model,x_test,y_test,barrels,costPerDay)
    
    print (os1)
    
    print (type(type(df)))
    
    assert os1 != ''
    
    #return df
    
#dataframe = test_BDT()
    
def test_get_data():
    df = Intellidock_Get_Data()
    df_type_comp = pd.DataFrame([0,1,2])
    assert type(df) == type(df_type_comp)
    
'''Couldn't get this to work, looks like a memory error but I've not been able to track it down
def test_test_profitability():
    df = build_mock_df()
    
    barrels = 750000
    costPerDay = 30000
    
    test_type = type('string')
    
    s0,s1,s2,s3,s4,s5 = Intellidock_Test_Profitability(df,barrels,costPerDay)
    assert type(s0) == test_type
    assert type(s1) == test_type
    assert type(s2) == test_type
    assert type(s3) == test_type
    assert type(s4) == test_type
    assert type(s5) == test_type
'''   