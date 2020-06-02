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

from Chris_Intellidock import Intellidock_Predict_Next_Day
from Chris_Intellidock import Intellidock_Train
from Chris_Intellidock import Intellidock_Get_Data

import pandas as pd
print(pd.__file__)
import requests
import numpy as np
import matplotlib.pyplot as plt

import dataFunctions as dataFun

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

import csv

##-----------------------------------------------------------------------------    
#End of function Definitions

#Autopredict functionality

#Default Parameters  
    
barrels = 750000
costPerDay = 30000
days = 1
option = -1

df = Intellidock_Get_Data()
df,model,x_test,y_test = Intellidock_Train(df)
os1,os2,os3,os4,os5,os6,os7,os8,os9 = Intellidock_Predict_Next_Day(df,model,x_test,y_test,barrels,costPerDay)

with open('BDT_Predicted_Data.csv','w',newline = '') as file:
    writer = csv.writer(file)
    
    writer.writerow([os1])
    
    writer.writerow(os2)
    
    writer.writerow(os3)
    
    writer.writerow(os4)
    
    writer.writerow(os5)
    
    writer.writerow(os6)
    
    writer.writerow(os7)
    
    writer.writerow(os8)
    
    writer.writerow(os9)

#---------------------------------------------