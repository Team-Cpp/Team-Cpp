#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:09:19 2020

@author: nj18237
"""


import csv
import requests
import pandas as pd
from datetime import datetime

#-----ECDC Covid Data

def ECDC_covid_data_download():    
    url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
    data = pd.read_csv(url)
    time_strings = pd.Series(data['dateRep'])
    datetime_list = []
    for i in time_strings:
        datetime_list.append(datetime.strptime(i,'%d/%m/%Y')) 
    datetime_series = pd.Series(datetime_list)
    data['Date'] = datetime_series
    return data

def ECDC_covid_data_combine_worldwide(df):
    #df_totals = df.sort_values(by = ['dateRep'])
    df_totals = df.resample('d',on='Date')['cases'].sum()
    return df_totals

def ECDC_Covid_Data():#returns a series of the worldwide totals for each day from the start of 2020
    return ECDC_covid_data_combine_worldwide(ECDC_covid_data_download())
