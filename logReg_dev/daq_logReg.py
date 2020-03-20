# -*- coding: utf-8 -*-

import urllib.request
import urllib.error
import shutil
import csv
import time
import pandas as pd
import numpy as np
import os.path

fname = 'Testdata.csv'
import_new_data = False

error = False#you can use this flag to tell the programme to basically skip everything else and go to the end so you don't get a crash when there's a problem

#Allows the user to choose to use a previously downloaded datafile, or to download a new one. Mostly so I don't have to keep downloading the same datafile over and over again
if os.path.isfile(fname) == True:
    while True:
        user_selection = input("Download new data (Y/N)? \n")
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
    url = 'https://datahub.io/core/oil-prices/r/brent-daily.csv'
    try:
        response = urllib.request.urlopen(url)
        print("Connection ok")
        data = response.read()      # a `bytes` object
        text = data.decode('utf-8') # a `str`; this step can't be used if data is binary
        with urllib.request.urlopen(url) as response, open(fname, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    except (urllib.error.HTTPError, urllib.error.URLError)  as e:
            requrl = url
            print ("Error, something went wrong with the download Error code:" , e.code)
            print ("If the error code is '404', try checking the url and/or your internet connection - 404 means that it can't find the address")
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

    frame = pd.DataFrame({'Dates': Dates_List,'Prices': Prices_List})

    print("Dataframe created!")

print( "Programme complete!")
    #print(frame)
        