#import tkinter
from tkinter import *


from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import sys
import time
from sklearn.externals import joblib

import predict_lstm as pl

root = Tk()
root.wm_title("Intellidock")

df = pd.read_csv('/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/inputs/inputData.csv')

PATH = os.environ["DF_ROOT"]
sys.path.insert(1, PATH)


INPUT_PATH = PATH + "/LSTM_dev/inputs/"
OUTPUT_PATH = PATH + "/LSTM_dev/outputs/"



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TZ"] = "Europe/London"



dataFileName = "inputData.csv"
modFileName = "LSTM_model.h5"
histFileName = "LSTM_history.csv"
scalerFileName = "dataScaler.save"

def updateData():
    window = Toplevel(root)  
    lbl = Label(window, text="Data is downloaded",font = ('Arial',30))
    lbl.grid(column=0, row=0)    
    #df = pl.getLstmData()
    df = pd.read_csv('/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/inputs/inputData.csv')

    
    btn = Button(root, text="Train new model",bg = 'green',command = lambda: new(df), font = ('Arial',30))
    btn.grid(column=0, row=2)
    
    btn = Button(root, text="Load existing model",bg = 'green',command = load, font = ('Arial',30))
    btn.grid(column=0, row=3)

def load():
    model = pl.load_model(os.path.join(OUTPUT_PATH, modFileName))
    history = pd.read_csv(os.path.join(OUTPUT_PATH, histFileName))
    sc = joblib.load(os.path.join(OUTPUT_PATH, scalerFileName))
    weights = model.get_weights()
    
    btn = Button(root, text="Predict price for tomorrow",bg = 'green', command = lambda: predicc(df,weights),font = ('Arial',30))
    btn.grid(column=0, row=4)
    
    
    btn = Button(root, text="Plot correlations of variables",bg = 'green',command = lambda: correls(df), font = ('Arial',30))
    btn.grid(column=0, row=5)
    
        
    btn = Button(root, text="Plot variables used for training",bg = 'green',command = lambda: trainvars(df), font = ('Arial',30))
    btn.grid(column=0, row=6)
    
    
    btn = Button(root, text="Plot model training/test loss",bg = 'green',command = lambda: traintestloss(history), font = ('Arial',30))
    btn.grid(column=0, row=7)
    
    btn = Button(root, text="Plot predicted vs real price",bg = 'green',command = lambda: pricevs(df,weights),font = ('Arial',30))
    btn.grid(column=0, row=8)
    
    btn = Button(root, text="Plot model architecture",bg = 'green',command = lambda: archi(model), font = ('Arial',30))
    btn.grid(column=0, row=9)
    
    
    return()

def new(df):
    
    model, history, weights = pl.train_model(df)
    
    btn = Button(root, text="Predict price for tomorrow",bg = 'green', command = lambda: predicc(df,weights),font = ('Arial',30))
    btn.grid(column=0, row=4)
    
    
    btn = Button(root, text="Plot correlations of variables",bg = 'green',command = lambda: correls(df), font = ('Arial',30))
    btn.grid(column=0, row=5)
    
        
    btn = Button(root, text="Plot variables used for training",bg = 'green',command = lambda: trainvars(df), font = ('Arial',30))
    btn.grid(column=0, row=6)
    
    
    btn = Button(root, text="Plot model training/test loss",bg = 'green',command = lambda: traintestloss(history), font = ('Arial',30))
    btn.grid(column=0, row=7)
    
    btn = Button(root, text="Plot predicted vs real price",bg = 'green',command = lambda: pricevs(df,weights),font = ('Arial',30))
    btn.grid(column=0, row=8)
    
    btn = Button(root, text="Plot model architecture",bg = 'green',command = lambda: archi(model), font = ('Arial',30))
    btn.grid(column=0, row=9)


    return()
    

def predicc(df,weights):
    window = Toplevel(root) 
    window.wm_title("Prediction for tomorrow")
    string = pl.predict_new(weights, df)
    lbl = Label(window, text=string,font = ('Arial',32))
    lbl.grid(column=0, row=0)

def correls(df):
    pl.plot_correlations(df)
    
    window = Toplevel(root)  
    
    window.wm_title("Correlations of variables")
    canvas = Canvas(window, width = 580, height = 450)      
    canvas.pack()          
    window.img = img = PhotoImage(file="/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/outputs/mainFeatureCorrelations.png")  

    canvas.create_image(0,0, anchor=NW, image=img)
    


def trainvars(df):
    train_cols = [
    "Prices",
    "OilProduction",
    "NatGasPrices",
    "Nasdaq",
    "daysAbove200dSMA",
    "MACD_12_26",
    "Momentum_14",
    ]
    
    pl.plot_data(df,variables=train_cols, days=60)
    
    for i in train_cols:
        window = Toplevel(root)  
        window.wm_title(i)
        
        canvas = Canvas(window, width = 700, height = 500)      
        canvas.pack()          
        window.img = img = PhotoImage(file="/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/outputs/featurePlots/"+i+".png")  
    
        canvas.create_image(0,0, anchor=NW, image=img)
    
    
def traintestloss(history):
    pl.test_train_loss(history)
    
    window = Toplevel(root)  
    window.wm_title("Model training/test loss")

    
    canvas = Canvas(window, width = 700, height = 500)      
    canvas.pack()          
    window.img = img = PhotoImage(file="/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/outputs/train_vis_BS_.png")  

    canvas.create_image(0,0, anchor=NW, image=img)

def go(entry1,entry2,df,weights):
    
    numDaysAgo = int(entry1.get())
    numDaysUntil = int(entry2.get())
    
    pl.visualise_prediction(df, numDaysAgo, numDaysUntil,weights)
    
    window = Toplevel(root)  
    
    window.wm_title("Predicted vs real price")
            
    canvas = Canvas(window, width = 680, height = 470)      
    canvas.pack()          
    window.img = img = PhotoImage(file="/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/outputs/pred_vs_real_BS.png")  

    canvas.create_image(0,0, anchor=NW, image=img)
    
    
def pricevs(df,weights):
    window = Toplevel(root)  

    lbl = Label(window, text="Enter how many days ago you want the prediction to start and end",font = ('Arial',12))
    lbl.grid(column=0, row=0)

    entry1 = Entry(window)
    entry1.grid(column=0, row=1)
    
    entry2 = Entry(window)
    entry2.grid(column=1, row=1)
    
    btn = Button(window, text="Go",bg = 'green', command = lambda: go(entry1,entry2,df,weights),font = ('Arial',30))
    btn.grid(column=0, row=2)
    
    return()
 
    
def archi(model):
    pl.plot_model(model.model, to_file=os.path.join(OUTPUT_PATH, "model.png"))
    
    window = Toplevel(root) 
    window.wm_title("Model architecture")
    
    canvas = Canvas(window, width = 240, height = 770)      
    canvas.pack()          
    window.img = img = PhotoImage(file="/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/outputs/model.png")  

    canvas.create_image(0,0, anchor=NW, image=img)
    return()
    

lbl = Label(root, text="Welcome to Intellidock",font = ('Arial Bold',50))
lbl.grid(column=0, row=0)
     
btn = Button(root, text="Download Data",bg = 'green', command = updateData,font = ('Arial',30))
btn.grid(column=0, row=1)



btn = Button(root, text="Train new model",bg = 'green',command = lambda: new(df), font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=2)

btn = Button(root, text="Load existing model",bg = 'green',command = load, font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=3)


btn = Button(root, text="Predict price for tomorrow",bg = 'green', command = lambda: predicc(df),font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=4)


btn = Button(root, text="Plot correlations of variables",bg = 'green',command = lambda: correls(df), font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=5)

    
btn = Button(root, text="Plot variables used for training",bg = 'green',command = lambda: trainvars(df), font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=6)


btn = Button(root, text="Plot model training/test loss",bg = 'green',command = lambda: traintestloss(history), font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=7)

btn = Button(root, text="Plot predicted vs real price",bg = 'green',command = lambda: pricevs(df),font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=8)

btn = Button(root, text="Plot model architecture",bg = 'green',command = lambda: archi(model), font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=9)

mainloop()

