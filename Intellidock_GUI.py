#import tkinter
from tkinter import *

import os
import sys
#sys.path.insert(0,os.environ['DF_ROOT'])

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from GUI.Chris_Intellidock import *


barrels = 750000
costPerDay = 30000
days = 1
trained = False
data_acquired = False

root = Tk()
root.wm_title("Intellidock")

#DEBUG_df = Intellidock_Get_Data() #purely for debug purposes

#root['bg'] = '#49A'

lbl = Label(root, text="Welcome to Intellidock",font = ('Arial Bold',50))
lbl.grid(column=0, row=0)
   
def download_data():
    window = Toplevel(root)  
    lbl = Label(window, text="Data is downloaded",font = ('Arial',30))
    lbl.grid(column=0, row=0)    
    df = Intellidock_Get_Data()  
    data_acquired = True
    
    #action_with_arg = partial(Train,df)
    
    btn = Button(root, text="Train the model (must do pre prediction)",bg = 'green',command = lambda: Train(df),font = ('Arial',30))
    btn.grid(column=0, row=3)

    btn = Button(root, text="Test the accuracy of the system",bg = 'green',command = lambda: Accuracy(df),font = ('Arial',30))
    btn.grid(column=0, row=2)
    
    btn = Button(root, text="Run a profitability check",bg = 'green',command = lambda: profit(df),font = ('Arial',30))
    btn.grid(column=0, row=5)
    
    #print(df)
    

def Accuracy(df):
    window = Toplevel(root)      
    Intellidock_Test_Accuracy(df,window,barrels,costPerDay)
    
def Train(df):
    window = Toplevel(root)  
    lbl = Label(window, text="Training complete",font = ('Arial',30))
    lbl.grid(column=0, row=0)    
    df,model,X_test,y_test = Intellidock_Train(df)
    
    btn = Button(root, text="Predict if staying undocked is worthwhile",bg = 'green',command = lambda: Predict(df,model,X_test,y_test),font = ('Arial',30))
    btn.grid(column=0, row=4)
    
    fig, ax = plt.subplots(figsize=(9,10))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.savefig('Decision_Tree.png')
    
    btn = Button(root, text="Show Feature Importance",bg = 'green',command = features,font = ('Arial',30))
    btn.grid(column=0, row=6)
    

def Predict(df,model,X_test,y_test):
    window = Toplevel(root)  

    Intellidock_Predict_Next_Day(df,model,X_test,y_test,window,barrels,costPerDay)
    

def profit(df):
    window = Toplevel(root)  

    Intellidock_Test_Profitability(df,window,barrels,costPerDay)
    
    window2 = Toplevel(window)
    canvas = Canvas(window2, width = 400, height = 300)      
    canvas.pack()          
    window2.img = img = PhotoImage(file="Deviation_Histogram.png")  

    canvas.create_image(0,0, anchor=NW, image=img)

def features():
    window = Toplevel(root)  
    
    canvas = Canvas(window, width = 1000, height = 800)      
    canvas.pack()          
    window.img = img = PhotoImage(file="Decision_Tree.png")  

    canvas.create_image(0,0, anchor=NW, image=img)

     
btn = Button(root, text="Download Data",bg = 'green',command = download_data,font = ('Arial',30))
btn.grid(column=0, row=1)


btn = Button(root, text="Test the accuracy of the system",bg = 'green',command = Accuracy,font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=2)

    
btn = Button(root, text="Train the model (must do pre prediction)",bg = 'green',command = Train,font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=3)


btn = Button(root, text="Predict if staying undocked is worthwhile",bg = 'green',command = Predict,font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=4)

    
btn = Button(root, text="Run a profitability check",bg = 'green',command = profit,font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=5)


btn = Button(root, text="Show Feature Importance",bg = 'green',command = features,font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=6)

mainloop()
