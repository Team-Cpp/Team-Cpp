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

import predict_lstm as pl

root = Tk()
root.wm_title("Intellidock")

def updateData():
    window = Toplevel(root)  
    lbl = Label(window, text="Data is downloaded",font = ('Arial',30))
    lbl.grid(column=0, row=0)    
    df = pl.getLstmData()
    
    btn = Button(root, text="Train new model",bg = 'green',command = new, font = ('Arial',30))
    btn.grid(column=0, row=2)
    
    btn = Button(root, text="Load existing model",bg = 'green',command = load, font = ('Arial',30))
    btn.grid(column=0, row=3)

def load():
    model = pl.load_model(os.path.join(OUTPUT_PATH, modFileName))
    history = pd.read_csv(os.path.join(OUTPUT_PATH, histFileName))
    sc = joblib.load(os.path.join(OUTPUT_PATH, scalerFileName))
    weights = model.get_weights()
    
    btn = Button(root, text="Predict price for tomorrow",bg = 'green', command = predicc,font = ('Arial',30))
    btn.grid(column=0, row=4)

    btn = Button(root, text="Plot correlations of variables",bg = 'green',command = correls, font = ('Arial',30))
    btn.grid(column=0, row=5)    
            
    btn = Button(root, text="Plot variables used for training",bg = 'green',command = trainvars, font = ('Arial',30))
    btn.grid(column=0, row=6)
    
    
    btn = Button(root, text="Plot model training/test loss",bg = 'green',command = traintestloss, font = ('Arial',30))
    btn.grid(column=0, row=7)
    
    btn = Button(root, text="Plot predicted vs real price",bg = 'green',command = pricevs,font = ('Arial',30))
    btn.grid(column=0, row=8)
    
    btn = Button(root, text="Plot model architecture",bg = 'green',command = archi, font = ('Arial',30))
    btn.grid(column=0, row=9)


    return()

def new():
    model, history, weights = pl.train_model(df)
    
    btn = Button(root, text="Predict price for tomorrow",bg = 'green', command = predicc,font = ('Arial',30))
    btn.grid(column=0, row=4)

    btn = Button(root, text="Plot correlations of variables",bg = 'green',command = correls, font = ('Arial',30))
    btn.grid(column=0, row=5)    
            
    btn = Button(root, text="Plot variables used for training",bg = 'green',command = trainvars, font = ('Arial',30))
    btn.grid(column=0, row=6)
    
    
    btn = Button(root, text="Plot model training/test loss",bg = 'green',command = traintestloss, font = ('Arial',30))
    btn.grid(column=0, row=7)
    
    btn = Button(root, text="Plot predicted vs real price",bg = 'green',command = pricevs,font = ('Arial',30))
    btn.grid(column=0, row=8)
    
    btn = Button(root, text="Plot model architecture",bg = 'green',command = archi, font = ('Arial',30))

    return()
    

def predicc():
    window = Toplevel(root) 
    string = pl.predict_new(weights, df)
    lbl = Label(root, text=string,font = ('Arial Bold',50))
    lbl.grid(column=0, row=0)

def correls():
    pl.plot_correlations(df)
    
    window = Toplevel(root)  
    
    canvas = Canvas(window, width = 600, height = 700)      
    canvas.pack()          
    window.img = img = PhotoImage(file="mainFeatureCorrelations.png")  

    canvas.create_image(0,0, anchor=NW, image=img)
    


def trainvars():
    pl.plot_data(df)
    
    window = Toplevel(root)  
    
    canvas = Canvas(window, width = 600, height = 700)      
    canvas.pack()          
    window.img = img = PhotoImage(file="featurePlots/*.png")  

    canvas.create_image(0,0, anchor=NW, image=img)
    
    
def traintestloss():
    pl.test_train_loss(history)
    
    window = Toplevel(root)  
    
    canvas = Canvas(window, width = 600, height = 700)      
    canvas.pack()          
    window.img = img = PhotoImage(file="train_vis_BS.png")  

    canvas.create_image(0,0, anchor=NW, image=img)

def go(numDaysAgo,numDaysUntil):
    pl.visualise_prediction(df, numDaysAgo, numDaysUntil)
    
    window = Toplevel(root)  
    
    canvas = Canvas(window, width = 600, height = 700)      
    canvas.pack()          
    window.img = img = PhotoImage(file="pred_vs_real_BS.png")  

    canvas.create_image(0,0, anchor=NW, image=img)
    
    
def pricevs():
    window = Toplevel(root)  


    entry1 = Entry(root)
    entry1.grid(column=0, row=0)
    
    entry2 = Entry(root)
    entry2.grid(column=1, row=0)
    
    numDaysAgo = entry1.get()
    numDaysUntil = entry2.get()
    
    btn = Button(root, text="Go",bg = 'green', command = go,font = ('Arial',30))
    btn.grid(column=0, row=1)
    
    return()
 
    
def archi():
    pl.plot_model(model.model, to_file=os.path.join(OUTPUT_PATH, "model.png"))
    
    window = Toplevel(root)  
    
    canvas = Canvas(window, width = 600, height = 700)      
    canvas.pack()          
    window.img = img = PhotoImage(file="model.png")  

    canvas.create_image(0,0, anchor=NW, image=img)
    return()
    

lbl = Label(root, text="Welcome to Intellidock",font = ('Arial Bold',50))
lbl.grid(column=0, row=0)
     
btn = Button(root, text="Download Data",bg = 'green', command = updateData,font = ('Arial',30))
btn.grid(column=0, row=1)


btn = Button(root, text="Train new model",bg = 'green',command = new, font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=2)

btn = Button(root, text="Load existing model",bg = 'green',command = load, font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=3)


btn = Button(root, text="Predict price for tomorrow",bg = 'green', command = predicc,font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=4)


btn = Button(root, text="Plot correlations of variables",bg = 'green',command = correls, font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=5)

    
btn = Button(root, text="Plot variables used for training",bg = 'green',command = trainvars, font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=6)


btn = Button(root, text="Plot model training/test loss",bg = 'green',command = traintestloss, font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=7)

btn = Button(root, text="Plot predicted vs real price",bg = 'green',command = pricevs,font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=8)

btn = Button(root, text="Plot model architecture",bg = 'green',command = archi, font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=9)

mainloop()

