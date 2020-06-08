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
from PIL import Image
import predict_lstm as pl

root = Tk()
root.wm_title("Intellidock")





PATH = os.environ["DF_ROOT"]
sys.path.insert(1, PATH)


INPUT_PATH = PATH + "/LSTM_dev/inputs/"
OUTPUT_PATH = PATH + "/LSTM_dev/outputs/"



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TZ"] = "Europe/London"



def load():
    """
    from paramiko import SSHClient, RSAKey, AutoAddPolicy
    import os
    vm = SSHClient()
    key = RSAKey.from_private_key_file("/Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem")
    vm.load_system_host_keys()
    vm.set_missing_host_key_policy(AutoAddPolicy())
    vm.connect("ec2-35-168-8-193.compute-1.amazonaws.com", username="ec2-user", pkey=key)
    print("connected")
    command = "source /home/ec2-user/CC/Team-Cpp/LSTM_dev/cc/env/bin/activate && source /home/ec2-user/CC/Team-Cpp/env.sh && python3 /home/ec2-user/CC/Team-Cpp/LSTM_dev/predict_lstm.py --getData --trainModel"
    stdin,stdout,stderr = vm.exec_command(command)
    pred = stdout.readlines()
    vm.close()
    del vm, stdin, stdout, stderr
    print(pred[-1])
    """


    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/LSTM_dev/inputs/inputData.csv /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_dev/VM")
    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/LSTM_dev/outputs/LSTM_model.h5 /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/VM")
    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/LSTM_dev/outputs/LSTM_history.csv /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_dev/VM")
    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/LSTM_dev/outputs/dataScaler.save /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/VM")
    



    df = pd.read_csv('/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/VM/inputData.csv')
    #load model with paths to the stuff

    model = pl.load_model('/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/VM/LSTM_model.h5')
    history = pd.read_csv('/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/VM/LSTM_history.csv')
    sc = joblib.load('/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/VM/dataScaler.save')
    weights = model.get_weights()
    
    btn = Button(root, text="Predict price for tomorrow",bg = 'green', command = lambda: predicc(df,weights),font = ('Arial',30))
    btn.grid(column=0, row=3)
    
    
    btn = Button(root, text="Plot correlations of variables",bg = 'green',command = lambda: correls(df), font = ('Arial',30))
    btn.grid(column=0, row=4)
    
        
    btn = Button(root, text="Plot variables used for training",bg = 'green',command = lambda: trainvars(df), font = ('Arial',30))
    btn.grid(column=0, row=5)
    
    
    btn = Button(root, text="Plot model training/test loss",bg = 'green',command = lambda: traintestloss(history), font = ('Arial',30))
    btn.grid(column=0, row=6)
    
    btn = Button(root, text="Plot predicted vs real price",bg = 'green',command = lambda: pricevs(df,weights),font = ('Arial',30))
    btn.grid(column=0, row=7)
    
    btn = Button(root, text="Plot model architecture",bg = 'green',command = lambda: archi(model), font = ('Arial',30))
    btn.grid(column=0, row=8)
    
    
    return()

def new():
    
    from paramiko import SSHClient, RSAKey, AutoAddPolicy
    import os
    vm = SSHClient()
    key = RSAKey.from_private_key_file("/Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem")
    vm.load_system_host_keys()
    vm.set_missing_host_key_policy(AutoAddPolicy())
    vm.connect("ec2-35-168-8-193.compute-1.amazonaws.com", username="ec2-user", pkey=key)
    print("connected")
    command = "source /home/ec2-user/CC/Team-Cpp/LSTM_dev/cc/env/bin/activate && source /home/ec2-user/CC/Team-Cpp/env.sh && python3 /home/ec2-user/CC/Team-Cpp/LSTM_dev/predict_lstm.py --getData --trainModel"
    stdin,stdout,stderr = vm.exec_command(command)
    pred = stdout.readlines()
    vm.close()
    del vm, stdin, stdout, stderr
    print(pred[-1])
   


    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/LSTM_dev/inputs/inputData.csv /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_dev/VM")
    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/LSTM_dev/outputs/LSTM_model.h5 /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/VM")
    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/LSTM_dev/outputs/LSTM_history.csv /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_dev/VM")
    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/LSTM_dev/outputs/dataScaler.save /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/VM")
    



    df = pd.read_csv('/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/VM/inputData.csv')
    #load model with paths to the stuff

    model = pl.load_model('/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/VM/LSTM_model.h5')
    history = pd.read_csv('/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/VM/LSTM_history.csv')
    sc = joblib.load('/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/VM/dataScaler.save')
    weights = model.get_weights()
    
    btn = Button(root, text="Predict price for tomorrow",bg = 'green', command = lambda: predicc(df,weights),font = ('Arial',30))
    btn.grid(column=0, row=3)
    
    
    btn = Button(root, text="Plot correlations of variables",bg = 'green',command = lambda: correls(df), font = ('Arial',30))
    btn.grid(column=0, row=4)
    
        
    btn = Button(root, text="Plot variables used for training",bg = 'green',command = lambda: trainvars(df), font = ('Arial',30))
    btn.grid(column=0, row=5)
    
    
    btn = Button(root, text="Plot model training/test loss",bg = 'green',command = lambda: traintestloss(history), font = ('Arial',30))
    btn.grid(column=0, row=6)
    
    btn = Button(root, text="Plot predicted vs real price",bg = 'green',command = lambda: pricevs(df,weights),font = ('Arial',30))
    btn.grid(column=0, row=7)
    
    btn = Button(root, text="Plot model architecture",bg = 'green',command = lambda: archi(model), font = ('Arial',30))
    btn.grid(column=0, row=8)
    
    
    return()


def predicc(df,weights):
    
    from paramiko import SSHClient, RSAKey, AutoAddPolicy
    import os
    vm = SSHClient()
    key = RSAKey.from_private_key_file("/Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem")
    vm.load_system_host_keys()
    vm.set_missing_host_key_policy(AutoAddPolicy())
    vm.connect("ec2-35-168-8-193.compute-1.amazonaws.com", username="ec2-user", pkey=key)
    print("connected")
    command = "source /home/ec2-user/CC/Team-Cpp/LSTM_dev/cc/env/bin/activate && source /home/ec2-user/CC/Team-Cpp/env.sh && python3 /home/ec2-user/CC/Team-Cpp/LSTM_dev/predict_lstm.py --predict"
    stdin,stdout,stderr = vm.exec_command(command)
    pred = stdout.readlines()
    vm.close()
    del vm, stdin, stdout, stderr
    string = (pred[-1])
   
    window = Toplevel(root) 
    window.wm_title("Prediction for tomorrow")
    lbl = Label(window, text=string,font = ('Arial',32))
    lbl.grid(column=0, row=0)

def correls(df):
    
    from paramiko import SSHClient, RSAKey, AutoAddPolicy
    import os
    vm = SSHClient()
    key = RSAKey.from_private_key_file("/Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem")
    vm.load_system_host_keys()
    vm.set_missing_host_key_policy(AutoAddPolicy())
    vm.connect("ec2-35-168-8-193.compute-1.amazonaws.com", username="ec2-user", pkey=key)
    print("connected")
    command = "source /home/ec2-user/CC/Team-Cpp/LSTM_dev/cc/env/bin/activate && source /home/ec2-user/CC/Team-Cpp/env.sh && python3 /home/ec2-user/CC/Team-Cpp/LSTM_dev/predict_lstm.py --plotCorrelations"
    stdin,stdout,stderr = vm.exec_command(command)
    pred = stdout.readlines()
    vm.close()
    del vm, stdin, stdout, stderr
    print(pred[-1])
   
    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/LSTM_dev/outputs/mainFeatureCorrelations.png /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_dev/VM")
    file = "/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/VM/mainFeatureCorrelations.png"
    
    window = Toplevel(root)  
    
    window.wm_title("Correlations of variables")
    canvas = Canvas(window, width = 500, height = 500)      
    canvas.pack()
    
    pil_image = Image.open(file)
    image200x100 = pil_image.resize((600, 500), Image.ANTIALIAS)
    image200x100.save(file)        
    window.img = img = PhotoImage(file=file)  

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
    
    from paramiko import SSHClient, RSAKey, AutoAddPolicy
    import os
    vm = SSHClient()
    key = RSAKey.from_private_key_file("/Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem")
    vm.load_system_host_keys()
    vm.set_missing_host_key_policy(AutoAddPolicy())
    vm.connect("ec2-35-168-8-193.compute-1.amazonaws.com", username="ec2-user", pkey=key)
    print("connected")
    command = "source /home/ec2-user/CC/Team-Cpp/LSTM_dev/cc/env/bin/activate && source /home/ec2-user/CC/Team-Cpp/env.sh && python3 /home/ec2-user/CC/Team-Cpp/LSTM_dev/predict_lstm.py --plotVariables"
    stdin,stdout,stderr = vm.exec_command(command)
    pred = stdout.readlines()
    vm.close()
    del vm, stdin, stdout, stderr
    print(pred[-1])

    
    for i in train_cols:
        os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/LSTM_dev/outputs/featurePlots/"+i+".png /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_dev/VM")

        file = "/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/VM/"+i+".png"
        window = Toplevel(root)  
        window.wm_title(i)
        
        canvas = Canvas(window, width = 900, height = 600)      
        canvas.pack()   
        
        pil_image = Image.open(file)
        image200x100 = pil_image.resize((900, 650), Image.ANTIALIAS)
        image200x100.save(file)
        window.img = img = PhotoImage(file=file)  
    
        canvas.create_image(0,0, anchor=NW, image=img)
    
    
def traintestloss(history):
    from paramiko import SSHClient, RSAKey, AutoAddPolicy
    import os
    vm = SSHClient()
    key = RSAKey.from_private_key_file("/Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem")
    vm.load_system_host_keys()
    vm.set_missing_host_key_policy(AutoAddPolicy())
    vm.connect("ec2-35-168-8-193.compute-1.amazonaws.com", username="ec2-user", pkey=key)
    print("connected")
    command = "source /home/ec2-user/CC/Team-Cpp/LSTM_dev/cc/env/bin/activate && source /home/ec2-user/CC/Team-Cpp/env.sh && python3 /home/ec2-user/CC/Team-Cpp/LSTM_dev/predict_lstm.py --trainTestLoss"
    stdin,stdout,stderr = vm.exec_command(command)
    pred = stdout.readlines()
    vm.close()
    del vm, stdin, stdout, stderr
    print(pred[-1])
    
    
    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/LSTM_dev/outputs/train_vis_BS_.png /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_dev/VM")

    
    window = Toplevel(root)  
    window.wm_title("Model training/test loss")

    
    canvas = Canvas(window, width = 650, height = 500)      
    canvas.pack()          
    window.img = img = PhotoImage(file="/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/VM/train_vis_BS_.png")  

    canvas.create_image(0,0, anchor=NW, image=img)

def go(entry1,entry2,df,weights):
    
    numDaysAgo = int(entry1.get())
    numDaysUntil = int(entry2.get())
    
    from paramiko import SSHClient, RSAKey, AutoAddPolicy
    import os
    vm = SSHClient()
    key = RSAKey.from_private_key_file("/Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem")
    vm.load_system_host_keys()
    vm.set_missing_host_key_policy(AutoAddPolicy())
    vm.connect("ec2-35-168-8-193.compute-1.amazonaws.com", username="ec2-user", pkey=key)
    print("connected")
    command = "source /home/ec2-user/CC/Team-Cpp/LSTM_dev/cc/env/bin/activate && source /home/ec2-user/CC/Team-Cpp/env.sh && python3 /home/ec2-user/CC/Team-Cpp/LSTM_dev/predict_lstm.py --daysSince"+ str(numDaysAgo) +" --daysUntil"+str(numDaysUntil)
    stdin,stdout,stderr = vm.exec_command(command)
    pred = stdout.readlines()
    vm.close()
    del vm, stdin, stdout, stderr
    print(pred[-1])
    
        
    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/LSTM_dev/outputs/pred_vs_real_BS.png /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_dev/VM")


    
    window = Toplevel(root)  
    
    window.wm_title("Predicted vs real price")
            
    canvas = Canvas(window, width = 630, height = 470)      
    canvas.pack()          
    window.img = img = PhotoImage(file="/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/VM/pred_vs_real_BS.png")  

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
    
    from paramiko import SSHClient, RSAKey, AutoAddPolicy
    import os
    vm = SSHClient()
    key = RSAKey.from_private_key_file("/Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem")
    vm.load_system_host_keys()
    vm.set_missing_host_key_policy(AutoAddPolicy())
    vm.connect("ec2-35-168-8-193.compute-1.amazonaws.com", username="ec2-user", pkey=key)
    print("connected")
    command = "source /home/ec2-user/CC/Team-Cpp/LSTM_dev/cc/env/bin/activate && source /home/ec2-user/CC/Team-Cpp/env.sh && python3 /home/ec2-user/CC/Team-Cpp/LSTM_dev/predict_lstm.py plotArchitecture"
    stdin,stdout,stderr = vm.exec_command(command)
    pred = stdout.readlines()
    vm.close()
    del vm, stdin, stdout, stderr
    print(pred[-1])
    
      
        
    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/LSTM_dev/outputs/model.png /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_dev/VM")

    
    window = Toplevel(root) 
    window.wm_title("Model architecture")
    
    canvas = Canvas(window, width = 240, height = 770)      
    canvas.pack()          
    window.img = img = PhotoImage(file="/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/LSTM_Dev/VM/model.png")  

    canvas.create_image(0,0, anchor=NW, image=img)
    return()
    

lbl = Label(root, text="Welcome to Intellidock",font = ('Arial Bold',50))
lbl.grid(column=0, row=0)
     


btn = Button(root, text="Load",bg = 'green',command = load, font = ('Arial',30))
btn.grid(column=0, row=1)

btn = Button(root, text="Train New",bg = 'green',command = new, font = ('Arial',30))
btn.grid(column=0, row=2)

btn = Button(root, text="Predict price for tomorrow",bg = 'green', command = lambda: predicc(df),font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=3)


btn = Button(root, text="Plot correlations of variables",bg = 'green',command = lambda: correls(df), font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=4)

    
btn = Button(root, text="Plot variables used for training",bg = 'green',command = lambda: trainvars(df), font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=5)


btn = Button(root, text="Plot model training/test loss",bg = 'green',command = lambda: traintestloss(history), font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=6)

btn = Button(root, text="Plot predicted vs real price",bg = 'green',command = lambda: pricevs(df),font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=7)

btn = Button(root, text="Plot model architecture",bg = 'green',command = lambda: archi(model), font = ('Arial',30),state = 'disabled')
btn.grid(column=0, row=8)

mainloop()

