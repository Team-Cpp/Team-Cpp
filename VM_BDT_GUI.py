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
import csv
import client
from PIL import Image
root = Tk()
root.wm_title("Intellidock")

def predict():
    
    from paramiko import SSHClient, RSAKey, AutoAddPolicy
    import os
    vm = SSHClient()
    key = RSAKey.from_private_key_file("/Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem")
    vm.load_system_host_keys()
    vm.set_missing_host_key_policy(AutoAddPolicy())
    vm.connect("ec2-35-168-8-193.compute-1.amazonaws.com", username="ec2-user", pkey=key)
    print("connected")
    command = "source /home/ec2-user/CC/Team-Cpp/LSTM_dev/cc/env/bin/activate && source /home/ec2-user/CC/Team-Cpp/env.sh && python3 /home/ec2-user/CC/Team-Cpp/GUI/Chris_Intellidock.py --predict"
    stdin,stdout,stderr = vm.exec_command(command)
    pred = stdout.readlines()
    vm.close()
    del vm, stdin, stdout, stderr
    print(pred)
    
   

    #os.system("ssh teamc@52.170.190.142 && source env.sh && python3 team-Cpp/GUI/Chris_Intellidock.py --predict")
    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/commonFunctions/BDT_Predicted_Data.csv /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/VM")
    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/commonFunctions/Feature_Importance.png /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/VM")
    
    
    window = Toplevel(root)
    
    with open("/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/VM/BDT_Predicted_Data.csv", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):        
            lbl = Label(window, text=line,font = ('Arial',30))
            lbl.grid(column=0, row=i)  

    
    window = Toplevel(root)  
    
    canvas = Canvas(window, width = 600, height = 800)      
    canvas.pack() 
    file = "/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/VM/Feature_Importance.png"
    pil_image = Image.open(file)
    image200x100 = pil_image.resize((600, 800), Image.ANTIALIAS)
    image200x100.save(file)
    window.img = img = PhotoImage(file=file)  

    img.zoom(2, 2)
    canvas.create_image(0,0, anchor=NW, image=img)
    
    return()
    
   
def profit():
    from paramiko import SSHClient, RSAKey, AutoAddPolicy
    import os
    vm = SSHClient()
    key = RSAKey.from_private_key_file("/Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem")
    vm.load_system_host_keys()
    vm.set_missing_host_key_policy(AutoAddPolicy())
    vm.connect("ec2-35-168-8-193.compute-1.amazonaws.com", username="ec2-user", pkey=key)
    print("connected")
    command = "source /home/ec2-user/CC/Team-Cpp/LSTM_dev/cc/env/bin/activate && source /home/ec2-user/CC/Team-Cpp/env.sh && python3 /home/ec2-user/CC/Team-Cpp/GUI/Chris_Intellidock.py --testProfit"
    stdin,stdout,stderr = vm.exec_command(command)
    pred = stdout.readlines()
    vm.close()
    del vm, stdin, stdout, stderr

    os.system("scp -i /Users/eleonoraparrag/Documents/Coding_Challenge/AWSKey.pem ec2-user@ec2-35-168-8-193.compute-1.amazonaws.com:/home/ec2-user/CC/Team-Cpp/commonFunctions/BDT_Profitability_Test.csv /Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/VM")
    
    
    window = Toplevel(root)
    
    with open("/Users/eleonoraparrag/Documents/Coding_Challenge/Team-Cpp-master/VM/BDT_Profitability_Test.csv", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):        
            lbl = Label(window, text=line,font = ('Arial',30))
            lbl.grid(column=0, row=i)  
    
    
    return()

lbl = Label(root, text="Welcome to Intellidock",font = ('Arial Bold',50))
lbl.grid(column=0, row=0)
 
btn = Button(root, text="Make a predicition for tomorrow",bg = 'green',command = predict,font = ('Arial',30))
btn.grid(column=0, row=1)


btn = Button(root, text="Test the accuracy of the system",bg = 'green',command = profit,font = ('Arial',30))
btn.grid(column=0, row=2)

mainloop()
