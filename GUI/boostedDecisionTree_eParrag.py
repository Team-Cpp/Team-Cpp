import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image  
from sklearn import tree


print('Running...')


#US Oil Production in 1000 barrels per day
#url2 = 'https://www.eia.gov/dnav/pet/hist_xls/WCRFPUS2w.xls'
#r2 = requests.get(url2)
#data_xls = pd.read_excel(url2, 'Data 1', skiprows = 2, comment='#') 
#data_xls.columns = ['Date','Production of Crude Oil']
#Oil_Production = data_xls

#Gathering data to use

#WTI Oil Price
new1 = pd.read_csv('https://datahub.io/core/oil-prices/r/wti-daily.csv')
Date = new1['Date']   
Price = new1['Price']
Prices = pd.DataFrame(np.zeros([len(Date),2]),columns=['Date','Price'])
new_df1 = pd.DataFrame({'Date': Date})
Prices.update(new_df1)
new_df2 = pd.DataFrame({'Price': Price})
Prices.update(new_df2)


#Natural Gas Price
new = pd.read_csv('https://datahub.io/core/natural-gas/r/daily.csv')
Date_gas = new['Date']   
Price_gas = new['Price']
Gas = pd.DataFrame(np.zeros([len(Date_gas),2]),columns=['Date','Price_gas'])
new_df1 = pd.DataFrame({'Date': Date_gas})
Gas.update(new_df1)
new_df2 = pd.DataFrame({'Price_gas': Price_gas})
Gas.update(new_df2)


#Brent Oil Price
brent = pd.read_csv('https://datahub.io/core/oil-prices/r/brent-daily.csv')
brentd = brent['Date']   
brentp = brent['Price']
Brent = pd.DataFrame(np.zeros([len(brentd),2]),columns=['Date','Brent'])
new_df1 = pd.DataFrame({'Date': brentd})
Brent.update(new_df1)
new_df2 = pd.DataFrame({'Brent': brentp})
Brent.update(new_df2)


#Converting to datetime-type in order to merge
Prices['Date'] =  pd.to_datetime(Prices['Date'])
Gas['Date'] =  pd.to_datetime(Gas['Date'])
Brent['Date'] =  pd.to_datetime(Brent['Date'])


#Merging Dataframes
#merged_df = Oil_Production.merge(Prices, how = 'inner', on = ['Date'])
merged_df = Gas.merge(Prices, how = 'inner', on = ['Date'])
merged_df = merged_df.merge(Brent, how = 'inner', on = ['Date'])


#Creating x and y (features and price)
x = merged_df.drop(columns="Price")
x = x.drop(columns="Date")
y = merged_df.drop(columns="Date")
#y = y.drop(columns="Production of Crude Oil")
y = y.drop(columns="Price_gas")
y = y.drop(columns="Brent")


#Creating a Plot

y = np.array(y)
y = y.astype(np.float)
z = y*100
date = np.linspace(0,len(y),len(y))
w = x['Price_gas'] * 500
j = x['Brent'] * 100

plt.title('Correlations')
plt.plot(date,z, label = 'Price of WTI Oil')
#plt.plot(date,x['Production of Crude Oil'], label = 'Production of Crude Oil')
plt.plot(date,w, label = 'Price of Natural Gas')
plt.plot(date,j, label = 'Price of Brent Oil')
plt.xlabel('Date')
plt.ylabel('Not to Scale')
plt.legend()
plt.show()


#Preparing the data


#Delete first y and last x to match features to the next day
y_t = np.delete(y,[0])
n = len(x)
x_t = x.drop([n-1],axis = 0)

#Boosted Decision Tree
X_train,X_test,y_train,y_test = train_test_split(x_t,y_t)#split data into testing and training sets

regressor = GradientBoostingRegressor( 
    max_depth = 2, #no. leaves on each tree
    n_estimators = 3, #total no. tress in ensemble
    learning_rate = 1.0 #scales contribution of each tree
)

#finds optimal number of trees by measuring validation error at each stage of training
regressor.fit(X_train, y_train)
errors = [mean_squared_error(y_test,y_pred) for y_pred in
    regressor.staged_predict(X_test)]
best_n_estimators = np.argmin(errors) 


#build and fit model using optimal number of trees
best_regressor = GradientBoostingRegressor( 
    max_depth = 3,
    n_estimators = best_n_estimators,
    learning_rate = 0.1
)



best_regressor.fit(X_train,y_train) #Train the model


td = 1000 #no days past to test
x_test = x.iloc[-td:] #X_test 
y_test = y[-td:] #y_test 
#y_pred = best_regressor.predict(x.iloc[-td:]) #Predict price based on test values
#err = mean_absolute_error(y[-td:],y_pred) #average distance from predictions and absolute values

y_pred = best_regressor.predict(x_test) #Predict price based on test values
err = mean_absolute_error(y_test,y_pred) #average distance from predictions and absolute values



#taking most recent td days to test
p = np.linspace(0,len(x_test), len(x_test))#setting up x axis for prediction plot


plt.figure('Prediction')
plt.plot(p,y_test, label = 'Test')
plt.plot(p,y_pred, label = 'Prediction')

plt.legend()
plt.xlabel('Time (Weeks)')
plt.ylabel('Oil Price')
plt.show()

plt.plot(p,y_pred, label = 'Prediction')
plt.show()


tp = 0 #True Positive
tn = 0 #True Negative
fp = 0 #False Positive
fn = 0 #False Negative

#Compare the prediction for today with the prediction for tomorrow to see if increase
#Calculate values of confusion matrix

for i in range(len(y_test) - 1):
    if y_pred[i] > y_pred[i-1]:#Predicted to be Higher
        if y_test[i+1] > y_test[i]:#Actually Increased
            tp = tp + 1
        if y_test[i+1] < y_test[i]:#Actually Decreased
            fp = fp+1
    if y_pred[i] < y_pred[i-1]: #Predicted to be Lower
        if y_test[i+1] > y_test[i]:#Actually Increased
            fn = fn+1
        if y_test[i+1] < y_test[i]:#Actually Decreased
            tn = tn + 1
    if y_pred[i] == y_pred[i-1]: 
        if y_pred[i] > y_test[i]:#Predicted to be Higher
            if y_test[i+1] > y_test[i]:#Actually Increased
                tp = tp + 1
            if y_test[i+1] < y_test[i]:#Actually Decreased
                fp = fp+1
        if y_pred[i] < y_test[i]: #Predicted to be Lower
            if y[i+1] > y_test[i]:#Actually Increased
                fn = fn+1
            if y[i+1] < y_test[i]:#Actually Decreased
                tn = tn + 1
        

print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)

print('Success Rate  - percent correct predictions = %.2f'%(((tp+tn)/(tp+tn+fp+fn))*100))
print('Failure Rate (Lost Money)  - where you should have docked and you waited = %.2f'%((fp/(tp+tn+fp+fn))*100))
print('Percent Correctly telling you to dock = %.2f'%((tp/(tp+fp))*100))
print('Percent Correctly telling you to wait = %.2f'%((tn/(tn+fn))*100))