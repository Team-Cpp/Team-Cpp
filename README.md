# Team-C(++) CDT Coding Challenge work

This is the logistic regression branch. This will use the logistic regression model to predict whether the closing price of WTI oil will increase a sufficient amount the next day for an oil tanker to wait.

Considering the running costs of A = 30000$ per day and number of barrels B = 700000 the model will predict if $B\*D2 - A > B\*D1 $ where D2 is the next day and D1 is the current day specified by the user

##Quick start:

To use the existing model to make predictions with:
$ python Predict.py

The program will prompt to enter a date for which you wish to make a prediction for. This must be a date between 2000-01-01 and tomorrow. The input format must be "yyyy-mm-dd".
The program will then ask to enter the current price of WTI oil. If you want to predict the price tomorrow then this will be the price today. If you wish to predict price at 2013-02-03 then the input should be the price at 2013-02-02 or closest to that date (if 2013-02-02 falls on a weekend).

The program will then output whether the price will increase or decrease and associated probability.

If you wish to use a custom logistic regression model then run as:
$ python Predict.py <custom_model_name>

##Training a fresh model:

In order to train a new logistic regression model use:
$ python logRegression.py

This will fetch the latest WTI oil data and train a new model and output a model file. You can then use it to make predictions by typing:
$ python Predict.py <new_model_name>