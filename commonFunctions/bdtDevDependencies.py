import pandas as pd
import requests
import numpy as np
import quandl
import yfinance as yf
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

import dataFunctions as dataFun

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

from IPython.display import Image, display

import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import plot_importance, plot_tree