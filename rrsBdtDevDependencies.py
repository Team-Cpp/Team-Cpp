import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import dataFunctions as dataFun

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image, display
from sklearn import tree
