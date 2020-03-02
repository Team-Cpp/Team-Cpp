# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import os.path
#abspath = os.path.abspath(__file__)
#dname = os.path.dirname(abspath)
#os.chdir(dname)
import rrsBdtDevDependencies
import dataFunctions as dataFun
from datetime import datetime as dt


# %%
### CONFIGURE ###
barrels = 750000
costPerDay = 30000
days = 1
trainDataDate = '2018-01-01'


# %%
print('Running...')

def show_more(df, lines):
    with pd.option_context("display.max_rows", lines):
        display(df)


# %%
import quandl
wtiData = quandl.get("FRED/DCOILWTICO")


# %%
wtiData.reset_index(level=0, inplace=True)
wtiData = wtiData.rename(columns={"Value": "Prices"})
wtiData.head()


# %%
import yfinance as yf

yfStartDate = wtiData['Date'].iloc[-1].strftime('%Y-%m-%d')
stocks = "CL=F"
period = "1d"
Stocks, yfInfo = dataFun.yFinData(yfStartDate)

wtiData = wtiData.append(Stocks, ignore_index =True)
wtiData = wtiData.sort_values(by = ["Date"])


# %%
import numpy as np
oilDF = dataFun.oilProduction()

df = dataFun.combineFrames(wtiData,oilDF)
df = df[np.isfinite(df['Prices'])]
df = df.reset_index().drop(["index"], axis = 1)
df


# %%
natGasData = quandl.get("EIA/NG_RNGWHHD_D")
natGasData.reset_index(level=0, inplace=True)
natGasData = natGasData.rename(columns={"Value": "NatGasPrices"})

yfStartDate = natGasData['Date'].iloc[-1].strftime('%Y-%m-%d')
stocks = "NG=F"
period = "1d"
NGStocks, yfInfo = dataFun.yFinData(yfStartDate,stock=stocks,name ="NatGasPrices")
natGasData = natGasData.append(NGStocks, ignore_index =True)
natGasData = natGasData.sort_values(by = ["Date"])
natGasData


# %%
import pandas as pd
newdf = pd.merge(df, natGasData, on=['Date'], how ="left")
newdf.head()


# %%
brentData = quandl.get("FRED/DCOILBRENTEU")
brentData.reset_index(level=0, inplace=True)
name = "BrentPrices"
brentData = brentData.rename(columns={"Value": name})

yfStartDate = brentData['Date'].iloc[-1].strftime('%Y-%m-%d')
stocks = "BZ=F"
period = "1d"
BStocks, yfInfo = dataFun.yFinData(yfStartDate,stock=stocks,name = name)
brentData = brentData.append(BStocks, ignore_index =True)
brentData = brentData.sort_values(by = ["Date"])
brentData


# %%
df = pd.merge(newdf, brentData, on=['Date'], how ="left")
df = df[df["Date"] > trainDataDate]
df = df.rename(columns={"Production of Crude Oil": "OilProduction"})
df.isna().sum()


# %%
df["BrentPrices"] = df["BrentPrices"].interpolate(method='nearest')
df["NatGasPrices"] = df["NatGasPrices"].interpolate(method='nearest')
df.isna().sum()


# %%
df = df.reset_index().drop(["index"], axis = 1)
df["20dSMA"] = dataFun.SMA(20, df["Prices"])
df["10dSMA"] = dataFun.SMA(10, df["Prices"])
df["5dSMA"] = dataFun.SMA(5, df["Prices"])
df["50dSMA"] = dataFun.SMA(50, df["Prices"])
df["200dSMA"] = dataFun.SMA(200, df["Prices"])


df["boll_lo"] = dataFun.bollinger(df['Prices'])[0]
df["boll_hi"] = dataFun.bollinger(df['Prices'])[1]

df = dataFun.momentum(df, 14)
df = dataFun.macd(df, 12, 26)
df = dataFun.rate_of_change(df, 14)
df = dataFun.relative_strength_index(df)

df["boll_hi"] = pd.to_numeric(df["boll_hi"])
df["boll_lo"] = pd.to_numeric(df["boll_lo"])
df["20dSMA"] = pd.to_numeric(df["20dSMA"])
df["10dSMA"] = pd.to_numeric(df["10dSMA"])
df["5dSMA"] = pd.to_numeric(df["5dSMA"])
df["50dSMA"] = pd.to_numeric(df["50dSMA"])
df["200dSMA"] = pd.to_numeric(df["200dSMA"])

df["bollAmplitude"] = df["boll_hi"] - df["boll_lo"]
df["distFromTopBoll"] = df["boll_hi"] - df["Prices"]
df["distFromLowBoll"] = df["boll_lo"] - df["Prices"]
df["20d200dDist"] = np.abs(df["20dSMA"] - df["200dSMA"])
df


# %%
df = df[np.isfinite(df['200dSMA'])]
df.isna().sum()


# %%
df = df.drop_duplicates("Date",keep="first")
df


# %%
df = df[df["Date"] > trainDataDate]
df = df.reset_index().drop(["index"], axis = 1)
df


# %%
def create_features(df, label=None, shift = 0):
    """
    Creates time series features from datetime index
    """
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['dayofmonth'] = df['Date'].dt.day
    df['weekofyear'] = df['Date'].dt.weekofyear
    df = df.set_index('Date')
    X = df[['OilProduction', 'NatGasPrices', 'BrentPrices', '20dSMA', 'Momentum_14', 'MACD_12_26', 'MACDdiff_12_26', 'ROC_14', 'RSI_14', 'bollAmplitude', 'distFromTopBoll', 'distFromLowBoll', '20d200dDist','dayofyear','dayofmonth','weekofyear']]
    if shift > 0:
        tiems = X[['dayofyear','dayofmonth','weekofyear']]
        X = X[['OilProduction', 'NatGasPrices', 'BrentPrices', '20dSMA', 'Momentum_14', 'MACD_12_26', 'MACDdiff_12_26','ROC_14', 'RSI_14', 'bollAmplitude', 'distFromTopBoll', 'distFromLowBoll', '20d200dDist']].shift(shift)
        X = X.merge(tiems, how='inner', left_index=True, right_index=True)

    if label:
        y = df[label]
        return X, y
    return X


# %%
testSplitDate = '2019-12-01'
df_train = df[df["Date"] <= testSplitDate].copy()
df_test = df[df["Date"] > testSplitDate].copy()

X_train, y_train = create_features(df_train, label='Prices', shift =1)
X_test, y_test = create_features(df_test, label='Prices', shift =1)
X_train = X_train.iloc[1:]
X_test = X_test.iloc[1:]
y_train = y_train.iloc[1:]
y_test = y_test.iloc[1:]


# %%
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
plt.style.use('fivethirtyeight')

model = XGBRegressor(
    n_estimators=1000,
    #max_depth=8,
    #min_child_weight=300, 
    #colsample_bytree=0.8, 
    #subsample=0.8, 
    #eta=0.3,    
    #seed=42
    )

model.fit(
    X_train, 
    y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, y_train), (X_test, y_test)], 
    verbose=False, 
    early_stopping_rounds = 100)


# %%
model.feature_importances_


# %%
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()


# %%
correl = X_train.corr()
sns.heatmap(correl, xticklabels=correl.columns, yticklabels=correl.columns)
plt.tight_layout()
plt.show()


# %%
correl


# %%
df_test = df_test.iloc[1:]
df_train = df_train.iloc[1:]
df_test['WTI_Prediction'] = model.predict(X_test)
df_all = pd.concat([df_test, df_train], sort=False)
df_all = df_all.set_index("Date")
_ = df_all[df_all.index > '2019-11-01'][['Prices','WTI_Prediction']].plot(figsize=(15, 5))


# %%
X_new = df[df["Date"]>'2020-01-09'].copy()
X_new = create_features(X_new)
X_new


# %%
new_pred = model.predict(X_new)
new_pred


# %%
df["Prices"].iloc[-1]


# %%
new_pred > df["Prices"].iloc[-1]


# %%
# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = df_all[['WTI_Prediction','Prices']].plot(ax=ax,style=['-','.'])
ax.set_xbound(lower='2019-12-01', upper='2019-12-10')
ax.set_ylim(54, 60)
plot = plt.suptitle('Dec 2019 1Week Forecast vs Actuals')


# %%
mean_squared_error(y_true=df_test['Prices'],
                   y_pred=df_test['WTI_Prediction'])


# %%

testSplitDate = '2020-01-03'
fin_df_train = df[df["Date"] <= testSplitDate].copy()
fin_df_test = df[df["Date"] > testSplitDate].copy()

X_train, y_train = create_features(fin_df_train, label='Prices')
X_test, y_test = create_features(fin_df_test, label='Prices')

model = XGBRegressor(
    n_estimators=1000,
    #max_depth=8,
    #min_child_weight=300, 
    #colsample_bytree=0.8, 
    #subsample=0.8, 
    #eta=0.3,    
    #seed=42
    )

model.fit(
    X_train, 
    y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, y_train), (X_test, y_test)], 
    verbose=False, 
    early_stopping_rounds = 100)


# %%
from sklearn.externals import joblib

modDate = str(fin_df_train["Date"].iloc[-1].strftime('%Y-%m-%d'))
filename = 'finalized_model_'+modDate+'.sav'
joblib.dump(model, filename)


# %%
fin_df_test['WTI_Prediction'] = model.predict(X_test)
fin_df_all = pd.concat([fin_df_test, fin_df_train], sort=False)
fin_df_all = fin_df_all.set_index("Date")
_ = fin_df_all[fin_df_all.index > '2019-12-01'][['Prices','WTI_Prediction']].plot(figsize=(15, 5))



