import predictDeps

priceDF = dataFun.dataHub(
    url="https://datahub.io/core/oil-prices/r/wti-daily.csv", import_new_data=True)
oilDF = dataFun.oilProduction()
df = dataFun.combineFrames(priceDF, oilDF)
df = df[np.isfinite(df['Prices'])]

df = df.sort_values(by=['Date'])
df = df.reset_index().drop(["index"], axis=1)

today = dt.today()
defaultDay = (today + timedelta(days=1)).strftime('%Y-%m-%d')
# df2 = pd.DataFrame


print("Please enter the date you want have a prediction for (format: 'yyyy-mm-dd')")

while True:
    try:
        predDate = str(input())
        datey = dt.strptime(predDate, '%Y-%m-%d').strftime('%Y-%m-%d')
        if datey > defaultDay:
            print(
                "Input date too far in future, currently only 1 day into the future is predictable")
            raise ValueError
        if datey < '2000-01-01':
            print("Input date too far in the past, enter a more recent date")
            raise ValueError

    except(ValueError):
        try:
            print("Date {} entered is not suitable".format(datey))
            agree = "n"

            print("Do you want to default to {}? (Y/N)".format(defaultDay))

            while True:
                try:
                    agree = str(input()).lower()

                    if agree == "y":
                        datey = defaultDay
                        break

                    elif agree == "n":
                        break

                    else:
                        print("Please choose Y or N!")

                except ValueError:
                    print("Please choose Y or N!")

        except(NameError):
            print("Please enter a date in correct format (yyyy-mm-dd)!")
            continue

        if agree == "y":
            break
        elif agree == "n":
            print(
                "Please enter the date you want have a prediction for (format: 'yyyy-mm-dd')")
            continue
    else:
        break

print("Please enter the current ({}) WTI Oil Price".format(
    (dt.strptime(datey, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')))

while True:
    try:
        newPrice = float(input())
    except(ValueError):
        print("Please enter a number")
        continue
    else:
        break

if datey < defaultDay:
    try:
        newProd = float(df[df["Date"] == datey]['Production of Crude Oil'])
    except(TypeError):
        try:
            newday = (dt.strptime(datey, '%Y-%m-%d') -
                      timedelta(days=1)).strftime('%Y-%m-%d')
            newProd = float(df[df["Date"] == newday]
                            ['Production of Crude Oil'])
        except(TypeError):
            try:
                newday = (dt.strptime(datey, '%Y-%m-%d') -
                          timedelta(days=2)).strftime('%Y-%m-%d')
                newProd = float(df[df["Date"] == newday]
                                ['Production of Crude Oil'])
            except(TypeError):
                try:
                    newday = (dt.strptime(datey, '%Y-%m-%d') -
                              timedelta(days=23)).strftime('%Y-%m-%d')
                    newProd = float(df[df["Date"] == newday]
                                    ['Production of Crude Oil'])
                except(TypeError):
                    newProd = df['Production of Crude Oil'].iloc[-1]
else:
    newProd = df['Production of Crude Oil'].iloc[-1]

df = df[df["Date"] < datey]

df = df.append({"Date": datey, "Prices": newPrice,
                "Production of Crude Oil": newProd}, ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"])


def SMA(period, data):
    sma = data.rolling(window=period).mean()
    return sma


def ema(data, window=20):

    exp = data.ewm(span=window, adjust=False).mean()
    return exp


def bollinger(data, window=20):

    mid_band = SMA(window, data)
    std_dev = data.rolling(window=window).std()
    up_band = mid_band + 2*std_dev
    low_band = mid_band - 2*std_dev
    return low_band, up_band


def momentum(df, n):
    """
    :param df: pandas.DataFrame 
    :param n: 
    :return: pandas.DataFrame
    """
    M = pd.Series(df['Prices'].diff(n), name='Momentum_' + str(n))
    df = df.join(M)
    return df


def relative_strength_index(df, n=14):
    """Calculate Relative Strength Index(RSI) for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = df.index.min()
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        # UpMove = df.loc[i + 1, 'High'] - df.loc[i, 'High']
        # DoMove = df.loc[i, 'Low'] - df.loc[i + 1, 'Low']
        Move = df.loc[i, 'Prices'] - df.loc[i + 1, 'Prices']

        if Move > 0:
            UpD = Move
        else:
            UpD = 0
        UpI.append(UpD)
        if Move < 0:
            DoD = Move
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    df = df.join(RSI)
    return df


def rate_of_change(df, n):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    M = df['Prices'].diff(n - 1)
    N = df['Prices'].shift(n - 1)
    ROC = pd.Series(M / N, name='ROC_' + str(n))
    df = df.join(ROC)
    return df


def macd(df, n_fast, n_slow):
    """Calculate MACD, MACD Signal and MACD difference
    
    :param df: pandas.DataFrame
    :param n_fast: 
    :param n_slow: 
    :return: pandas.DataFrame
    """
    EMAfast = pd.Series(df['Prices'].ewm(
        span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df['Prices'].ewm(
        span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' +
                     str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(
    ), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' +
                         str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df


df["20dSMA"] = SMA(20, df["Prices"])
df["200dSMA"] = SMA(20, df["Prices"])
df["boll_lo"] = bollinger(df['Prices'])[0]
df["boll_hi"] = bollinger(df['Prices'])[1]

df = momentum(df, 14)
df = macd(df, 12, 26)
df = rate_of_change(df, 14)
df = relative_strength_index(df)

df["boll_hi"] = pd.to_numeric(df["boll_hi"])
df["boll_lo"] = pd.to_numeric(df["boll_lo"])
df["20dSMA"] = pd.to_numeric(df["20dSMA"])
df["200dSMA"] = pd.to_numeric(df["200dSMA"])

df["bollAmplitude"] = df["boll_hi"] - df["boll_lo"]
df["distFromTopBoll"] = df["boll_hi"] - df["Prices"]
df["distFromLowBoll"] = df["boll_lo"] - df["Prices"]
df["20d200dDist"] = np.abs(df["20dSMA"] - df["200dSMA"])

try:
    filename = sys.argv[1]
except(NameError):
    print("Specified model does not exist")
    abspath = str(os.getcwd()+"/*.sav")
    try:
        filename = glob.glob(abspath)[-1]
        print("Using latest model file {}".format(filename))
    except(NameError):
        print("Model does not exist, exiting")
except(IndexError):
    print("No model specified")
    abspath = str(os.getcwd()+"/*.sav")
    try:
        filename = glob.glob(abspath)[-1]
        print("Using latest model file {}".format(filename))
    except(NameError):
        print("Model does not exist, exiting")

model = joblib.load(filename)
#result = model.score(X_test, Y_test)
#print(result)
x_test = df.tail(1)
x_test = x_test.drop(["200dSMA", "Date",
                      "Prices", "boll_lo", "boll_hi", "MACDsign_12_26"], axis=1)

pred = model.predict(x_test)
proba = model.predict_proba(x_test)

if pred[0] == 0:
    print("The price of oil is likely to decrease tomorrow with probability {:.1%}".format(
        proba[0][0]))
else:
    print("The price of oil is likely to increase tomorrow with probability {:.1%}".format(
        proba[0][1]))
