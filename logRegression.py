
def SMA(period, data):
    sma = data.rolling(window=period).mean()
    return sma


combined.head(25)
len(combined["Prices"][combined["Prices"].isnull()])
combined["20dSMA"][combined["20dSMA"].isnull()]

test = combined["Prices"]
tes = test.dropna()


combined["20dSMA"] = SMA(20, tes)
combined["10dSMA"] = SMA(10, tes)
combined["5dSMA"] = SMA(5, tes)
combined["50dSMA"] = SMA(50, tes)
combined["200dSMA"] = SMA(200, tes)


def bolli_bois(data):

    mid_band = SMA(20, data)
    std_dev = np.std(mid_band)
    up_band = mid_band + 2*std_dev
    low_band = mid_band - 2*std_dev
    return low_band, up_band


combined["bollinger_low"] = bolli_bois(combined["20dSMA"])[0]
combined["bollinger_high"] = bolli_bois(combined["20dSMA"])[1]


show_more(combined, 300)

plot2axis(combined["Date"], combined["Prices"].astype(
    float), combined["Production of Crude Oil"], "Date", "Price (USD)",
    'Production of Crude Oil (Thousand Barrels per Day)', lineax1=True,
    lineax1y=combined["20dSMA"], lineax1name="20d SMA",
    fill_boll=True, bol_high=combined["bollinger_high"],
    bol_low=combined["bollinger_low"])
