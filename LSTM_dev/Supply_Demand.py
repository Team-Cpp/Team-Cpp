import requests
import numpy as np
import pandas as pd


#Energy usage in kg Oil per Capita split by year and country
url = 'https://datahub.io/world-bank/eg.use.pcap.kg.oe/r/data.csv'
r = requests.get(url)
data =  r.content


data = data.replace('"','') 
lines = data.splitlines()
Country = []
Year = []
Oil = []
for line in lines:
    line2 =  line.split(',')
    Country.append(line2[0])
    Year.append(line2[-2])
    Oil.append(line2[-1])


Country.pop(0)
Year.pop(0)
Oil.pop(0)
Energy_use = pd.DataFrame(np.zeros([len(Year),3]),columns=['Country','Year', 'Kg Oil Per Capita'])
new_df1 = pd.DataFrame({'Year': Year})
Energy_use.update(new_df1)
new_df2 = pd.DataFrame({'Country': Country})
Energy_use.update(new_df2)
new_df3 = pd.DataFrame({'Kg Oil Per Capita': Oil})
Energy_use.update(new_df3)


#US Oil Production in 1000 barrels per day
url2 = 'https://www.eia.gov/dnav/pet/hist_xls/WCRFPUS2w.xls'
r2 = requests.get(url2)
data_xls = pd.read_excel(url2, 'Data 1', skiprows = 2, comment='#') 
data_xls.columns = ['Date','Production of Crude Oil']
Oil_Production = data_xls

print Energy_use
print Oil_Production



