from ib_insync import *
import matplotlib as mpl
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
from datetime import timedelta
import pandas as pd
util.startLoop()  # uncomment this line when in a notebook

#Connection Establish
ib = IB()
ib.connect('127.0.0.1', 7597, clientId=1)

#Select Data Set
contract = Contract()
# contract.symbol = "NDX"
# contract.secType = "IND"
# contract.currency = "USD"
# contract.exchange = "NASDAQ"
contract.symbol = "MNQ"
contract.secType = "FUT"
contract.exchange = "CME"
contract.currency = "USD"
contract.lastTradeDateOrContractMonth = "202406"

#ib.reqMarketDataType(3)

#Take History Data
bars = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='30 Y',
    barSizeSetting='1 day', whatToShow='TRADES', useRTH=False)

df = util.df(bars)

#Save Data
df.to_pickle('MNQtraindata')

#Show Data in Open Price
print(df.head())
print(df.tail())
date=df[['date']]
open=df[['open']]
date['date']=(date['date']-date['date'][0])/ timedelta(minutes=1)
date['date'].astype("int")
plt.plot(date,open,'-')
plt.xlabel('Time')
plt.ylabel('MNQ')
plt.tight_layout()
plt.show()
