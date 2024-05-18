import ib_insync
from ib_insync import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
import pandas as pd
# util.startLoop()  # uncomment this line when in a notebook
print("Hello")
ib = IB()
ib.connect('127.0.0.1', 7597, clientId=0)

#contract = Stock('INTC','SMART','USD', primaryExchange='NASDAQ')
#contract = Contract(conId=481691285)
contract = Stock('COIN','SMART','USD')
#contract = Stock('QQQ','SMART','USD')
#contract = Forex('EURUSD') 
bars1 = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='3 D',
    barSizeSetting='5 mins', whatToShow='MIDPOINT', useRTH=False)

contract = Stock('QQQ','SMART','USD')
#contract = Forex('EURUSD')
bars2 = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='3 D',
    barSizeSetting='5 mins', whatToShow='MIDPOINT', useRTH=False)

# convert to pandas dataframe:
df1 = util.df(bars1)
date1=df1[['date']]
open1=df1[['open']]

df2 = util.df(bars2)
date2=df2[['date']]
open2=df2[['open']]
start_every_day = datetime.datetime(2022,1,1,4,0,0,0)
end_every_day   = datetime.datetime(2022,1,1,20,0,0,0)
timeToStart=(abs(start_every_day-date1['date'])%86400000000000)<np.timedelta64(30,'m')
timeToEnd=abs(end_every_day-date1['date'])%86400000000000<np.timedelta64(30,'m')

timeToEnd = np.delete([timeToEnd],np.where(timeToStart)[0],None)
date1 = np.delete([date1],np.where(timeToStart)[0],None)
date2 = np.delete([date2],np.where(timeToStart)[0],None)
open1 = np.delete([open1],np.where(timeToStart)[0],None)
open2 = np.delete([open2],np.where(timeToStart)[0],None)

timeToStart = np.delete([timeToStart],np.where(timeToEnd)[0],None)
date1 = np.delete([date1],np.where(timeToEnd)[0],None)
date2 = np.delete([date2],np.where(timeToEnd)[0],None)
open1 = np.delete([open1],np.where(timeToEnd)[0],None)
open2 = np.delete([open2],np.where(timeToEnd)[0],None)

open1=pd.DataFrame(open1)
open2=pd.DataFrame(open2)
print(open1.size)
print(open2.size)


# SK Learn Linear Regression

lrModel = LinearRegression()
lrModel.fit(open1,open2)
print("Score=",lrModel.score(open1,open2))
print("b=",lrModel.intercept_[0])
print("k=",lrModel.coef_[0][0])
open2x=lrModel.predict(open1)


# plot data
plt.style.use('seaborn')

plt.subplot(2,1,1)
plt.plot_date(date1,open2,'-')
plt.plot_date(date1,open2x,'-')
plt.xlabel('Time')
plt.ylabel('QQQ/Linear COIN')
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot_date(date1,open2-open2x,'-')
plt.xlabel('Time')
plt.ylabel('QQQ-Linear COIN')
plt.tight_layout()

plt.show()