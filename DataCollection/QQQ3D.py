from ib_insync import *
import matplotlib as mpl
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
from datetime import timedelta
import pandas as pd
from utils import QQQXD
util.startLoop()  # uncomment this line when in a notebook

#Connection Establish
ib = IB()
ib.connect('127.0.0.1', 7597, clientId=1)

#Select Data Set
contract = Contract()
contract  = Stock('QQQ','SMART','USD')

df = QQQXD(ib,'20240506',7,'1 hour')

print(df.head(30))
df.to_csv('datatest.csv')
