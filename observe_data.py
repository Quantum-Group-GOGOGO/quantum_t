from ib_insync import *
from env import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from zoneinfo import ZoneInfo

#ib = IB()
#ib.connect('127.0.0.1', 4002, clientId=2)
#ib.reqMarketDataType(1)

# 1. 从 pkl 文件中读取 DataFrame
#df = pd.read_pickle(live_data_base+'/type0/QQQ/'+'QQQ_BASE.pkl')
#df.to_pickle(live_data_base+'/type0/QQQ/'+'QQQ_BASE.pkl')
df = pd.read_pickle(live_data_base+'/type0/NQ/'+'NQBASE2025U.pkl')
# 2. 保存为 CSV 文件（保留行索引）
df.to_csv('data.csv', index=True)

#contract = Future(symbol='NQ',exchange='CME',currency='USD',lastTradeDateOrContractMonth='202509')
#endtime=datetime.now()
#endtime.replace(tzinfo=ZoneInfo('America/New_York'))
#bars = ib.reqHistoricalData(
#    contract,
#    endDateTime=endtime.replace(tzinfo=ZoneInfo('America/New_York')),    # 结束时间：现在
#    durationStr='86400 S',             # 向前 7 天
#    barSizeSetting='1 min',        # 1 分钟 K 线
#    whatToShow='TRADES',           # 显示成交数据，也可以用 'MIDPOINT','BID','ASK' 等
#    useRTH=False,                  # 包括盘前盘后（如只要正常交易时段，设为 True）
#    formatDate=1                   # 返回的 date 字段为 Python datetime
#)

            
#df = util.df(bars)[['date', 'open', 'high', 'low', 'close', 'volume']].set_index('date')
#df.index.rename('datetime', inplace=True)
#df.sort_index(ascending=True, inplace=True)

#print(df.head())
#print(df.tail())