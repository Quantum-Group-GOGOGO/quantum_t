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
def min_max_t1_t4(df):
    """
    计算 df 中每行 t1 - t4 的值，并返回全局的最小值和最大值。

    参数
    ----
    df : pd.DataFrame
        必须包含两列：'t1' 和 't4'，它们均为 datetime 或数值类型。

    返回
    ----
    tuple
        (min_diff, max_diff)，与 df['t1']-df['t4'] 同类型：
        - 如果是时间差，则返回 pd.Timedelta；
        - 如果是数值，则返回数值。
    """
    # 1. 计算差值序列
    diff = df['t4'] - df['t1']
    # 2. 取最小和最大
    min_diff = diff.min()
    max_diff = diff.max()
    return min_diff, max_diff
#ib = IB()
#ib.connect('127.0.0.1', 4002, clientId=2)
#ib.reqMarketDataType(1)

# 1. 从 pkl 文件中读取 DataFrame
#df = pd.read_pickle(live_data_base+'/type0/QQQ/'+'QQQ_BASE.pkl')
#df.to_pickle(live_data_base+'/type0/QQQ/'+'QQQ_BASE.pkl')
#df = pd.read_pickle(data_base+'/type2/Nasdaq_qqq_align_base.pkl')
df = pd.read_pickle('data41.pkl')

min_diff, max_diff = min_max_t1_t4(df)
print("最小差值：", min_diff)  # 0:45:00
print("最大差值：", max_diff)  # 1:15:00
# 2. 保存为 CSV 文件（保留行索引）
#segment.to_csv('data.csv', index=True)

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