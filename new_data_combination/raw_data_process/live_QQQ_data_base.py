from ib_insync import *
from env import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from datetime import datetime
import pandas as pd
from tqdm import tqdm

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)
ib.reqMarketDataType(1)

def request_many_day_QQQ(daysN):
    now=datetime.now()
    dfs = pd.DataFrame(columns=['datetime','open','high','low','close','volume'])
    dfs.set_index('datetime', inplace=True)
    for day in tqdm(range(daysN), desc='Processing days'):
        endtime= now - timedelta(days=day)
        contract = Stock('QQQ', 'NASDAQ', 'USD')
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=endtime,    # 结束时间：现在
            durationStr='1 D',             # 向前 7 天
            barSizeSetting='1 min',        # 1 分钟 K 线
            whatToShow='TRADES',           # 显示成交数据，也可以用 'MIDPOINT','BID','ASK' 等
            useRTH=False,                  # 包括盘前盘后（如只要正常交易时段，设为 True）
            formatDate=1                   # 返回的 date 字段为 Python datetime
        )
        df = util.df(bars)[['date', 'open', 'high', 'low', 'close', 'volume']].set_index('date')
        dfs.index.rename('datetime', inplace=True)
        df.sort_index(ascending=True, inplace=True)
        if dfs.empty:
            dfs = df
        else:
            dfs = pd.concat([df, dfs])
    dfs = dfs[~dfs.index.duplicated(keep='last')]
    return dfs

def request_1_day_QQQ():
    contract = Stock('QQQ', 'NASDAQ', 'USD')
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=datetime.now(),    # 结束时间：现在
        durationStr='1 D',             # 向前 7 天
        barSizeSetting='1 min',        # 1 分钟 K 线
        whatToShow='TRADES',           # 显示成交数据，也可以用 'MIDPOINT','BID','ASK' 等
        useRTH=False,                  # 包括盘前盘后（如只要正常交易时段，设为 True）
        formatDate=1                   # 返回的 date 字段为 Python datetime
    )
    df = util.df(bars)[['date', 'open', 'high', 'low', 'close', 'volume']]
    df.set_index('date', inplace=True)
    df.index.rename('datetime', inplace=True)
    return df

def request_10_min_QQQ():
    contract = Stock('QQQ', 'NASDAQ', 'USD')
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=datetime.now(),    # 结束时间：现在
        durationStr='600 S',             # 向前 10分钟
        barSizeSetting='1 min',        # 1 分钟 K 线
        whatToShow='TRADES',           # 显示成交数据，也可以用 'MIDPOINT','BID','ASK' 等
        useRTH=False,                  # 包括盘前盘后（如只要正常交易时段，设为 True）
        formatDate=1                   # 返回的 date 字段为 Python datetime
    )
    df = util.df(bars)[['date', 'open', 'high', 'low', 'close', 'volume']]
    df.set_index('date', inplace=True)
    df.index.rename('datetime', inplace=True)
    return df

def sync_QQQ_base():
    QQQ_type0_path=live_data_base+'/type0/QQQ/QQQ_BASE.pkl'
    QQQBASE= pd.read_pickle(QQQ_type0_path)
    print(len(QQQBASE))
    last_BASE_time=QQQBASE.index[-1]
    now = datetime.now()
    delta = now - last_BASE_time    # 这是一个 timedelta 对象
    days = max(delta.days, 0)+1 # .days 已经是向下取整的天数，负数就算 0
    df=request_many_day_QQQ(days)
    #合并QQQBASE和df,重复datetime（索引列）项目则保留df中的值
    # 1. 找到 df 中最小（即最早）的索引：
    first_new_idx = df.index[0]
    # 2. 在 QQQBASE 的索引上做二分查找，定位到第一个 >= first_new_idx 的位置
    #    这就是所有可能重复的第一行
    pos = QQQBASE.index.searchsorted(first_new_idx, side='left')
    # 3. 切片：只保留 QQQBASE 中索引 < first_new_idx 的那部分
    base_trimmed = QQQBASE.iloc[:pos]
    # 4. 直接上下拼接
    print(base_trimmed.head())
    print(df.head())
    merged = pd.concat([base_trimmed, df])
    #merged重新赋值回QQQBASE
    merged.to_pickle(live_data_base+'/type0/QQQ/QQQ_BASE1.pkl')
    print(len(merged))

def main():
    # 1. 连接 IB Gateway / TWS
    

    #args.contract_symbol = 'QQQ'
    #args.secType = "STK"
    #args.exchange = "NASDAQ"
    #args.currency = "USD"

    
    # 2. 定义合约：QQQ 在 SMART 交易所，交易货币 USD
    
    # 5. （可选）把 date 列设置为索引
    #df.set_index('date', inplace=True)
    #df=request_1_day_QQQ()
    df=request_10_min_QQQ()
    sync_QQQ_base()
    # 6. 打印或返回
    print(df.head())
    print(df.tail())

if __name__ == '__main__':
    main()