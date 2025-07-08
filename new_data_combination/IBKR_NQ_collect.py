from ib_insync import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from datetime import datetime
import pandas as pd
from tqdm import tqdm

ib = IB()
ib.connect('127.0.0.1', 7597, clientId=1)
ib.reqMarketDataType(1)

def request_many_day_NQ(daysN):
    now=datetime.now()
    dfs = pd.DataFrame(columns=['date','open','high','low','close','volume'])
    dfs.set_index('date', inplace=True)
    for day in tqdm(range(daysN), desc='Processing days'):
        endtime= now - timedelta(days=day)
        contract = Future(symbol='NQ',exchange='CME',currency='USD',lastTradeDateOrContractMonth='202506')
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
        dfs = pd.concat([df, dfs])
    dfs = dfs[~dfs.index.duplicated(keep='last')]
    return dfs

def request_1_day_NQ(year,season):
    monthstr=calculate_contract_month_symbol(year,season)
    contract  = Future(symbol='NQ',exchange='CME',currency='USD',lastTradeDateOrContractMonth=monthstr)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=datetime.now(),    # 结束时间：现在
        durationStr='7 D',             # 向前 7 天
        barSizeSetting='1 min',        # 1 分钟 K 线
        whatToShow='TRADES',           # 显示成交数据，也可以用 'MIDPOINT','BID','ASK' 等
        useRTH=False,                  # 包括盘前盘后（如只要正常交易时段，设为 True）
        formatDate=1                   # 返回的 date 字段为 Python datetime
    )
    df = util.df(bars)[['date', 'open', 'high', 'low', 'close', 'volume']]
    df.set_index('date', inplace=True)
    return df

def request_10_min_NQ(year,season):
    monthstr=calculate_contract_month_symbol(year,season)
    contract  = Future(symbol='NQ',exchange='CME',currency='USD',lastTradeDateOrContractMonth=monthstr)
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
    return df

def calculate_contract_month_symbol(year,season):
    str1=str(year)
    match season:
        case 0:
            return str1+'03'
        case 1:
            return str1+'06'
        case 2:
            return str1+'09'
        case 3:
            return str1+'12'
        case _:
            return str1+'03'

def main():
    df=request_1_day_NQ(2025,1)
    # 6. 打印或返回
    print(df.head())
    print(df.tail())
    print(len(df))

if __name__ == '__main__':
    main()