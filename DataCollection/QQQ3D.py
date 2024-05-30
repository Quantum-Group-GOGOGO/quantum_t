from ib_insync import *
import matplotlib as mpl
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from utils import HistoricalDataCollector
util.startLoop()  # uncomment this line when in a notebook
from config import parse_args
#Connection Establish
def calculate_previous_date(date: str, num_days: int) -> str:
    """
    计算并返回给定日期前num_days的日期，格式为"YYYYMMDD"。

    :param date: 字符串形式的日期，格式为"YYYYMMDD"
    :param num_days: 整数，表示要回溯的天数
    :return: 回溯num_days天后的日期，格式为"YYYYMMDD"
    """
    # 将输入的字符串日期转换为datetime对象
    date_obj = datetime.strptime(date, "%Y%m%d")
    
    # 计算前num_days的日期
    previous_date_obj = date_obj - timedelta(days=num_days)
    
    # 将计算得到的日期转换回"YYYYMMDD"格式的字符串
    previous_date_str = previous_date_obj.strftime("%Y%m%d")
    
    return previous_date_str
def main():

    args = parse_args()
    ib = IB()
    ib.connect('127.0.0.1', 7597, clientId=1)

    if args.contract_symbol == 'QQQ':
        args.secType = "STK"
        args.exchange = "NASDAQ"
        args.currency = "USD"
    elif args.contract_symbol == 'NDX':
        args.secType = "IND"
        args.exchange = "NASDAQ"
        args.currency = "USD"
    elif args.contract_symbol == 'MNQ':
        args.secType = "FUT"
        args.exchange = "CME"
        args.currency = "USD"
        args.lastTradeDateOrContractMonth = "202409"  
    else:
        args.secType = "IND"
        args.exchange = "SMART"
        args.currency = "USD"       
    #Select Data Set
    first_time = True
    # import pdb;pdb.set_trace()
    all_dfs = []
    for i in tqdm(range(1)):
        DataCollector = HistoricalDataCollector(IBobject=ib, args=args)
        df = DataCollector.collect_historical_data(num_days=args.num_days)
        if df.empty:
            continue
        if first_time:
            df.to_csv(f'{args.contract_symbol}_1week_per_day.csv', mode='w', index=False)  # 首次写入，包括表头
            first_time = False
        else:
            df.to_csv(f'{args.contract_symbol}_1week_per_day.csv', mode='a', header=False, index=False)  # 后续迭代，追加数据，不写表头
        args.date = calculate_previous_date(args.date, args.num_days)
        print(i,args.date)
        print(df.head(5))
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_pickle(f'{args.contract_symbol}_1week_per_day.pickle')    

if __name__ == '__main__':
    main()