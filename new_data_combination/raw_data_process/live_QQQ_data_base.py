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
import asyncio

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)
ib.reqMarketDataType(1)

class qqq_live_t0:
    def __init__(self,ib):
        #初始化
        self.ibob=ib
        self.loop = asyncio.get_event_loop()
        self.sync_param()
        self.live_change=0 #是否发生在线状态下的合约转变
        self.last_minute_contract_num=self.current_num
        self.load_QQQ_harddisk()
        self.sync_QQQ_base()

    def request_many_day_QQQ(self,daysN):
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

    def request_1_day_QQQ(self):
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

    def request_many_min_QQQ(self,minute):
        lengthstr=str((minute+3)*60)
        contract = Stock('QQQ', 'NASDAQ', 'USD')
        task = self.loop.create_task(self.ibob.reqHistoricalData(
            contract,
            endDateTime=datetime.now(ZoneInfo('America/New_York')),    # 结束时间：现在
            durationStr=lengthstr+' S',             # 向前 10分钟
            barSizeSetting='1 min',        # 1 分钟 K 线
            whatToShow='TRADES',           # 显示成交数据，也可以用 'MIDPOINT','BID','ASK' 等
            useRTH=False,                  # 包括盘前盘后（如只要正常交易时段，设为 True）
            formatDate=1                   # 返回的 date 字段为 Python datetime
        ))
        bars = self.loop.run_until_complete(task)
        df = util.df(bars)[['date', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('date', inplace=True)
        df.index.rename('datetime', inplace=True)
        return df

    def fast_concat(self,main_data_base,new_data_base): #在大数据集main_data_base下方拼接new_data_base，并去掉重复部分，main和new都必须是时间升序排序完成的
        # 1. 找到 new_data_base 中最小（即最早）的索引：
        first_new_idx = new_data_base.index[0]
        # 2. 在 main_data_base 的索引上做二分查找，定位到第一个 >= first_new_idx 的位置
        #    这就是所有可能重复的第一行
        pos = main_data_base.index.searchsorted(first_new_idx, side='left')
        # 3. 切片：只保留 main_data_base 中索引 < first_new_idx 的那部分
        main_data_base = main_data_base.iloc[:pos]
        # 4. 直接上下拼接
        main_data_base = pd.concat([main_data_base, new_data_base])
        return main_data_base
    
    def sync_param(self):
        global live_data_base
        self.now=datetime.now(ZoneInfo('America/New_York'))
        #计算出当前期的期货合约和下个季度的期货合约
        self.current_file_str=self.format_contract(self.current_year,self.current_season)
        self.current_IBKR_tick_str=self.calculate_contract_month_symbol(self.current_year,self.current_season)
        #检查路径和文件是否存在
        self.QQQ_type0_path=live_data_base+'/type0/NQ/'
        self.current_filename = 'QQQ_BASE.pkl'

    def load_QQQ_harddisk(self):
            #先处理当前季度合约
            fullpath = os.path.join(self.QQQ_type0_path, self.current_filename)
            if os.path.isfile(fullpath):
                self.QQQBASE=pd.read_pickle(self.QQQ_type0_path+self.current_filename)


    def sync_QQQ_base(self):
        last_BASE_time=self.QQQBASE.index[-1]
        now = datetime.now()
        delta = now - last_BASE_time    # 这是一个 timedelta 对象
        days = max(delta.days, 0)+1 # .days 已经是向下取整的天数，负数就算 0
        df=self.request_many_day_QQQ(days)
        merged=self.fast_concat(self.QQQBASE, df)
        #merged重新赋值回QQQBASE
        self.QQQBASE=merged
        merged.to_pickle(live_data_base+'/type0/QQQ/QQQ_BASE1.pkl')
        #merged.to_pickle(self.QQQ_type0_path+self.current_filename)

    def minute_march(self):#每分钟需要做的事情
        self.sync_param()
        last_BASE_time=self.QQQBASE.index[-1]
        delta = self.now - last_BASE_time.replace(tzinfo=ZoneInfo('America/New_York'))    # 这是一个 timedelta 对象
        minute=int(delta.total_seconds() // 60)+2
        self.current_contract_data=self.fast_concat(self.QQQBASE,self.request_many_min_QQQ(minute,self.current_num))

    def fast_march(self,datetime_,open_,high_,low_,close_,volume_):
        # 这个函数快速录入当前数据，不需要激活request history，只有在发现数据不连续时再动用request history函数用于核对
        # 1) 把这一根 Bar 构造成只有一行的小 DataFrame，
        #    索引用 bar_datetime，列名必须和 self.current_contract_data 一致
        new_row = pd.DataFrame(
            [[open_, high_, low_, close_, volume_]],
            index=[datetime_],
            columns=['open', 'high', 'low', 'close', 'volume']
        )
        new_row.index.name = 'datetime'  # 如果你的 current_contract_data.index 名称也是 'datetime'

        # 2) 用 concat 拼接到原 DataFrame 底部
        self.QQQBASE = self.fast_concat([self.QQQBASE, new_row])




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