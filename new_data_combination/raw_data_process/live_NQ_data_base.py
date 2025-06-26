from ib_insync import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import recording_time_trigger as rtt
import os

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)
ib.reqMarketDataType(1)
class nq_live_t0:
    def __init__(self):
        #初始化
        self.sync_param()
        self.live_change=0 #是否发生在线状态下的合约转变
        self.last_minute_contract_num=self.current_num
        self.load_NQ_harddisk()
        self.sync_NQ_base()
        

    def request_many_day_NQ(self,daysN,num):
        monthstr=self.calculate_contract_month_symbol_by_int(num)
        now=datetime.now()
        dfs = pd.DataFrame(columns=['datetime','open','high','low','close','volume'])
        dfs.set_index('datetime', inplace=True)
        for day in tqdm(range(daysN), desc='Processing days'):
            endtime= now - timedelta(days=day)
            contract = Future(symbol='NQ',exchange='CME',currency='USD',lastTradeDateOrContractMonth=monthstr)
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
            df.index.rename('datetime', inplace=True)
            df.sort_index(ascending=True, inplace=True)
            if dfs is None:
                dfs = df
            else:
                dfs = pd.concat([df, dfs])
        dfs = dfs[~dfs.index.duplicated(keep='last')]
        return dfs

    def request_1_day_NQ(self,num):
        monthstr=self.calculate_contract_month_symbol_by_int(num)
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
        df.index.rename('datetime', inplace=True)
        return df

    def request_10_min_NQ(self,num):
        monthstr=self.calculate_contract_month_symbol_by_int(num)
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
        df.index.rename('datetime', inplace=True)
        return df

    def request_many_min_NQ(self,minute,num):
        monthstr=self.calculate_contract_month_symbol_by_int(num)
        lengthstr=str((minute+3)*60)
        contract  = Future(symbol='NQ',exchange='CME',currency='USD',lastTradeDateOrContractMonth=monthstr)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=datetime.now(),    # 结束时间：现在
            durationStr=lengthstr+' S',             # 向前 10分钟
            barSizeSetting='1 min',        # 1 分钟 K 线
            whatToShow='TRADES',           # 显示成交数据，也可以用 'MIDPOINT','BID','ASK' 等
            useRTH=False,                  # 包括盘前盘后（如只要正常交易时段，设为 True）
            formatDate=1                   # 返回的 date 字段为 Python datetime
        )
        df = util.df(bars)[['date', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('date', inplace=True)
        df.index.rename('datetime', inplace=True)
        return df
    
    def yearseason_to_int(self,year,season):
        number=(year-2000)*4+season
        return number

    def int_to_yearseason(self,number):
        year=2000+(number//4)
        season=number%4
        return year,season

    def calculate_contract_month_symbol(self,year,season):
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

    def calculate_contract_month_symbol_by_int(self,number):
        year=2000+(number//4)
        season=number%4
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

    def calculate_current_contract_year_season(self,now):
        if now.month==1 or now.month==2:
            return now.year,0
        elif now.month==4 or now.month==5:
            return now.year,1
        elif now.month==7 or now.month==8:
            return now.year,2
        elif now.month==10 or now.month==11:
            return now.year,3
        else:
            season=(now.month//3)-1
            if rtt.is_trigger_day_pass():
                season=season+1
            if season>3:
                return now.year+1,0
            else:
                return now.year,season

    def format_contract(self,year: int, season: int) -> str:
        """
        根据年份和季节序号生成期货合约代码。
        
        参数：
        - year: 4 位年份，如 2021
        - season: 季节序号，0->H, 1->M, 2->U, 3->Z
        
        返回值：
        - 合约代码字符串，例如 "2021H"
        
        抛出：
        - ValueError: 当 season 不在 [0,1,2,3] 时
        """
        season_map = {0: 'H', 1: 'M', 2: 'U', 3: 'Z'}
        
        if season not in season_map:
            raise ValueError(f"无效的季节序号: {season}，应为 0, 1, 2 或 3")
        
        return f"{year}{season_map[season]}"

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
        self.now=datetime.now()
        self.current_year,self.current_season=self.calculate_current_contract_year_season(self.now)
        self.current_num=self.yearseason_to_int(self.current_year,self.current_season)
        self.next_num=self.current_num+1
        self.next_year,self.next_season=self.int_to_yearseason(self.next_num)
        #计算出当前期的期货合约和下个季度的期货合约
        self.current_file_str=self.format_contract(self.current_year,self.current_season)
        self.next_file_str=self.format_contract(self.next_year,self.next_season)
        self.current_IBKR_tick_str=self.calculate_contract_month_symbol(self.current_year,self.current_season)
        self.next_IBKR_tick_str=self.calculate_contract_month_symbol(self.next_year,self.next_season)
        #检查路径和文件是否存在
        self.NQ_type0_path=live_data_base+'/type0/NQ/'
        self.current_filename = 'NQBASE'+self.current_file_str+'.pkl'
        self.next_filename = 'NQBASE'+self.next_file_str+'.pkl'
        
    def load_NQ_harddisk(self):
        #先处理当前季度合约
        fullpath = os.path.join(self.NQ_type0_path, self.current_filename)
        if os.path.isfile(fullpath):
            self.current_contract_data=pd.read_pickle(self.NQ_type0_path+self.current_filename)
        #再处理下一个季度合约
        fullpath = os.path.join(self.NQ_type0_path, self.next_filename)
        if os.path.isfile(fullpath):
            self.next_contract_data=pd.read_pickle(self.NQ_type0_path+self.next_filename)

    def sync_NQ_base(self):
        fullpath = os.path.join(self.NQ_type0_path, self.current_filename)
        #先处理当前季度合约
        if os.path.isfile(fullpath):
            #文件存在，则先读入文件，再拼接live数据
            last_BASE_time=self.current_contract_data.index[-1]
            delta = self.now - last_BASE_time    # 这是一个 timedelta 对象
            days = max(delta.days, 0)+1 # .days 已经是向下取整的天数，负数就算 0
            self.new_data=self.request_many_day_NQ(days,self.current_num)
            self.current_contract_data=self.fast_concat(self.current_contract_data,self.new_data)
            self.current_contract_data.to_pickle(self.NQ_type0_path+self.current_filename)
        else:
            #文件不存在，直接读入1000天的live数据构造
            self.current_contract_data=self.request_many_day_NQ(1000,self.current_num)
            self.current_contract_data.to_pickle(self.NQ_type0_path+self.current_filename)
        
        fullpath = os.path.join(self.NQ_type0_path, self.next_filename)
        #再处理下一个季度合约
        if os.path.isfile(fullpath):
            #文件存在，则先读入文件，再拼接live数据
            last_BASE_time=self.next_contract_data.index[-1]
            delta = self.now - last_BASE_time    # 这是一个 timedelta 对象
            days = max(delta.days, 0)+1 # .days 已经是向下取整的天数，负数就算 0
            self.new_data=self.request_many_day_NQ(days,self.next_num)
            self.next_contract_data=self.fast_concat(self.next_contract_data,self.new_data)
            self.next_contract_data.to_pickle(self.NQ_type0_path+self.next_filename)
        else:
            #文件不存在，直接读入1000天的live数据构造
            self.next_contract_data=self.request_many_day_NQ(1000,self.next_num)
            self.next_contract_data.to_pickle(self.NQ_type0_path+self.next_filename)

    def minute_march(self):#每分钟需要做的事情
        self.sync_param()
        if self.last_minute_contract_num != self.current_num:
            self.live_change=1 #是否发生在线状态下的合约转变
            self.last_minute_contract_num=self.current_num
        else:
            self.current_contract_data=self.fast_concat(self.current_contract_data,self.request_many_min_NQ(minute,self.current_num))


def main():
    df=request_1_day_NQ(103)
    # 6. 打印或返回
    print(df.head())
    print(df.tail())
    print(len(df))

if __name__ == '__main__':
    main()