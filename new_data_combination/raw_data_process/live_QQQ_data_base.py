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
import time
from preallocdataframe import PreallocDataFrame
from t2_processor import live_t2


class qqq_live_t0:
    def __init__(self,ib):
        global live_data_base
        #初始化
        self.ibob=ib
        self.QQQ_type0_path=live_data_base+'/type0/QQQ/'
        self.QQQ_filename = 'QQQ_BASE.pkl'
        self.sync_param()
        self.live_change=0 #是否发生在线状态下的合约转变
        self.load_QQQ_harddisk()
        self.sync_QQQ_base()
        print(f'QQQ T0处理器初始化完成  {datetime.now()}')

    def link_t2obj(self, t2_processor:live_t2):#只允许被link_sub函数调用
        self.t2_p=t2_processor

    def request_many_day_QQQ(self,daysN):
        now=datetime.now()
        dfs = pd.DataFrame(columns=['datetime','open','high','low','close','volume'])
        dfs.set_index('datetime', inplace=True)
        for day in tqdm(range(daysN), desc='QQQ 历史数据同步中'):
            endtime= now - timedelta(days=day)
            contract = Stock('QQQ', 'SMART', 'USD')
            bars = self._safe_reqHistorical(
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
            df['volume'] = df['volume'] * 100
            if dfs.empty:
                dfs = df
            else:
                dfs = pd.concat([df, dfs])
        dfs = dfs[~dfs.index.duplicated(keep='last')]        
        return dfs

    async def request_many_day_QQQAsync(self,daysN):
        now=datetime.now()
        dfs = pd.DataFrame(columns=['datetime','open','high','low','close','volume'])
        dfs.set_index('datetime', inplace=True)
        for day in tqdm(range(daysN), desc='Processing days'):
            endtime= now - timedelta(days=day)
            contract = Stock('QQQ', 'SMART', 'USD')
            bars = await self._safe_reqHistoricalAsync(
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
            df['volume'] = df['volume'] * 100
            if dfs.empty:
                dfs = df
            else:
                dfs = pd.concat([df, dfs])
        dfs = dfs[~dfs.index.duplicated(keep='last')]        
        return dfs

    def request_1_day_QQQ(self):
        contract = Stock('QQQ', 'SMART', 'USD')
        bars = self._safe_reqHistorical(
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
        df['volume'] = df['volume'] * 100
        return df
    
    async def request_1_day_QQQAsync(self):
        contract = Stock('QQQ', 'SMART', 'USD')
        bars = await self._safe_reqHistoricalAsync(
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
        df['volume'] = df['volume'] * 100
        return df

    def request_many_min_QQQ(self,minute):
        lengthstr=str((minute+3)*60)
        contract = Stock('QQQ', 'SMART', 'USD')
        bars = self._safe_reqHistorical(
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
        df['volume'] = df['volume'] * 100
        return df
    
    async def request_many_min_QQQAsync(self,minute):
        lengthstr=str(minute*60)
        contract = Stock('QQQ', 'SMART', 'USD')
        bars = await self._safe_reqHistoricalAsync(
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
        df['volume'] = df['volume'] * 100
        return df

    def fast_concat(self,main_data_base,new_data_base): #在大数据集main_data_base下方拼接new_data_base，并去掉重复部分，main和new都必须是时间升序排序完成的
        # 1. 找到 new_data_base 中最小（即最早）的索引：
        first_new_idx = new_data_base.index[0]
        # 2. 在 main_data_base 的索引上做二分查找，定位到第一个 >= first_new_idx 的位置
        #    这就是所有可能重复的第一行
        pos = main_data_base.index.searchsorted(first_new_idx, side='left')
        # 3. 切片：只保留 main_data_base 中索引 < first_new_idx 的那部分
        main_data_base.cut_tail(pos)
        # 4. 直接上下拼接
        main_data_base.concat_small(new_data_base)
    
    def fast_concat_savemain(self,main_data_base,new_data_base): #在大数据集main_data_base下方拼接new_data_base，并去掉重复部分，main和new都必须是时间升序排序完成的
        # 两个 DataFrame 都已按时间升序排序，且索引为时间
        # 1. 找到 main_data_base 中最大的索引（最新时间）
        last_main_idx = main_data_base.index[-1]
        # 2. 在 new_data_base 的索引上做二分查找，定位到第一个 > last_main_idx 的位置
        pos = new_data_base.index.searchsorted(last_main_idx, side='right')
        # 3. 只保留 new_data_base 中索引 > last_main_idx 的那部分（去掉所有重复或更早的行）
        to_append = new_data_base.iloc[pos:]
        # 4. 拼接
        main_data_base.concat_small(to_append)
    
    def check_qqq_memory(self):
        self.QQQBASE.ensure_capacity()

    def sync_param(self):
        global live_data_base
        self.now=datetime.now(ZoneInfo('America/New_York'))
        

    def load_QQQ_harddisk(self):
            #先处理当前季度合约
            fullpath = os.path.join(self.QQQ_type0_path, self.QQQ_filename)
            if os.path.isfile(fullpath):
                self.QQQBASE=PreallocDataFrame(pd.read_pickle(self.QQQ_type0_path+self.QQQ_filename))
            else:
                print('Cannot find QQQ database: '+fullpath)

    def save(self):
        self.QQQBASE.to_dataframe().to_pickle(self.QQQ_type0_path+self.QQQ_filename)
        print("已保存QQQ合约")

    def sync_QQQ_base(self):
        last_BASE_time=self.QQQBASE.index[-1]
        now = datetime.now()
        delta = now - last_BASE_time    # 这是一个 timedelta 对象
        days = max(delta.days, 0)+1 # .days 已经是向下取整的天数，负数就算 0
        df=self.request_many_day_QQQ(days)
        self.fast_concat(self.QQQBASE, df)
        self.QQQBASE.to_dataframe().to_pickle(self.QQQ_type0_path+self.QQQ_filename)

    async def sync_QQQ_baseAsync(self):
        last_BASE_time=self.QQQBASE.index[-1]
        now = datetime.now()
        delta = now - last_BASE_time    # 这是一个 timedelta 对象
        days = max(delta.days, 0)+1 # .days 已经是向下取整的天数，负数就算 0
        df=await self.request_many_day_QQQAsync(days)
        self.fast_concat(self.QQQBASE, df)
        self.QQQBASE.to_dataframe().to_pickle(self.QQQ_type0_path+self.QQQ_filename)

    async def minute_march(self):#每分钟需要做的事情
        self.sync_param()
        last_BASE_time=self.QQQBASE.index[-1]
        delta = self.now - last_BASE_time.replace(tzinfo=ZoneInfo('America/New_York'))    # 这是一个 timedelta 对象
        minute=int(delta.total_seconds() // 60)+2
        df=await self.request_many_min_QQQ(minute)
        self.fast_concat(self.QQQBASE,df)

    async def fast_march(self,datetime_,open_,high_,low_,close_,volume_,NQstatus):
        # 这个函数快速录入当前数据，不需要激活request history，只有在发现数据不连续时再动用request history函数用于核对
        # 1) 把这一根 Bar 构造成只有一行的小 DataFrame，
        #    索引用 bar_datetime，列名必须和 self.current_contract_data 一致
        self.now=datetime.now(ZoneInfo('America/New_York'))
        last_BASE_time=self.QQQBASE.index[-1]
        delta = datetime_ - last_BASE_time
        minute=int(delta.total_seconds() // 60)
        if minute<=1:
            new_row = pd.DataFrame(
                [[open_, high_, low_, close_, volume_]],
                index=[datetime_],
                columns=['open', 'high', 'low', 'close', 'volume']
            )
            new_row.index.name = 'datetime'  # 如果你的 current_contract_data.index 名称也是 'datetime'
            # 2) 用 concat 拼接到原 DataFrame 底部
            self.fast_concat_savemain(self.QQQBASE, new_row)
            print(f'{datetime.now()}  QQQ   1分钟连续数据处理完毕：{datetime_} {open_} {high_} {low_} {close_} {volume_}')
            await self.t2_p.fast_march(datetime_,open_,high_,low_,close_,volume_,0,NQstatus)
        elif minute<1440:
            df=await self.request_many_min_QQQAsync(minute+1)
            self.fast_concat_savemain(self.QQQBASE, df)
            print(f'{datetime.now()}  QQQ   {minute}分钟不连续数据处理完毕：{self.QQQBASE.tail()}')
            await self.t2_p.multi_fast_march(0,NQstatus)
        else:
            days=minute//1440
            df=await self.request_many_day_QQQAsync(days+1)
            self.fast_concat_savemain(self.QQQBASE, df)
            print(f'{datetime.now()}  QQQ   {days}日不连续数据处理完毕：{self.QQQBASE.tail()}')
            self.t2_p.slow_march()

            
            

    async def _safe_reqHistoricalAsync(self, contract, **kwargs):
        """
        封装 ib.reqHistoricalData，遇到网络/连接异常时自动重连并重试。
        """
        #loop = asyncio.get_event_loop()
        max_retries = 50
        delay = 20  # 每次重试前等待秒数
        for attempt in range(1, max_retries + 1):
            try:
                # 同步请求历史数据
                #task=loop.create_task(self.ibob.reqHistoricalDataAsync(contract, **kwargs))
                #return loop.run_until_complete(task)
                return await self.ibob.reqHistoricalDataAsync(contract, **kwargs)       
            except Exception as e:
                # 如果是因为断线导致的错误
                print(f"⚠️ requestHistoricalData 第 {attempt} 次失败：{e}")
                # 尝试重连
                if not self.ibob.isConnected():
                    print("🔄 IB disconnected, trying to reconnect...")
                    try:
                        self.ibob.connect('127.0.0.1', 4002, clientId=2)
                        print("✅ Reconnected to IB.")
                    except Exception as connErr:
                        print(f"❌ Reconnect failed: {connErr}")
                # 如果不是最后一次重试，就等待后重试
                if attempt < max_retries:
                    print(f"⏱ Waiting {delay}s before retry...")
                    time.sleep(delay)
                    continue
                else:
                    # 重试用尽，抛异常给上层处理或跳过
                    print("❌ 超过最大重试次数，跳过该请求。")
                    raise
        # 理论上不会走到这
        return []
    
    def _safe_reqHistorical(self, contract, **kwargs):
        """
        封装 ib.reqHistoricalData，遇到网络/连接异常时自动重连并重试。
        """
        #loop = asyncio.get_event_loop()
        max_retries = 50
        delay = 20  # 每次重试前等待秒数
        for attempt in range(1, max_retries + 1):
            try:
                # 同步请求历史数据
                #task=loop.create_task(self.ibob.reqHistoricalDataAsync(contract, **kwargs))
                #return loop.run_until_complete(task)
                return self.ibob.reqHistoricalData(contract, **kwargs)       
            except Exception as e:
                # 如果是因为断线导致的错误
                print(f"⚠️ requestHistoricalData 第 {attempt} 次失败：{e}")
                # 尝试重连
                if not self.ibob.isConnected():
                    print("🔄 IB disconnected, trying to reconnect...")
                    try:
                        self.ibob.connect('127.0.0.1', 4002, clientId=2)
                        print("✅ Reconnected to IB.")
                    except Exception as connErr:
                        print(f"❌ Reconnect failed: {connErr}")
                # 如果不是最后一次重试，就等待后重试
                if attempt < max_retries:
                    print(f"⏱ Waiting {delay}s before retry...")
                    time.sleep(delay)
                    continue
                else:
                    # 重试用尽，抛异常给上层处理或跳过
                    print("❌ 超过最大重试次数，跳过该请求。")
                    raise
        # 理论上不会走到这
        return []



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
    # 6. 打印或返回
    print(df.head())
    print(df.tail())

if __name__ == '__main__':
    main()