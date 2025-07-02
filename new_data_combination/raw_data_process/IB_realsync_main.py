import asyncio
import threading
from env import *
from ib_insync import IB, Future, util, Forex, Crypto, Stock
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os, psutil

from live_NQ_data_base import nq_live_t0
from live_QQQ_data_base import qqq_live_t0

class minute_bar:
    def __init__(self):
        self.clear()
        self.start=1

    def accumulate_open(self,open_,high_,low_,close_,volume_):
        self.open_=open_
        self.high_=high_
        self.low_=low_
        self.close_=close_
        self.volume_=volume_

    def accumulate(self,open_,high_,low_,close_,volume_):
        self.high_=max(self.high_,high_)
        self.low_=min(self.low_,low_)
        self.close_=close_
        self.volume_=self.volume_+volume_

    def clear(self):
        self.last_bar=0
        self.open_=0
        self.high_=0
        self.low_=0
        self.close_=0
        self.volume_=0


# 独立的异步函数：请求最近 5 分钟的 1 分钟 Bar
async def fetchHistorytest(ib: IB, end_time: datetime):
    return 1
async def fetchHistory(ib: IB, end_time: datetime):
    # 把 UTC 时间转换成 美东本地时间，并去掉 tzinfo
    #local_dt = end_time.astimezone( ('America/New_York')).replace(tzinfo=None)
    #endDT = end_time.strftime("%Y%m%d %H:%M:%S")
    contract = Stock('QQQ', 'SMART', 'USD')
    try:
        barsHist = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_time,
            durationStr='300 S',       # 5 分钟 = 300 秒
            barSizeSetting='1 min',    # 1 分钟 Bar
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )
    except Exception as e:
        print('历史数据请求出错：', e)
        return

    if not barsHist:
        print(f'[{end_time}] 没有返回历史数据（合约可能不活跃）')
        df = util.df(barsHist)
        return df
    else:
        df = util.df(barsHist)
        print(f'历史 5 分钟 K 线（截至 {end_time}）：')
        print(df)
        return df

def subscribe_contract(nqt0:nq_live_t0,qqqt0:qqq_live_t0,ib:IB,bars_list:list):
    for i,bars in enumerate(bars_list):
        bars.cancel()
        bars_list.pop(i)

    current_str,next_str=nqt0.request_current_next_symbol()

    contract_current = Future(
        symbol='NQ',
        lastTradeDateOrContractMonth=current_str,
        exchange='CME',
        currency='USD'
    )
    contract_next = Future(
        symbol='NQ',
        lastTradeDateOrContractMonth=next_str,
        exchange='CME',
        currency='USD'
    )
    current_bars=minute_bar()
    next_bars=minute_bar()
    qqq_bars=minute_bar()
    contract_QQQ = Stock('QQQ', 'SMART', 'USD')
    bars = ib.reqRealTimeBars(contract_current, barSize=5, whatToShow='TRADES', useRTH=False)
    bars.updateEvent += lambda barsList, hasNew: asyncio.create_task(onBar_current(nqt0, ib, current_bars, barsList, hasNew))
    bars_list.append(bars)
    bars = ib.reqRealTimeBars(contract_next, barSize=5, whatToShow='TRADES', useRTH=False)
    bars.updateEvent += lambda barsList, hasNew: asyncio.create_task(onBar_next(nqt0, ib, next_bars, barsList, hasNew))
    bars_list.append(bars)
    bars = ib.reqRealTimeBars(contract_QQQ, barSize=5, whatToShow='TRADES', useRTH=False)
    bars.updateEvent += lambda barsList, hasNew: asyncio.create_task(onBar_QQQ(qqqt0, ib, qqq_bars, barsList, hasNew))
    bars_list.append(bars)
    return bars_list

# 回调：打印实时价，整分钟时触发历史请求
async def onBar_current(t0, ib, current_bars, barsList, hasNew):
    threading.current_thread().name='Worker'
    bar = barsList[-1]
    bar.time=bar.time.astimezone(ZoneInfo('America/New_York')).replace(tzinfo=None)
    if bar.time.second == 55:#收到一根线，反应
        if current_bars.start==1:
            current_bars.start=0
            current_bars.last_bar=0
            current_bars.clear()
        else:
            current_bars.accumulate(bar.open_,bar.high,bar.low,bar.close,bar.volume)
            await t0.fast_march(bar.time.replace(second=0, microsecond=0),current_bars.open_,current_bars.high_,current_bars.low_,current_bars.close_,current_bars.volume_,1)
            current_bars.last_bar=0
            current_bars.clear()
    else:
        if current_bars.last_bar==0:
            current_bars.accumulate_open(bar.open_,bar.high,bar.low,bar.close,bar.volume)
            current_bars.last_bar=1
        else:
            current_bars.accumulate(bar.open_,bar.high,bar.low,bar.close,bar.volume)
            current_bars.last_bar=1

    if bar.time.second == 15:#空闲时间维护内存
        t0.check_current_memory()

async def onBar_next(t0, ib, next_bars, barsList, hasNew):
    threading.current_thread().name='Worker'
    bar = barsList[-1]
    bar.time=bar.time.astimezone(ZoneInfo('America/New_York')).replace(tzinfo=None)

    if bar.time.second == 55:#收到一根线，反应
        if next_bars.start==1:
            next_bars.start=0
            next_bars.last_bar=0
            next_bars.clear()
        else:
            next_bars.accumulate(bar.open_,bar.high,bar.low,bar.close,bar.volume)
            await t0.fast_march(bar.time.replace(second=0, microsecond=0),next_bars.open_,next_bars.high_,next_bars.low_,next_bars.close_,next_bars.volume_,0)
            next_bars.last_bar=0
            next_bars.clear()
            
    else:
        if next_bars.last_bar==0:
            next_bars.accumulate_open(bar.open_,bar.high,bar.low,bar.close,bar.volume)
            next_bars.last_bar=1
        else:
            next_bars.accumulate(bar.open_,bar.high,bar.low,bar.close,bar.volume)
            next_bars.last_bar=1

    if bar.time.second == 15:#空闲时间维护内存
        t0.check_next_memory()

async def onBar_QQQ(t0, ib, qqq_bars, barsList, hasNew):
    threading.current_thread().name='Worker'
    bar = barsList[-1]
    bar.time=bar.time.astimezone(ZoneInfo('America/New_York')).replace(tzinfo=None)
    #print(f'{bar.time} 实时价3：{bar.time} {bar.open_} {bar.high} {bar.low} {bar.close} {bar.volume*100} ')
    if bar.time.second == 55:#收到一根线，反应
        if qqq_bars.start==1:
            qqq_bars.start=0
            qqq_bars.last_bar=0
            qqq_bars.clear()
        else:
            qqq_bars.accumulate(bar.open_,bar.high,bar.low,bar.close,bar.volume*100)
            await t0.fast_march(bar.time.replace(second=0, microsecond=0),qqq_bars.open_,qqq_bars.high_,qqq_bars.low_,qqq_bars.close_,qqq_bars.volume_)
            qqq_bars.last_bar=0
            qqq_bars.clear()
    else:
        if qqq_bars.last_bar==0:
            qqq_bars.accumulate_open(bar.open_,bar.high,bar.low,bar.close,bar.volume*100)
            qqq_bars.last_bar=1
        else:
            qqq_bars.accumulate(bar.open_,bar.high,bar.low,bar.close,bar.volume*100)
            qqq_bars.last_bar=1
    
    if bar.time.second == 15:#空闲时间维护内存
        t0.check_qqq_memory()

    # 如果秒数恰好是 0，就异步输入一次数据，但如果数据与最后一行数据相差大于1分钟，则拉一次历史数据同步
    # 拿到当前事件循环
    #p = psutil.Process(os.getpid())
    #print("系统线程总数：", p.num_threads())

class main_Program:
    def __init__(self):
        global localhost
        self.ib = IB()
        self.ib.connect(localhost, 4002, clientId=2)
        self.ib.reqMarketDataType(1)
        print('已连接到 IB Gateway/TWS')

        self.t0_obj_nq = nq_live_t0(self.ib)
        self.t0_obj_qqq = qqq_live_t0(self.ib)
        self.bars_list = []
        self.bars_list = subscribe_contract(self.t0_obj_nq,self.t0_obj_qqq,self.ib,self.bars_list)
    def run(self):
        # 挂起，保持订阅不断开
        
        self.ib.run()
    def save(self):
        self.t0_obj_nq.save()
        self.t0_obj_qqq.save()

    

if __name__ == '__main__':
    main_program=main_Program()
    main_program.run()
