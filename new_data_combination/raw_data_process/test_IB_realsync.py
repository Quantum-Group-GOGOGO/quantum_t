import asyncio
from env import *
from ib_insync import IB, Future, util, Forex, Crypto, Stock
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from live_NQ_data_base import nq_live_t0

# 独立的异步函数：请求最近 5 分钟的 1 分钟 Bar
async def fetchHistory(ib: IB, contract: Future, end_time: datetime):
    # 把 UTC 时间转换成 美东本地时间，并去掉 tzinfo
    local_dt = end_time.astimezone( ('America/New_York')).replace(tzinfo=None)
    endDT = local_dt.strftime("%Y%m%d %H:%M:%S")

    try:
        barsHist = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=endDT,
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
        print(f'[{local_dt}] 没有返回历史数据（合约可能不活跃）')
    else:
        df = util.df(barsHist)
        print(f'历史 5 分钟 K 线（截至 {local_dt}）：')
        print(df)

def main():
    global localhost
    ib = IB()
    ib.connect(localhost, 4002, clientId=2)
    ib.reqMarketDataType(1)
    print('已连接到 IB Gateway/TWS')

    test=nq_live_t0(ib)
    # NQ2025U 合约
    contract = Future(
        symbol='NQ',
        lastTradeDateOrContractMonth='202509',
        exchange='CME',
        currency='USD'
    )
    #contract = Stock('QQQ', 'SMART', 'USD')
    # 订阅 5 秒实时 Bar
    bars = ib.reqRealTimeBars(contract, barSize=5, whatToShow='TRADES', useRTH=False)

    # 回调：打印实时价，整分钟时触发历史请求
    def onBar(barsList, hasNew):
        bar = barsList[-1]
        print(f'{bar.time} 实时价：{bar.time} {bar.open_} {bar.high} {bar.low} {bar.close} {bar.volume} ')
        # 如果秒数恰好是 0，就异步拉一次历史
        #if bar.time.second == 0:
            #test.minute_march()
            #asyncio.create_task(fetchHistory(ib, contract, bar.time))

    bars.updateEvent += onBar
    #sleep(30)
    # 挂起，保持订阅不断开
    #asyncio.Event().wait()
    ib.run()

if __name__ == '__main__':
    main()
    #util.run(main())