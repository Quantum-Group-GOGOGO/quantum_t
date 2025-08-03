from ib_insync import IB, Contract,Future,Stock
from datetime import datetime, timezone, timedelta
import pandas as pd
import pytz
import re


def make_ts(date8, hm):
            fmt = '%Y%m%d%H%M' if len(hm) == 4 else '%Y%m%d%H%M%S'
            raw = date8 + hm
            # 先当作无时区解析
            ts = pd.to_datetime(raw, format=fmt)
            # 本地化到交易所时区
            return ts

async def is_contract_tradable(ib: IB, contract: Contract, time: datetime) -> bool:
    """
    返回指定合约当前是否在交易时段内。

    参数
    ----
    ib : IB
        已连接且登录的 IB 实例
    contract : Contract
        待检测的合约对象

    返回
    ----
    bool
        True = 当前可交易；False = 不可交易或查询失败
    """
    # 1. 请求合约详情
    details = await ib.reqContractDetailsAsync(contract)
    if not details:
        # 合约不存在或查询不到详情
        return False
    
    # 2. tradingHours 是一个字符串，每天格式类似 "20250708:090000-160000;20250709:000000-000000;..."
    #    我们只关心当天（UTC 日期或交易所时区）的时段
    th = details[0].tradingHours
    
    # 3. 解析 tradingHours 字符串，找出所有当日的区间
    #    tradingHours 格式：YYYYMMDD:HHMMSS-HHMMSS;YYYYMMDD:...
    today = datetime.now(timezone.utc).astimezone().date()  # 当前本地日期
    today_str = today.strftime('%Y%m%d')

    intervals = []
    for part in th.split(';'):
        # 匹配 “YYYYMMDD:HHMMSS-HHMMSS”
        m = re.match(r'(\d{8}):(\d{4})-(\d{8}):(\d{4})', part)
        
        
        if not m:
            continue
        start_date, start_hms, end_date, end_hms = m.groups()
        start_ex = make_ts(start_date, start_hms)
        end_ex   = make_ts(end_date, end_hms)

        if start_ex <= time <= end_ex:
            
            return True

    return False





def main():
    ib = IB()
    ib.connect('127.0.0.1', 4004, clientId=2)

    # 以 QQQ 股票为例
    contract=Future(symbol='NQ',exchange='CME',currency='USD',lastTradeDateOrContractMonth='202509')
    contract = Stock('QQQ', 'SMART', 'USD')
    if is_contract_tradable(ib, contract):
        print("合约当前可交易")
    else:
        print("合约暂不可交易")

if __name__ == "__main__":
    main()