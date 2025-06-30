from ib_insync import IB, Future
import pandas as pd
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# 初始化 IB 连接函数，用于历史数据拉取
# 使用 clientId=2 连接到端口4002

def connect_ib(host='127.0.0.1', port=4002, clientId=2):
    ib = IB()
    while True:
        try:
            ib.connect(host, port, clientId)
            print(f'历史 IB 连接已建立 (clientId={clientId})')
            return ib
        except Exception as e:
            print(f'连接失败：{e}，5秒后重试...')
            time.sleep(5)

# 定义 NQ 2025年9月合约 (U 月)
contract = Future(
    symbol='NQ',
    exchange='CME',
    currency='USD',
    lastTradeDateOrContractMonth='202509'
)

# DataFrame 列与类型，使用美东时区
df_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
df_dtypes = {
    'datetime': 'datetime64[ns, America/New_York]',
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': float
}

# 主函数：连接并定时拉取历史 Bar
if __name__ == '__main__':
    ib = connect_ib()
    print('启动定时历史拉取任务 (每分钟第01秒)')
    while True:
        now = datetime.now()
        next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        target = next_min + timedelta(seconds=0)
        time.sleep(max(0, (target - now).total_seconds()))
        # 请求最近1分钟历史数据
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=next_min,
                durationStr='60 S',  # 60秒
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1
            )
        except Exception as e:
            print(f'[HIST] 请求异常: {e}')
            continue
        if not bars:
            print('[HIST] 未收到历史 Bar，跳过本次')
            continue
        # 取最后一根 Bar 并转换时区
        last = bars[-1]
        # last.date 是字符串 'YYYY-MM-DD HH:MM:SS'
        dt = pd.to_datetime(last.date).tz_localize('UTC').astimezone(
            ZoneInfo('America/New_York')
        )
        row = {
            'datetime': dt,
            'open': last.open,
            'high': last.high,
            'low': last.low,
            'close': last.close,
            'volume': last.volume
        }
        df_line = pd.DataFrame([row], columns=df_columns).astype(df_dtypes)
        print(f"[HIST] {contract.symbol}{contract.lastTradeDateOrContractMonth} Bar at {dt}:")
        print(df_line)
        print(datetime.now())