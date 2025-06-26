from ib_insync import IB, Future
import pandas as pd
import time

# 初始化 IB 连接函数
def connect_ib(ib, host='127.0.0.1', port=7497, clientId=1):
    while True:
        try:
            ib.connect(host, port, clientId)
            print('已连接到 IB Gateway/TWS')
            return
        except Exception as e:
            print(f'连接失败：{e}，5秒后重试...')
            time.sleep(5)

# 定义两个合约：2025年9月(U) 与 2025年12月(Z)
contracts = {
    '202509': Future(symbol='NQ', exchange='GLOBEX', currency='USD', lastTradeDateOrContractMonth='202509'),
    '202512': Future(symbol='NQ', exchange='GLOBEX', currency='USD', lastTradeDateOrContractMonth='202512')
}

# 创建 DataFrame 存储各自数据
columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
dtypes = {'datetime': 'datetime64[ns]', 'open': float, 'high': float,
          'low': float, 'close': float, 'volume': float}

df = {month: pd.DataFrame(columns=columns).astype(dtypes) for month in contracts}

# 缓存 pending，以同步同一分钟的多个合约数据
pending = {}  # key: datetime, value: dict of month->row

# 当两个合约的同一分钟 Bar 都到齐时的统一处理函数
def process_both(dt, rows):
    # dt 是 datetime，rows 是 dict month->row
    # 示例操作：打印并可在此处执行你的业务逻辑
    print(f"两合约齐到: {dt}")
    for month, row in rows.items():
        print(f"  {month}: {row}")
    # TODO: 在此执行批量操作，例如写入数据库或指标计算

# 回调函数：处理整分 Bar，并同步两合约
def on_realtime_bar(bars, hasNewBar):
    bar = bars[-1]
    dt = pd.to_datetime(bar.time)
    # 仅处理整分 Bar
    if dt.second != 0:
        return

    month = bars.contract.lastTradeDateOrContractMonth
    row = {
        'datetime': dt,
        'open': float(bar.open),
        'high': float(bar.high),
        'low': float(bar.low),
        'close': float(bar.close),
        'volume': float(bar.volume)
    }
    # 存入对应 DataFrame
    df[month] = pd.concat([df[month], pd.DataFrame([row])], ignore_index=True)
    print(f"{month} 合约整分更新: {row}")

    # 累积到 pending
    if dt not in pending:
        pending[dt] = {}
    pending[dt][month] = row
    # 检查是否所有合约的该分钟数据都已到齐
    if set(pending[dt].keys()) == set(contracts.keys()):
        # 统一处理
        process_both(dt, pending[dt])
        # 清理缓存
        del pending[dt]

# 主流程：连接 + 订阅 + 断线重连
def main():
    ib = IB()
    connect_ib(ib)

    # 注册回调和断线监听
    ib.realtimeBarsEvent += on_realtime_bar
    ib.disconnectedEvent += lambda: print('与 IB Gateway/TWS 断开连接')

    # 订阅所有合约
    def subscribe_all():
        for contract in contracts.values():
            ib.reqRealTimeBars(contract, barSize=60, whatToShow='TRADES', useRTH=False)
        print('已订阅所有合约的实时数据')

    subscribe_all()

    # 事件循环，断线时重连并重新订阅
    while True:
        try:
            ib.run()
        except KeyboardInterrupt:
            print('已停止实时数据订阅')
            break
        except Exception as e:
            print(f'运行异常：{e}，尝试重连...')
            ib.disconnect()
            time.sleep(5)
            connect_ib(ib)
            subscribe_all()

    # 停止后可选保存
    # for month, data in df.items():
    #     data.to_csv(f'nq_{month}_bars.csv', index=False)

if __name__ == '__main__':
    main()