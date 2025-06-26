from ib_insync import IB, Future
import pandas as pd
import time

# 初始化 IB 连接函数
def connect_ib(ib, host='127.0.0.1', port=4002, clientId=1):
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

# 回调函数：只在整分时处理数据，根据合约追加到 DataFrame
def on_realtime_bar(bars, hasNewBar):
    bar = bars[-1]
    # 仅处理整分 Bar
    dt = pd.to_datetime(bar.time)
    if dt.second != 0:
        return

    row = {
        'datetime': dt,
        'open': float(bar.open),
        'high': float(bar.high),
        'low': float(bar.low),
        'close': float(bar.close),
        'volume': float(bar.volume)
    }
    month = bars.contract.lastTradeDateOrContractMonth
    df[month] = pd.concat([df[month], pd.DataFrame([row])], ignore_index=True)
    print(f"{month} 合约整分更新:", row)

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