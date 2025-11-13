from ib_insync import *
import time

def main():
    ib = IB()

    # === 请修改为你实际的连接参数 ===
    host = '127.0.0.1'
    port = 4004  # TWS默认: 7497，IB Gateway默认: 4002
    client_id = 5 # 任意整数，但不能和其它客户端重复

    print(f"尝试连接 IB: {host}:{port}, clientId={client_id}")
    ib.connect(host, port, clientId=client_id)

    if not ib.isConnected():
        print("❌ 无法连接，请检查 TWS/Gateway 是否启动并允许 API。")
        return
    else:
        print("✅ 已连接成功")

    # 定义 QQQ 合约
    contract = Stock('QQQ', 'SMART', 'USD')

    while True:
        try:
            # 请求最近1分钟（60秒）的历史数据
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='60 S',
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=False,      # 是否只看常规交易时段
                formatDate=1,
                keepUpToDate=False
            )

            if bars:
                last_bar = bars[-1]
                print(f"时间: {last_bar.date} | 开: {last_bar.open} | 高: {last_bar.high} | "
                      f"低: {last_bar.low} | 收: {last_bar.close} | 量: {last_bar.volume}")
            else:
                print("⚠ 未获取到数据")

        except Exception as e:
            print(f"拉取数据出错: {e}")
            if not ib.isConnected():
                print("尝试重新连接...")
                ib.disconnect()
                time.sleep(2)
                ib.connect(host, port, clientId=client_id)

        time.sleep(2)  # 每两秒循环一次


if __name__ == "__main__":
    main()