from ib_insync import *
import time
import math

HOST = '127.0.0.1'
PORT = 4004   # TWS=7497, Gateway=4002
CLIENT_ID = 5

def reconnect_with_backoff(ib: IB, max_backoff=60):
    """
    指数回退连接：1s -> 2s -> 4s ... 直到 max_backoff（默认60s）
    Gateway 还在自动登录阶段时，这段会安静地等到可连接为止。
    """
    attempt = 0
    while True:
        try:
            if ib.isConnected():
                return
            # 先确保干净的连接状态
            try:
                ib.disconnect()
            except Exception:
                pass

            print(f"[RECONNECT] 尝试连接 {HOST}:{PORT} clientId={CLIENT_ID} ...")
            ib.connect(HOST, PORT, clientId=CLIENT_ID)
            wait(1000)

            if ib.isConnected():
                print("[RECONNECT] ✅ 重连成功")
                return
            else:
                raise RuntimeError("连接未建立")

        except Exception as e:
            wait = min(max_backoff, 2 ** attempt)  # 1,2,4,8,...,<=60
            attempt += 1
            print(f"[RECONNECT] 失败: {e} -> {wait}s 后重试")
            time.sleep(wait)

def safe_req_hist(ib: IB, contract: Contract):
    """
    包一层请求：失败则抛出异常交给上层处理（从而触发重连）
    """
    return ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='60 S',
        barSizeSetting='1 min',
        whatToShow='TRADES',
        useRTH=False,
        formatDate=1,
        keepUpToDate=False
    )

def main():
    ib = IB()

    # 初次连接（若 Gateway 正在自动登录，可能会失败，进入回退重连）
    reconnect_with_backoff(ib)

    contract = Stock('QQQ', 'SMART', 'USD')

    # 可选：监听断线/重连事件（仅做日志）
    def on_disconnect():
        print("[EVENT] 🔌 掉线了")
    def on_connect():
        print("[EVENT] 🔗 已连接")
    ib.disconnectedEvent += on_disconnect
    ib.connectedEvent += on_connect

    while True:
        try:
            bars = safe_req_hist(ib, contract)
            if bars:
                b = bars[-1]
                print(f"时间:{b.date} | 开:{b.open} | 高:{b.high} | 低:{b.low} | 收:{b.close} | 量:{b.volume}")
            else:
                print("⚠ 未获取到数据")

            time.sleep(30)

        except Exception as e:
            # 常见情形：网络波动、Gateway重启/自动登录中、错误1100/1101/1102等
            print(f"[LOOP] 请求异常: {e}")
            reconnect_with_backoff(ib)  # 阻塞等待直到恢复，然后继续下一轮

if __name__ == "__main__":
    main()