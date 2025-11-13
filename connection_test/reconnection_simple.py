#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import time
from datetime import datetime
from ib_insync import IB, Stock

HOST = '127.0.0.1'
PORT = 4004
CLIENT_ID = 7

FIRST_WAIT_TIMEOUT = 30.0   # 首次连接等待第一根bar的超时（秒）
RECONNECT_GAP = 120.0         # 断开后重连前的间隔（秒）
AFTER_FIRST_DISCONNECT_DELAY = 2.0  # 收到第一根bar后延迟断开（秒）
SECOND_WAIT_TIMEOUT = 20.0  # 重连后等待第一根bar超时（秒）

def log(msg: str):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f'[{now}] {msg}', flush=True)

async def subscribe_and_wait_first_bar(ib: IB, *, useRTH: bool, wait_timeout: float):
    """
    连接已建立的前提下：
    - 发送 reqMarketDataType(1)
    - 订阅 QQQ 的 5s 实时bar（TRADES, useRTH=...）
    - 等待第一根bar（最多 wait_timeout 秒）
    - 返回 (rt_bars, first_bar)；超时则抛 TimeoutError
    """
    ib.reqMarketDataType(1)
    log('reqMarketDataType(1) sent.')

    contract = Stock('QQQ', 'SMART', 'USD')
    log('QQQ contract prepared.')

    rt_bars = ib.reqRealTimeBars(contract, 5, 'TRADES', useRTH, [])
    log('reqRealTimeBars sent.')

    first_bar_holder = {'bar': None}
    first_event = asyncio.Event()

    def on_update(rt_list, hasNewBar):
        if not hasNewBar:
            return
        b = rt_list[-1]
        # 兼容 open / open_
        o = getattr(b, 'open', getattr(b, 'open_', None))
        h = getattr(b, 'high', None)
        l = getattr(b, 'low', None)
        c = getattr(b, 'close', None)
        v = getattr(b, 'volume', None)
        t = b.time if isinstance(b.time, datetime) else datetime.fromtimestamp(b.time)
        log(f'bar received: time={t} o={o} h={h} l={l} c={c} v={v}')
        first_bar_holder['bar'] = b
        first_event.set()

    rt_bars.updateEvent += on_update
    log('updateEvent handler attached.')

    try:
        try:
            await asyncio.wait_for(first_event.wait(), timeout=wait_timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f'No real-time bars within {wait_timeout:.0f}s')
        return rt_bars, first_bar_holder['bar']
    except Exception:
        # 订阅失败或超时；把订阅清干净再往外抛
        try:
            rt_bars.updateEvent -= on_update
        except Exception:
            pass
        try:
            ib.cancelRealTimeBars(rt_bars)
        except Exception:
            pass
        raise

async def graceful_unsubscribe(ib: IB, rt_bars):
    """尽力取消订阅并解绑事件（不抛错）"""
    try:
        # 强制移除所有回调（eventkit的Event没有简单的clear，这里容错处理）
        try:
            rt_bars.updateEvent.clear()
        except Exception:
            # 没有 clear 方法就忽略
            pass
        ib.cancelRealTimeBars(rt_bars)
        log('cancelRealTimeBars called.')
    except Exception as e:
        log(f'ignore error on cancelRealTimeBars: {e}')

async def run_probe():
    # ---------- 第一次连接 ----------
    ib = IB()
    log(f'Connecting {HOST}:{PORT} clientId={CLIENT_ID} ...')
    await ib.connectAsync(HOST, PORT, clientId=CLIENT_ID, timeout=10, readonly=True)
    log('Connected (first session).')

    try:
        # 订阅并等第一根bar
        rt_bars, first_bar = await subscribe_and_wait_first_bar(
            ib, useRTH=False, wait_timeout=FIRST_WAIT_TIMEOUT
        )
        # 收到第一根bar后，再等2秒，随后断开
        await asyncio.sleep(AFTER_FIRST_DISCONNECT_DELAY)
        await graceful_unsubscribe(ib, rt_bars)
    finally:
        if ib.isConnected():
            ib.disconnect()
            log('Disconnected (first session).')

    # ---------- 重连前等待 ----------
    await asyncio.sleep(RECONNECT_GAP)

    # ---------- 第二次连接 ----------
    ib2 = IB()
    log(f'Re-connecting {HOST}:{PORT} clientId={CLIENT_ID} ...')
    # 说明：这里继续使用相同 clientId。如果你的网关在短时间内“保留旧会话”，
    # 可能出现握手卡顿或拒绝。若遇到 1102/1009/超时，可尝试换一个 clientId 再试。
    await ib2.connectAsync(HOST, PORT, clientId=CLIENT_ID, timeout=10, readonly=True)
    log('Connected (second session).')

    try:
        rt_bars2, first_bar2 = await subscribe_and_wait_first_bar(
            ib2, useRTH=False, wait_timeout=SECOND_WAIT_TIMEOUT
        )
        # 打印第二阶段的第一根bar并结束
        b = first_bar2
        t = b.time if isinstance(b.time, datetime) else datetime.fromtimestamp(b.time)
        o = getattr(b, 'open', getattr(b, 'open_', None))
        h = getattr(b, 'high', None)
        l = getattr(b, 'low', None)
        c = getattr(b, 'close', None)
        v = getattr(b, 'volume', None)
        log(f'[SUCCESS] Reconnected & got first bar: time={t} o={o} h={h} l={l} c={c} v={v}')
        await graceful_unsubscribe(ib2, rt_bars2)
    except TimeoutError as e:
        log(f'[FAIL] {e}')
    finally:
        if ib2.isConnected():
            ib2.disconnect()
            log('Disconnected (second session).')

def main():
    try:
        asyncio.run(run_probe())
    except KeyboardInterrupt:
        log('Interrupted by user.')

if __name__ == '__main__':
    main()
