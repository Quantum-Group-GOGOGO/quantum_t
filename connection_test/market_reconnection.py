#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import asyncio
from dataclasses import dataclass
from datetime import datetime
from ib_insync import IB, Stock

HOST = '127.0.0.1'
PORT = 4004
CLIENT_ID_BASE = 7   # 起始 clientId，将与 (BASE, BASE+1) 交替使用

# 若连续 STALE_TIMEOUT 秒未收到任何 5s 实时 bar，认为连接“功能性中断”，触发重连
STALE_TIMEOUT = 120.0

# ---------- 统一时间戳日志 ----------
def log(message: str):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f'[{now}] {message}', flush=True)

# ---------- 1min 聚合器 ----------
@dataclass
class MinuteAgg:
    minute: datetime     # 去掉秒和微秒（本地时区），作为这一分钟的键
    open: float
    high: float
    low: float
    close: float
    volume: float

current_agg: MinuteAgg | None = None   # 正在聚合的 1 分钟
last_emit_minute: datetime | None = None

def floor_to_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)

def fmt_min_bar(bar_time_minute: datetime, agg: MinuteAgg) -> str:
    ts_str = bar_time_minute.strftime('%Y-%m-%d %H:%M')
    return (f'[{ts_str}] QQQ  open={agg.open:.2f}  high={agg.high:.2f}  '
            f'low={agg.low:.2f}  close={agg.close:.2f}  volume={agg.volume:.0f}')

def emit_and_reset(new_minute_key: datetime):
    global current_agg, last_emit_minute
    if current_agg is not None:
        log(fmt_min_bar(current_agg.minute, current_agg))
        last_emit_minute = current_agg.minute
    # 新的一分钟由 on_rtbar_update 里创建 current_agg

# ---------- 实时 5s bar 订阅与事件 ----------
def on_rtbar_update(rt_bars, hasNewBar):
    if not rt_bars or not hasNewBar:
        return

    # 收到新bar，刷新“最后一次收到实时bar”的时间，用于停滞自检
    on_rtbar_update.last_rtbar_mono = time.monotonic()
    log(f'real-time bar received, count={len(rt_bars)}')

    bar = rt_bars[-1]  # 最新这根5s bar

    # 兼容不同版本字段名：open 或 open_
    def _f(name, name_alt=None):
        if name_alt is None:
            name_alt = name
        v = getattr(bar, name, None)
        if v is None:
            v = getattr(bar, name_alt, None)
        return v

    o = _f('open', 'open_')
    h = _f('high')
    l = _f('low')
    c = _f('close')
    v = float(_f('volume') or 0.0)

    bar_time = bar.time if isinstance(bar.time, datetime) else datetime.fromtimestamp(bar.time)
    bar_time = bar_time.astimezone(tz=None)
    minute_key = floor_to_minute(bar_time)

    global current_agg
    if current_agg is None:
        current_agg = MinuteAgg(
            minute=minute_key, open=o, high=h, low=l, close=c, volume=v
        )
        log(f'start new minute agg: {minute_key}')
        return

    if minute_key == current_agg.minute:
        log(f'aggregating bar into current minute {minute_key}')
        if h > current_agg.high:
            current_agg.high = h
        if l < current_agg.low:
            current_agg.low = l
        current_agg.close = c
        current_agg.volume += v
    else:
        log(f'minute change detected from {current_agg.minute} to {minute_key}, emitting previous minute')
        emit_and_reset(minute_key)
        current_agg = MinuteAgg(
            minute=minute_key, open=o, high=h, low=l, close=c, volume=v
        )

# 自检用字段初始化
on_rtbar_update.last_rtbar_mono = None  # 最近一次收到 5s 实时 bar 的 monotonic 时间

# ---------- 连接/订阅/清理 ----------
def connect_and_subscribe(ib: IB, client_id: int):  # ### CHANGED: 传入 client_id
    """
    建立连接并订阅 QQQ 的 5s 实时 bar（TRADES），返回 (rt_bars, on_disconnected, on_error)
    """
    log(f'Connecting to {HOST}:{PORT} with clientId={client_id} ...')
    ib.connect(HOST, PORT, clientId=client_id, timeout=10, readonly=True)
    log('Connected.')

    ib.reqMarketDataType(1)
    log('reqMarketDataType(1) sent.')

    contract = Stock('QQQ', 'SMART', 'USD')
    log('QQQ contract prepared.')

    rt_bars = ib.reqRealTimeBars(
        contract,
        5,                # barSize = 5秒（IB固定值）
        'TRADES',
        False,            # useRTH：False=全部时间，True=仅正则交易时段
        []
    )
    log('reqRealTimeBars sent.')

    # 绑定回调
    rt_bars.updateEvent += on_rtbar_update
    log('updateEvent handler attached.')

    def on_disconnected():
        log('[warn] Socket disconnected from Gateway/TWS.')
    ib.disconnectedEvent += on_disconnected
    log('disconnectedEvent handler attached.')

    def on_error(reqId, errorCode, errorString, contract):
        if errorCode in (1100, 1101, 1102):
            log(f'[warn] API error {errorCode}: {errorString} → will reconnect...')
            on_rtbar_update.last_rtbar_mono = time.monotonic() - (STALE_TIMEOUT + 1)
    ib.errorEvent += on_error
    log('errorEvent handler attached (1100/1101/1102).')

    # 初始刷新“最后收到 5s bar 时间”，避免刚连上就被判停滞
    on_rtbar_update.last_rtbar_mono = time.monotonic()

    # 重置本地聚合器
    global current_agg, last_emit_minute
    current_agg = None
    last_emit_minute = None

    log('Subscribed to QQQ real-time 5s bars (aggregating to 1-min). Running...')
    return rt_bars, on_disconnected, on_error

def safe_cleanup(ib: IB, rt_bars, on_disconnected, on_error=None):
    """
    取消订阅、解绑回调、断开连接；在重连前调用，确保无资源泄漏。
    """
    log('Starting safe_cleanup...')
    try:
        if rt_bars is not None:
            # 尝试清空 updateEvent 队列，最大化去引用
            try:
                try:
                    rt_bars.updateEvent.clear()  # eventkit>=1.0 可能支持
                    log('rt_bars.updateEvent cleared.')
                except Exception:
                    rt_bars.updateEvent -= on_rtbar_update
                    log('rt_bars.updateEvent handler detached.')
            except Exception as e:
                log(f'ignore error detaching updateEvent: {e}')
            try:
                ib.cancelRealTimeBars(rt_bars)
                log('cancelRealTimeBars called.')
            except Exception as e:
                log(f'ignore error cancelRealTimeBars: {e}')
    finally:
        try:
            if on_disconnected:
                ib.disconnectedEvent -= on_disconnected
                log('disconnectedEvent handler detached.')
        except Exception as e:
            log(f'ignore error detaching disconnectedEvent: {e}')
        try:
            if on_error:
                ib.errorEvent -= on_error
                log('errorEvent handler detached.')
        except Exception as e:
            log(f'ignore error detaching errorEvent: {e}')
        try:
            if ib.isConnected():
                ib.disconnect()
                log('Disconnected from IB.')
        except Exception as e:
            log(f'ignore error on disconnect: {e}')
    log('safe_cleanup done.')

# ### NEW: 统一的“丢旧建新”逻辑，返回 (new_ib, next_client_id)
def recreate_ib_and_rotate_client_id(curr_next_id: int):
    """
    为下一次连接创建一个全新的 IB 实例，并返回 (ib, client_id_to_use, next_id)
    我们在 CLIENT_ID_BASE 与 CLIENT_ID_BASE+1 之间交替，以规避网关短保留旧会话的问题。
    """
    client_id = curr_next_id
    next_id = (CLIENT_ID_BASE if client_id == CLIENT_ID_BASE + 1 else CLIENT_ID_BASE + 1)
    ib = IB()
    log(f'[recreate] New IB() created, will use clientId={client_id} (next={next_id})')
    return ib, client_id, next_id

def main():
    # ### CHANGED: 使用“轮换 clientId”的思路
    next_client_id = CLIENT_ID_BASE
    ib, client_id, next_client_id = recreate_ib_and_rotate_client_id(next_client_id)

    # 指数退避重连参数（最大不超过 60 秒）
    backoff = 3.0
    backoff_factor = 1.7
    backoff_cap = 60.0

    rt_bars = None
    on_disconnected = None
    on_error = None

    try:
        while True:
            try:
                # 连接 & 订阅（使用当前 client_id）
                rt_bars, on_disconnected, on_error = connect_and_subscribe(ib, client_id)

                # 连接成功，重置退避
                backoff = 3.0
                log('reset backoff to 3s.')

                # 主循环：事件驱动 + 自检
                while True:
                    ib.waitOnUpdate(timeout=2.0)

                    # 1) 物理断连
                    if not ib.isConnected():
                        log('[warn] Detected ib.isConnected() == False, will reconnect...')
                        raise ConnectionError('socket disconnected')

                    # 2) 功能性停滞（长时间没有任何 5s 实时 bar）
                    last_mono = on_rtbar_update.last_rtbar_mono
                    if last_mono is not None and (time.monotonic() - last_mono >= STALE_TIMEOUT):
                        log(f'[warn] No real-time bars for {STALE_TIMEOUT:.0f}s, treating as stale. Will reconnect...')
                        raise TimeoutError('realtime bars stale')

            except (ConnectionError, TimeoutError, OSError, asyncio.TimeoutError) as e:
                # 清理并进入重连
                log(f'enter reconnect path due to: {type(e).__name__}: {e}')
                safe_cleanup(ib, rt_bars, on_disconnected, on_error)
                rt_bars, on_disconnected, on_error = None, None, None

                # ### CHANGED: 丢弃旧 IB，创建新 IB，并轮换 clientId
                ib, client_id, next_client_id = recreate_ib_and_rotate_client_id(next_client_id)

                sleep_s = min(backoff, backoff_cap)
                log(f'[reconnect] Retry in {sleep_s:.0f}s...')
                time.sleep(sleep_s)
                backoff = min(backoff * backoff_factor, backoff_cap)
                log(f'backoff increased to {backoff:.1f}s (capped at {backoff_cap}s).')

            except KeyboardInterrupt:
                log('Stopping by user...')
                break

    finally:
        # 最后一轮清理（注意 ib 已经是“当前新实例”）
        safe_cleanup(ib, rt_bars, on_disconnected, on_error)
        log('Disconnected. Bye.')

if __name__ == '__main__':
    main()
