# -*- coding: utf-8 -*-
import asyncio
import threading
from env import *
from ib_insync import IB, Future, util, Forex, Crypto, Stock
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import psutil
import pandas as pd
from live_NQ_data_base import nq_live_t0
from live_QQQ_data_base import qqq_live_t0
from t2_processor import live_t2
from t3_processor import live_t3
from t6_processor import live_t6
from decision_machine import live_dm
from neuro_network_processor import live_nn
import trading_status

# === 新增：关机/最终化检测 & 信号处理 ===
import atexit
import signal
import sys
import time

_finalizing_flag = threading.Event()

def _mark_finalizing():
    _finalizing_flag.set()

def is_finalizing():
    try:
        if _finalizing_flag.is_set():
            return True
        if hasattr(sys, "is_finalizing") and sys.is_finalizing():
            return True
    except Exception:
        # 极端情况下 sys 也可能被回收
        return True
    return False

atexit.register(_mark_finalizing)


def onError(reqId, errorCode, errorString, contract):
    # 162 历史数据为空；2174 常见无害错误
    if errorCode in (162, 2174):
        return
    print(f"[Error {errorCode}] ReqId {reqId}: {errorString}")


class minute_bar:
    def __init__(self):
        self.clear()
        self.start = 1

    def accumulate_open(self, open_, high_, low_, close_, volume_):
        self.open_ = open_
        self.high_ = high_
        self.low_ = low_
        self.close_ = close_
        self.volume_ = volume_

    def accumulate(self, open_, high_, low_, close_, volume_):
        self.high_ = max(self.high_, high_)
        self.low_ = min(self.low_, low_)
        self.close_ = close_
        self.volume_ = self.volume_ + volume_

    def clear(self):
        self.last_bar = 0
        self.open_ = 0
        self.high_ = 0
        self.low_ = 0
        self.close_ = 0
        self.volume_ = 0


# ====（保留占位）异步历史拉取 ====
async def fetchHistorytest(ib: IB, end_time: datetime):
    return 1

async def fetchHistory(ib: IB, end_time: datetime):
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


# ==== 安全的订阅刷新 ====
def subscribe_contract(nqt0: nq_live_t0, qqqt0: qqq_live_t0, ib: IB, bars_list: list):
    """
    仅负责：
    1) 安全取消旧订阅
    2) qualify 合约（NQ: 先 CME 再 GLOBEX；QQQ: SMART）
    3) 成功者才订阅；绑定 updateEvent 到你原来的 onBar_*
    4) 不创建任何 watchdog、不过问 onBar_* 内部逻辑
    """
    # 取消旧订阅
    for b in list(bars_list):
        try:
            b.cancel()
        except Exception:
            pass
    bars_list.clear()

    # 取得当前/下季月份字符串
    current_str, next_str = nqt0.request_current_next_symbol()

    # 助手：尝试不同交易所 qualify NQ
    def _qualify_nq(month_str: str):
        for exch in ("CME", "GLOBEX"):
            c = Future(symbol='NQ', lastTradeDateOrContractMonth=month_str, exchange=exch, currency='USD')
            try:
                q = ib.qualifyContracts(c)
                if q:
                    qc = q[0]
                    print(f"[qualify] NQ {month_str} OK: conId={qc.conId} exch={qc.exchange}")
                    return qc
            except Exception as e:
                print(f"[qualify] NQ {month_str} on {exch} 失败: {e}")
        print(f"[qualify] NQ {month_str} 两个交易所都失败（可能无权限或月份不合法）")
        return None

    # qualify NQ 当前/下季
    nq_cur  = _qualify_nq(current_str)
    nq_next = _qualify_nq(next_str)

    # qualify QQQ
    qqq = Stock('QQQ', 'SMART', 'USD')
    try:
        qqq = ib.qualifyContracts(qqq)[0]
        print(f"[qualify] QQQ OK: conId={qqq.conId} primary={getattr(qqq, 'primaryExchange', '')}")
    except Exception as e:
        print(f"[qualify] QQQ 失败: {e}")
        qqq = None

    def _get_running_loop_fallback():
        lp = getattr(ib, "owner_loop", None)
        if lp is not None:
            return lp
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.get_event_loop()
            except Exception:
                return None

    def _bind_current(minbar):
        def _cb(barsList, hasNew):
            # ★ 这行用来确认 updateEvent 有没有被触发
            try:
                print(f"[cb-fire] NQ current len={len(barsList)} last={getattr(barsList[-1],'time',None)}")
            except Exception:
                print("[cb-fire] NQ current (no barsList meta)")

            loop = _get_running_loop_fallback()
            if loop is not None:
                # 用 call_soon_threadsafe 把创建 task 的动作排进 loop 线程
                loop.call_soon_threadsafe(
                    asyncio.create_task, onBar_current(nqt0, ib, minbar, barsList, hasNew)
                )
            else:
                print("[cb] NQ current tick but no loop")
        return _cb

    def _bind_next(minbar):
        def _cb(barsList, hasNew):
            try:
                print(f"[cb-fire] NQ next    len={len(barsList)} last={getattr(barsList[-1],'time',None)}")
            except Exception:
                print("[cb-fire] NQ next (no barsList meta)")
            loop = _get_running_loop_fallback()
            if loop is not None:
                loop.call_soon_threadsafe(
                    asyncio.create_task, onBar_next(nqt0, ib, minbar, barsList, hasNew)
                )
            else:
                print("[cb] NQ next tick but no loop")
        return _cb

    def _bind_qqq(minbar):
        def _cb(barsList, hasNew):
            try:
                print(f"[cb-fire] QQQ        len={len(barsList)} last={getattr(barsList[-1],'time',None)}")
            except Exception:
                print("[cb-fire] QQQ (no barsList meta)")
            loop = _get_running_loop_fallback()
            if loop is not None:
                loop.call_soon_threadsafe(
                    asyncio.create_task, onBar_QQQ(qqqt0, ib, minbar, barsList, hasNew)
                )
            else:
                print("[cb] QQQ tick but no loop")
        return _cb



    # 订阅（哪个 qualify 成功就订哪个）
    if nq_cur is not None:
        current_minbar = minute_bar()
        cur_bars = ib.reqRealTimeBars(nq_cur, 5, 'TRADES', False)
        cur_bars.updateEvent += _bind_current(current_minbar)
        bars_list.append(cur_bars)
    else:
        print("[subscribe] 跳过 当前季 NQ 订阅（未能 qualify）")

    if nq_next is not None:
        next_minbar = minute_bar()
        next_bars = ib.reqRealTimeBars(nq_next, 5, 'TRADES', False)
        next_bars.updateEvent += _bind_next(next_minbar)
        bars_list.append(next_bars)
    else:
        print("[subscribe] 跳过 下季 NQ 订阅（未能 qualify）")

    if qqq is not None:
        qqq_minbar = minute_bar()
        qqq_bars = ib.reqRealTimeBars(qqq, 5, 'TRADES', False)
        qqq_bars.updateEvent += _bind_qqq(qqq_minbar)
        bars_list.append(qqq_bars)
    else:
        print("[subscribe] 跳过 QQQ 订阅（未能 qualify）")

    print("订阅完成")
    return bars_list




# ==== 三条实时回调 ====
async def onBar_current(t0, ib, current_bars, barsList, hasNew):
    bar = barsList[-1]
    bar.time = bar.time.astimezone(ZoneInfo('America/New_York')).replace(tzinfo=None)
    print('x秒NQCurrent')
    if bar.time.second == 55:  # 收到一根线，整分钟收束
        print('55秒NQCurrent')
        if current_bars.start == 1:
            current_bars.start = 0
            current_bars.last_bar = 0
            current_bars.clear()
        else:
            current_bars.accumulate(bar.open_, bar.high, bar.low, bar.close, bar.volume)
            await t0.fast_march(bar.time.replace(second=0, microsecond=0),
                                current_bars.open_, current_bars.high_, current_bars.low_,
                                current_bars.close_, current_bars.volume_, 1)
            current_bars.last_bar = 0
            current_bars.clear()
    else:
        if current_bars.last_bar == 0:
            current_bars.accumulate_open(bar.open_, bar.high, bar.low, bar.close, bar.volume)
            current_bars.last_bar = 1
        else:
            current_bars.accumulate(bar.open_, bar.high, bar.low, bar.close, bar.volume)
            current_bars.last_bar = 1

    if bar.time.second == 15:  # 空闲时间维护内存
        t0.check_current_memory()


async def onBar_next(t0, ib, next_bars, barsList, hasNew):
    bar = barsList[-1]
    bar.time = bar.time.astimezone(ZoneInfo('America/New_York')).replace(tzinfo=None)
    print('x秒NQNext')
    if bar.time.second == 55:
        print('55秒NQNext')
        if next_bars.start == 1:
            next_bars.start = 0
            next_bars.last_bar = 0
            next_bars.clear()
        else:
            next_bars.accumulate(bar.open_, bar.high, bar.low, bar.close, bar.volume)
            await t0.fast_march(bar.time.replace(second=0, microsecond=0),
                                next_bars.open_, next_bars.high_, next_bars.low_,
                                next_bars.close_, next_bars.volume_, 0)
            next_bars.last_bar = 0
            next_bars.clear()
    else:
        if next_bars.last_bar == 0:
            next_bars.accumulate_open(bar.open_, bar.high, bar.low, bar.close, bar.volume)
            next_bars.last_bar = 1
        else:
            next_bars.accumulate(bar.open_, bar.high, bar.low, bar.close, bar.volume)
            next_bars.last_bar = 1

    if bar.time.second == 15:
        t0.check_next_memory()


async def onBar_QQQ(t0, ib, qqq_bars, barsList, hasNew):
    bar = barsList[-1]
    bar.time = bar.time.astimezone(ZoneInfo('America/New_York')).replace(tzinfo=None)
    print('x秒QQQ')
    if bar.time.second == 55:
        print('55秒QQQ')
        if qqq_bars.start == 1:
            qqq_bars.start = 0
            qqq_bars.last_bar = 0
            qqq_bars.clear()
        else:
            qqq_bars.accumulate(bar.open_, bar.high, bar.low, bar.close, bar.volume * 100)
            _time = bar.time.replace(second=0, microsecond=0)
            NQstatus = await trading_status.is_contract_tradable(ib, t0.t2_p.nqt0_p.current_using_contract, _time)
            await t0.fast_march(_time,
                                qqq_bars.open_, qqq_bars.high_, qqq_bars.low_,
                                qqq_bars.close_, qqq_bars.volume_, NQstatus)
            qqq_bars.last_bar = 0
            qqq_bars.clear()
    else:
        if qqq_bars.last_bar == 0:
            qqq_bars.accumulate_open(bar.open_, bar.high, bar.low, bar.close, bar.volume * 100)
            qqq_bars.last_bar = 1
        else:
            qqq_bars.accumulate(bar.open_, bar.high, bar.low, bar.close, bar.volume * 100)
            qqq_bars.last_bar = 1

    if bar.time.second == 15:
        t0.check_qqq_memory()


class main_Program:
    def __init__(self):
        global localhost
        self.ib = IB()
        self.ib.connect(localhost, 4004, clientId=3)
        print('已连接到 IB Gateway/TWS')
        self.ib.reqMarketDataType(1)
        self.ib.errorEvent += onError
        self.ib.disconnectedEvent += lambda: onDisconnect(self)

        self.t0_obj_nq = nq_live_t0(self.ib)
        self.t0_obj_qqq = qqq_live_t0(self.ib)
        self.t2_obj = live_t2()
        self.t3_obj = live_t3()
        self.t6_obj = live_t6()
        self.nn_obj = live_nn()
        self.dm_obj = live_dm(self.ib)

        self.t2_obj.link_t2t0sub(self.t0_obj_nq, self.t0_obj_qqq)
        self.t3_obj.link_t3t2sub(self.t2_obj)
        self.t6_obj.link_t6t3sub(self.t3_obj)
        self.nn_obj.link_nnt6sub(self.t6_obj)
        self.dm_obj.link_dmnnsub(self.nn_obj)

        self.bars_list = []
        self.bars_list = subscribe_contract(self.t0_obj_nq, self.t0_obj_qqq, self.ib, self.bars_list)


        def _md_diag_handler(tickers):
            now = datetime.now()
            for tk in tickers:
                try:
                    sym = getattr(tk.contract, "localSymbol", "") or getattr(tk.contract, "symbol", "")
                    last = getattr(tk, "last", None)
                    bid  = getattr(tk, "bid", None)
                    ask  = getattr(tk, "ask", None)
                    print(f"[md] {now} {sym} last={last} bid={bid} ask={ask}")
                except Exception as _:
                    print(f"[md] {now} <ticker>")

        # 只绑定一次
        if not hasattr(self, "_md_diag_bound"):
            self.ib.pendingTickersEvent += _md_diag_handler
            self._md_diag_bound = True

        # 对已 qualify 的合约请求行情（与 realtime bars 并行）
        try:
            # NQ 当前/次季（如果有的话）
            cur_str, next_str = self.t0_obj_nq.request_current_next_symbol()

            # 重新用与 subscribe_contract 相同方式 qualify，确保是同一份合约对象
            def _qualify_nq(ms):
                for ex in ("CME", "GLOBEX"):
                    q = self.ib.qualifyContracts(Future("NQ", ms, ex, "USD"))
                    if q:
                        return q[0]
                return None

            _nq_cur  = _qualify_nq(cur_str)
            _nq_next = _qualify_nq(next_str)

            if _nq_cur:
                self.ib.reqMktData(_nq_cur, "", False, False)
                print(f"[md] reqMktData NQ {cur_str}")
            if _nq_next:
                self.ib.reqMktData(_nq_next, "", False, False)
                print(f"[md] reqMktData NQ {next_str}")

            # QQQ
            _qqq = self.ib.qualifyContracts(Stock("QQQ", "SMART", "USD"))[0]
            self.ib.reqMktData(_qqq, "", False, False)
            print("[md] reqMktData QQQ")

        except Exception as e:
            print("[md] 诊断订阅失败：", e)

        self.dm_obj.run()
        print('初始化结束')

        # 捕获信号：先紧急落盘再退出
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # 可选：定时 autosave（正常时用 pandas）
        #self._autosave_stop = threading.Event()
        #self._autosave_th = threading.Thread(target=self._autosave_loop, daemon=True)
        #self._autosave_th.start()

    def _handle_signal(self, signum, frame):
        print(f"收到信号 {signum}，安全落盘后退出")
        try:
            self.safe_quick_dump()
        finally:
            os._exit(1)  # 避免在 finalizing 阶段触发复杂析构

    def _autosave_loop(self, interval_sec: int = 60):
        while not self._autosave_stop.is_set():
            time.sleep(interval_sec)
            if is_finalizing():
                break
            try:
                self.save()
            except Exception as e:
                print("autosave 失败：", e)

    def run(self):
        # 保存当前事件循环引用，并挂到 IB 对象上，供回调线程安全投递
        loop = asyncio.get_event_loop()
        setattr(self.ib, "owner_loop", loop)
        print(f"[debug] owner_loop set on IB: {loop}, running={loop.is_running()}")

        # 挂起，保持订阅不断开
        self.ib.run()

    def save(self):
        # 正常保存（依赖 pandas）
        if is_finalizing():
            # 关机阶段避免 pandas ImportError
            self.safe_quick_dump()
            return
        try:
            self.t0_obj_nq.save()
            self.t0_obj_qqq.save()
        except ImportError as e:
            msg = str(e)
            if "sys.meta_path is None" in msg or "Python is likely shutting down" in msg:
                print("save() 处于解释器关机阶段，转为 quick_dump。")
                self.safe_quick_dump()
            else:
                raise
        except Exception as e:
            print("save() 异常，降级 quick_dump：", e)
            self.safe_quick_dump()

    def safe_quick_dump(self):
        """
        紧急轻量落盘（不依赖 pandas）。要求 t0 对象实现 quick_dump_npz()。
        若不存在该方法，则在非 finalizing 时回退为 save()。
        """
        for name, obj in (("NQ", self.t0_obj_nq), ("QQQ", self.t0_obj_qqq)):
            try:
                if hasattr(obj, "quick_dump_npz"):
                    obj.quick_dump_npz()
                else:
                    # 没有 quick_dump_npz，若允许则退回 save（仅在非 finalizing）
                    if not is_finalizing():
                        obj.save()
                    else:
                        print(f"{name} 缺少 quick_dump_npz 且处于 finalizing，跳过 pandas 落盘以免崩溃。")
            except Exception as e:
                print(f"{name} quick_dump 失败：", e)


def onDisconnect(program: main_Program):
    print("子进程断线，退出以触发重启")
    # finalizing 阶段绝不触碰 pandas
    if is_finalizing():
        program.safe_quick_dump()
        os._exit(1)

    # 非 finalizing：尽量正常保存，失败降级
    try:
        program.save()
    except Exception as e:
        print("onDisconnect 保存失败，降级 quick_dump：", e)
        program.safe_quick_dump()

    # 立刻退出给外层监督程序重启
    os._exit(1)


if __name__ == '__main__':
    main_program = main_Program()
    main_program.run()
