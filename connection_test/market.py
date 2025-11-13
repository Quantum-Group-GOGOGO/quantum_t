#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from datetime import datetime
from ib_insync import IB, Forex

HOST = '127.0.0.1'
PORT = 4004
CLIENT_ID = 6
PRINT_INTERVAL = 30.0  # 仅当距离上次打印 >= 30s 才打印

LAST_PRINT_MONO = None  # 模块级全局变量，记录上次打印的 monotonic 时间戳

def on_update(ticker):
    global LAST_PRINT_MONO
    now_mono = time.monotonic()

    # 首次 or 距离上次 >= 30 秒 才打印
    if (LAST_PRINT_MONO is None) or (now_mono - LAST_PRINT_MONO >= PRINT_INTERVAL):
        ts = ticker.time or datetime.now()
        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')

        bid = 'NA' if ticker.bid is None else f'{ticker.bid:.5f}'
        ask = 'NA' if ticker.ask is None else f'{ticker.ask:.5f}'
        last = 'NA' if ticker.last is None else f'{ticker.last:.5f}'
        try:
            mid = ticker.midpoint()
            mid_str = f'{mid:.5f}' if mid is not None else 'NA'
        except Exception:
            mid_str = 'NA'

        print(f'[{ts_str}] USDJPY  bid={bid}  ask={ask}  last={last}  mid={mid_str}')
        LAST_PRINT_MONO = now_mono
    else:
        # 30秒内已有打印，跳过
        pass

def main():
    ib = IB()
    print(f'Connecting to {HOST}:{PORT} with clientId={CLIENT_ID} ...')
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    print('Connected.')

    ib.reqMarketDataType(1)  # 无实时权限可改 3
    contract = Forex('USDJPY')

    ticker = ib.reqMktData(contract, '', False, False)
    ticker.updateEvent += on_update
    print('Subscribed to USD/JPY with 30s throttle. Press Ctrl+C to stop.')

    try:
        ib.run()
    except KeyboardInterrupt:
        print('Stopping...')
    finally:
        ticker.updateEvent -= on_update
        ib.cancelMktData(ticker)
        ib.disconnect()
        print('Disconnected.')

if __name__ == '__main__':
    main()