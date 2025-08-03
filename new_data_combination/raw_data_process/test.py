#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ib_insync import IB, util

def main():
    # 1. 创建 IB 客户端并连接
    ib = IB()
    # 默认 TWS 端口是 7496，Gateway 默认为 4001
    # 请根据你本地的 TWS/Gateway 配置修改 host/port
    ib.connect('127.0.0.1', 4004, clientId=7)

    ib.reqAllOpenOrders()
    ib.sleep(1)
    # 2. 拉取当前 clientId 下的所有 open orders
    #    如果想拉取所有 clientId 的 open orders，可改用 ib.reqAllOpenOrders()
    openTrades = ib.openTrades()

    # 3. 输出结果
    if not openTrades:
        print('当前没有未成交订单。')
    else:
        print(f'共 {len(openTrades)} 笔未成交订单：\n')
        for trade in openTrades:
            order  = trade.order
            status = trade.orderStatus
            contract = trade.contract

            print(f'• OrderId: {order.orderId}')
            print(f'  Symbol:  {contract.symbol}')
            if contract.secType == 'FUT':
                print(f'  LastDate:  {contract.lastTradeDateOrContractMonth}')
            print(f'  Action:  {order.action}  Qty: {order.totalQuantity}')
            # limitOrder 的挂单价，市价单则为 None
            print(f'  Limit:   {order.lmtPrice}')
            print(f'  Status:  {status.status}   Filled: {status.filled} / {order.totalQuantity}')
            print('')

    # 4. 断开连接
    ib.disconnect()

if __name__ == '__main__':
    main()
