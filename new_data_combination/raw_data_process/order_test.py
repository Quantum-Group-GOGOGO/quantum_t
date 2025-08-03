from ib_insync import IB, Future, MarketOrder,LimitOrder,util


# 创建IB客户端并连接
ib = IB()
ib.connect('127.0.0.1', 4004, clientId=6)

# 定义USD/JPY外汇合约
contract = Future(symbol='NQ',exchange='CME',currency='USD',lastTradeDateOrContractMonth='202512')  # 基础货币=USD, 报价货币=JPY

# 创建市场指令：卖出9USD兑换约1000JPY
ticker = ib.reqMktData(contract, snapshot=True)
ib.sleep(1)  # 等待市场数据

price = ticker.last or ticker.close
quantity = 1

#order = MarketOrder('SELL', quantity)
order = LimitOrder(action='BUY',totalQuantity=quantity,lmtPrice=20874)
trade = ib.placeOrder(contract, order)

# 等待订单完成
while trade.orderStatus.status not in ('Filled', 'Cancelled'):
    ib.sleep(3)
    print(f"订单状态: {trade.orderStatus.status}")
    ib.sleep(10)
    ib.cancelOrder(trade.order)
    ib.sleep(3)

print(f"订单状态: {trade.orderStatus.status}, 已成交数量: {trade.orderStatus.filled} USD, 平均成交价格: {trade.orderStatus.avgFillPrice}")

ib.sleep(10)  # 等待 IB 推送持仓数据




ib.disconnect()