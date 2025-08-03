import asyncio,threading
from ib_insync import IB, Future, MarketOrder,LimitOrder
from datetime import time,datetime, timedelta

def is_on_working_time(dt: datetime) -> bool:
    """
    判断给定的 datetime dt 的时间是否不在 07:30–18:00（含）范围内。
    返回 True 则表示在 07:30 之前或 18:00 之后（即“不在”区间内）。
    """
    t = dt.time()
    start = time(8, 7)
    end   = time(18, 0)
    return t < start or t > end

def monthCode(contract_month: str) -> str:
    """
    contract_month: 格式 'YYYYMM'，例如 '202509'、'202603'
    返回：交割月份代码 + 年份末位，例如 'U5'、'H6'
    """
    # 月份到代码的映射
    code_map = {
        '03': 'NQH',
        '06': 'NQM',
        '09': 'NQU',
        '12': 'NQZ',
    }
    year = contract_month[:4]      # '2025'
    month = contract_month[4:6]    # '09'
    code = code_map.get(month)
    if code is None:
        raise ValueError(f"不支持的月份：{month}")
    # 年份第四位，也就是 year[3]
    return f"{code}{year[3]}"

class live_dm:
    def __init__(self,ibob):
        self.ib=ibob
        self.buy_sended=0
        self.sell_sended=0
        self.x=0.02/100
        self.y=0.05/100

        
        return

    def link_dmnnsub(self, nn_processor):
        self.nn_p=nn_processor
        self.nn_p.link_dmobj(self)
        self.nnBase=self.nn_p.nnBase
        self.contract_renew()
        self.in_time=datetime.now()
        self.flat_time=datetime.now()
        ticker = self.ib.reqTickers(self.contract)[0]
        self.ib.sleep(1)
        self.in_price=ticker.last
        self.flat_price=ticker.last
        

        self.ib.reqPositions()
        self.ib.sleep(0.4)
        self.contract_renew()
        position=0
        for pos in self.ib.positions():
            # pos.contract 是合约对象，pos.position 是数量（正=多，负=空）
            if pos.contract.localSymbol== self.cString:
                print(f"{pos.contract.localSymbol:10s}  持仓量：{pos.position}")
                position=pos.position
    
        if position>0:
            self.position=1
        else:
            self.position=0

        self.ib.reqAllOpenOrders()
        self.ib.sleep(1)
        for trade in self.ib.openTrades():
            order  = trade.order
            print(f"取消订单 {order.orderId}：{order.action} {order.totalQuantity} @ {getattr(order, 'lmtPrice', 'MKT')}")
            self.ib.cancelOrder(order)
            self.buy_sended=0
            self.sell_sended=0
        self.ib.sleep(1)
        self.ib.reqGlobalCancel()

    def run(self):

        def thread_target():
            asyncio.run(self.decition())
        t = threading.Thread(target=thread_target, daemon=True)
        t.start()

    async def liveRenew(self,decide):
        if decide==1 and is_on_working_time(datetime.now()):
            if self.nn_p.t6_p.t3_p.t2_p.leap==0:
                low=self.nn_p.t6_p.t3_p.t2_p.nqt0_p.current_contract_data['low'].iloc[-1]
                high=self.nn_p.t6_p.t3_p.t2_p.nqt0_p.current_contract_data['high'].iloc[-1]
            else:
                low=self.nn_p.t6_p.t3_p.t2_p.nqt0_p.next_contract_data['low'].iloc[-1]
                high=self.nn_p.t6_p.t3_p.t2_p.nqt0_p.next_contract_data['high'].iloc[-1]
            if self.position==0:
                if self.nn_p.t6_p.t6Base['volweek_raw'].iloc[-1] < 2.2e-7:
                    # if self.buy_sended==1:
                    #     limitNeed=round(4*(self.flat_price*(1-self.x)))/4
                    #     now=datetime.now()
                    #     diff_minute=(now-self.in_time).total_seconds() / 60
                    #     print(f'等待买入限价单 {limitNeed} 成交中 已存在{diff_minute}分钟')
                    now=datetime.now()
                    diff_minute=(now-self.flat_time).total_seconds() / 60
                    if diff_minute>20.2:
                        if self.nn_p.t6_p.t3_p.t2_p.leap==0:
                            self.flat_price=self.nn_p.t6_p.t3_p.t2_p.nqt0_p.current_contract_data['close'].iloc[-1]
                        else:
                            self.flat_price=self.nn_p.t6_p.t3_p.t2_p.nqt0_p.next_contract_data['close'].iloc[-1]

                        result=await self.marketBuy(1)
                        if result.orderStatus.status !='Cancelled':
                            self.buy_sended=0
                            print('买一手市价')
                            self.position=1
                    else:
                        if low<self.flat_price*(1-self.x):
                            result=await self.marketBuy(1)
                            if result.orderStatus.status !='Cancelled':
                                self.buy_sended=0
                                print('买一手市价')
                                self.position=1
                        else:
                            print(f'等待价格低于{self.flat_price*(1-self.x)}时购买')

            elif self.position==1:
                # if self.sell_sended==1:
                #     limitNeed=round(4*(self.in_price*(1+self.y)))/4
                #     now=datetime.now()
                #     diff_minute=(now-self.flat_time).total_seconds() / 60
                #     print(f'等待卖出限价单 {limitNeed} 成交中 已存在{diff_minute}分钟')
                

                now=datetime.now()
                diff_minute=(now-self.in_time).total_seconds() / 60
                if diff_minute>110:
                    if self.nn_p.t6_p.t3_p.t2_p.leap==0:
                        self.in_price=self.nn_p.t6_p.t3_p.t2_p.nqt0_p.current_contract_data['close'].iloc[-1]
                    else:
                        self.in_price=self.nn_p.t6_p.t3_p.t2_p.nqt0_p.next_contract_data['close'].iloc[-1]
                    
                    result=await self.marketSell(1)
                    if result.orderStatus.status !='Cancelled':
                        self.sell_sended=0
                        print('卖一手市价')
                        self.position=0
                else:
                    if high>self.in_price*(1+self.y):
                        result=await self.marketSell(1)
                        if result.orderStatus.status !='Cancelled':
                            self.sell_sended=0
                            print('卖一手市价')
                            self.position=0
                    else:
                            print(f'等待价格高于{self.in_price*(1+self.y)}时卖出')

            await self.ib.reqPositionsAsync()
            await asyncio.sleep(1)
            self.contract_renew()
            for pos in self.ib.positions():
                # pos.contract 是合约对象，pos.position 是数量（正=多，负=空）
                if pos.contract.localSymbol== self.cString:
                    print(f"{pos.contract.localSymbol:10s}  持仓量：{pos.position}")
        elif decide==1 and self.position==1:
            result=await self.marketSell(1)
            if result.orderStatus.status !='Cancelled':
                self.sell_sended=0
                print('卖一手市价')
                self.position=0
        return
    
    def contract_renew(self):
        num=self.nn_p.t6_p.t3_p.t2_p.now_num
        monthstr=self.nn_p.t6_p.t3_p.t2_p.nqt0_p.calculate_contract_month_symbol_by_int(num)
        self.contract = Future(symbol='NQ',exchange='CME',currency='USD',lastTradeDateOrContractMonth=monthstr)
        self.cString = monthCode(monthstr)

    async def marketBuy(self,quantity):
        order = MarketOrder('BUY', quantity)
        trade = self.ib.placeOrder(self.contract, order)

        while trade.orderStatus.status not in ('Filled', 'Cancelled'):
            await asyncio.sleep(1)
        if trade.orderStatus.status =='Filled':
            self.in_time=datetime.now()
            self.in_price=trade.orderStatus.avgFillPrice
        return trade

    async def marketSell(self,quantity):
        order = MarketOrder('SELL', quantity)
        trade = self.ib.placeOrder(self.contract, order)

        while trade.orderStatus.status not in ('Filled', 'Cancelled'):
            await asyncio.sleep(1)
        if trade.orderStatus.status =='Filled':
            self.flat_time=datetime.now()
            self.flat_price=trade.orderStatus.avgFillPrice
        return trade

    def limitBuy(self,quantity,price):
        order = LimitOrder(action='BUY',totalQuantity=quantity,lmtPrice=price)
        trade = self.ib.placeOrder(self.contract, order)
        return trade
    
    def limitSell(self,quantity,price):
        order = LimitOrder(action='SELL',totalQuantity=quantity,lmtPrice=price)
        trade = self.ib.placeOrder(self.contract, order)
        return trade

    async def decition(self):
        

        while True:
            await asyncio.sleep(1)  # 等待 IB 推送持仓数据
            time_last=self.nn_p.t6_p.t6Base.index[-1]
            if not is_on_working_time(time_last):
                if self.buy_sended==1:
                    self.ib.cancelOrder(self.buytrade.order)
                    self.buy_sended=0
                    print(f'非活动时间取消买入限价单')
                if self.sell_sended==1:
                    self.ib.cancelOrder(self.selltrade.order)
                    self.sell_sended=0
                    print(f'非活动时间取消卖出限价单')
                if self.position == 1:
                    await self.marketSell(1)
                    print('非活动时间卖一手市价')
                    self.position=0
            



