from ib_insync import IB, Future, util
from datetime import datetime
import tkinter as tk

def main():
    global symbol,contract,ticker,summary
    # ———– IB 连接设置 ———–
    ib = IB()
    cliendId=5
    ib.connect('127.0.0.1', 4004, clientId=cliendId)
    
    # 计算当前合约年季度
    def yearseason_to_int(year, season):
        return (year - 2000) * 4 + season

    def calculate_contract_month_symbol_by_int(number):
        year = 2000 + (number // 4)
        season = number % 4
        return f"{year:04d}{['03','06','09','12'][season]}"
    
    def calculate_current_using_contract_year_season(now):
        if now.month==1 or now.month==2:
            return now.year,0
        elif now.month==4 or now.month==5:
            return now.year,1
        elif now.month==7 or now.month==8:
            return now.year,2
        elif now.month==10 or now.month==11:
            return now.year,3
        else:
            season=(now.month//3)-1
            if rtt.is_trigger_day2_pass():
                season=season+1
            if season>3:
                return now.year+1,0
            else:
                return now.year,season

    def get_current_contract():
        y, s = calculate_current_using_contract_year_season(datetime.now())
        num = yearseason_to_int(y, s)
        return calculate_contract_month_symbol_by_int(num)

    # ———– GUI 界面构建 ———–
    root = tk.Tk()
    root.title(f"IB 账户与行情监控 client={cliendId}")

    # 准备 Label
    labels = {
        'time': tk.Label(root, text="刷新时间：--", font=('Arial', 14)),
        'net_liq': tk.Label(root, text="净清算价值：--", font=('Arial', 14)),
        'last':    tk.Label(root, text="最新成交：--",  font=('Arial', 14)),
        'positions': tk.Label(root, text="持仓：--",   font=('Arial', 14), justify='left'),
        'orders': tk.Label(root, text="订单：--",   font=('Arial', 14), justify='left'),
        
    }
    for lbl in labels.values():
        lbl.pack(anchor='w', padx=10, pady=5)
    
    
    ib.reqPositions()
    summary = ib.accountSummary()
    # ———– 数据更新函数 ———–
    def update_data():
        global symbol,contract,ticker,summary
        labels['orders'].config(text="订单：\n")  
        labels['positions'].config(text="持仓：\n")  

        symbol = get_current_contract()
        contract = Future(symbol='NQ', exchange='CME', currency='USD', lastTradeDateOrContractMonth=symbol)
        ticker = ib.reqTickers(contract)[0]
        now=datetime.now()
        
        
        
        ib.sleep(1)

        try:
            labels['time'].config(text=f"刷新时间：{now} ")
            # 1. 获取账户摘要
            
            net = next((tag.value for tag in summary if tag.tag=='NetLiquidation'), '--')
            labels['net_liq'].config(text=f"净清算价值：{net} USD")

            # 2. 获取最新合约行情
            labels['last'].config(text=f"最新成交：{ticker.last}")
            
            # 3. 获取持仓列表
            pos_text = ""
            for pos in ib.positions():
                pos_text += f"{pos.contract.localSymbol:10s}  数量：{pos.position}\n"
            labels['positions'].config(text=f"持仓：\n{pos_text or '无持仓'}")

            ord_text = ""

            

            for order in ib.reqAllOpenOrders():
                trade = next((t for t in ib.openTrades() if t.order.orderId==order.orderId), None)
                if trade is not None:
                    order  = trade.order
                    status = trade.orderStatus
                    contract = trade.contract
                    if contract.secType == 'FUT':
                        ord_text += f"订单号:{order.orderId} 项目 {contract.symbol} {contract.lastTradeDateOrContractMonth} 多空{order.action} 成交态 {status.status}  数量 {order.totalQuantity} 价格 {getattr(order, 'lmtPrice', 'MKT')}\n"
                    else:
                        ord_text += f"订单号:{order.orderId} 项目 {contract.symbol} 多空 {order.action} 成交态 {status.status} 数量 {order.totalQuantity} 价格 {getattr(order, 'lmtPrice', 'MKT')}\n"
            labels['orders'].config(text=f"订单：\n{ord_text or '无活动订单'}")


    

        except Exception as e:
            print("刷新出错：", e)

        # 10秒后再次执行（10000 毫秒）
        root.after(2000, update_data)

    # 启动第一次更新
    root.after(0, update_data)

    # 进入 Tkinter 事件循环
    root.mainloop()

    # 断开 IB 连接（如果程序退出）
    ib.disconnect()

if __name__ == '__main__':
    main()