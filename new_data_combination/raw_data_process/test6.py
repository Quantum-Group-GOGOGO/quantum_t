import asyncio
from ib_insync import IB, Stock, Future

def print_loop(tag):
    loop = asyncio.get_running_loop()
    print(f"{tag} running in loop {hex(id(loop))}")

def subscribeOne(ib: IB):
    bars = ib.reqRealTimeBars(
        contract=Future(symbol='NQ',exchange='CME',currency='USD',lastTradeDateOrContractMonth='202512'),
        barSize=5, whatToShow='TRADES', useRTH=False
    )
    # 在回调里打印 loop
    bars.updateEvent += lambda lst, hasNew: (
        print_loop("updateEvent"),
        print(f"最新价：{lst[-1].close}")
    )

if __name__ == "__main__":
    ib = IB()
    # 启动前拿到主 loop
    main_loop = asyncio.get_event_loop()
    print(f"Main loop is {hex(id(main_loop))}")

    ib.connect('127.0.0.1', 4003, clientId=1)
    subscribeOne(ib)

    # run() 底层就是 run_forever() on that same loop
    ib.run()