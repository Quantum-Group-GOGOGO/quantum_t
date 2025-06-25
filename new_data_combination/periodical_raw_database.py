from ib_insync import IB, Future
from datetime import datetime
import re

# 1) 连接 TWS/Gateway
ib = IB()
ib.connect('127.0.0.1', 7597, clientId=1)
contract = Future(
    symbol='NQ',
    exchange='CME',                # NQ 在 CME GLOBEX
    currency='USD',
    lastTradeDateOrContractMonth='202509'  
)
details_list = ib.reqContractDetails(contract)
if not details_list:
    raise ValueError("No ContractDetails returned")
cd = details_list[0]
print("Trading hours string:", cd.tradingHours)

m = re.match(r'(\d{8})', cd.tradingHours)
if m:
    expiry_date = datetime.strptime(m.group(1), '%Y%m%d').date()
    print("Expiry date:", expiry_date)
else:
    print("Cannot parse expiry date from tradingHours")