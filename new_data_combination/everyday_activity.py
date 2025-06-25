#1确定好当天的NQ当季和NQ下季的合同号和文件号
#2拿到两天的QQQ
#3拿到两天的NQ当季和NQ下季
from ib_insync import *
from recording_time_trigger import *
from datetime import datetime, date, timedelta

del main

def calculate_current_contract_year_season(now):
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
        if is_trigger_day_pass():
            season=season+1
        if season>3:
            return now.year+1,0
        else:
            return now.year,season

def calculate_next_contract_year_season(current_contract_year,current_contract_season):
    if current_contract_season == 3:
        return current_contract_year+1,0
    else:
        return current_contract_year,current_contract_season+1

def obtain_two_days_NQ(contract_year,contract_season):
    database=''

ib = IB()
ib.connect('127.0.0.1', 7597, clientId=1)
ib.reqMarketDataType(1)

now = datetime.now()
#得到当前时间下的当季和下季的NQ月份代码
current_contract_year,current_contract_season=calculate_current_contract_year_season(now)
next_contract_year,next_contract_season=calculate_next_contract_year_season(current_contract_year,current_contract_season)
#获取两日的当前合约数据

#获取两日的下级合约数据
