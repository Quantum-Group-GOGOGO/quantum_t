from env import *  
import argparse
import torch
import pandas as pd
import numpy as np
from datetime import time,datetime, timedelta
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from types import SimpleNamespace
import matplotlib.pyplot as plt

labeled_result_path= data_base + "/type_p1/1.9+-0.095.pkl"
df = pd.read_pickle(labeled_result_path)
df['in_market'] = df['datetime'].dt.time.between(time(9, 31), time(16, 0)).astype(int)
df['d_price'] = df['open'].shift(-1)
#df['d_price'] = df['close']
df['next_high'] = df['high'].shift(-1)
df['next_low'] = df['low'].shift(-1)
df['vwap_l'] = df['vwap'].shift(1)
df = df.iloc[:-1]
#df.loc[:, 'prediction_tag'] = 2
df.iloc[-1, df.columns.get_loc('prediction_tag')] = 1
x=0.05
long_win_valve=x/100
long_lose_valve=x/100
short_win_valve=x/100
short_lose_valve=x/100
ref_price=df['close'].iloc[1]
profit = 0.0
status = 1   # 起始状态
in_price = 0.0
trade=0
long_profit=0
short_profit=0
long_win=0
short_win=0
long_lose=0
short_lose=0
long_win_time=0
long_lose_time=0
short_win_time=0
short_lose_time=0
d_price = 0
in_tick = 0
historical_max = 0
money = 10000
position = 0
lv = 3.0
max_back = 0
time_to_long = 0
time_to_short = 0
time_to_hold = 0
tick_life_time = 110

trade_logs = []  # 每笔交易的记录列表
current_trade = {}  # 临时存储当前开仓信息

def commission_calculation_NQ(trades):
    if trades<=1000:
        commission = 0.85*trades
    elif trades<=10000:
        commission = 0.65*(trades-1000)+0.85*1000
    elif trades<=20000:
        commission = 0.45*(trades-10000)+0.65*(10000-1000)+0.85*1000
    else:
        commission = 0.25*(trades-20000)+0.45*(20000-10000)+0.65*(10000-1000)+0.85*1000
    CME_NFA_commission = 1.40*trades
    spread=5.00*1*trades
    return (commission+CME_NFA_commission+spread)/trades
    
def commission_calculation_MNQ(trades):
    if trades<=1000:
        commission = 0.25*trades
    elif trades<=10000:
        commission = 0.20*(trades-1000)+0.25*1000
    elif trades<=20000:
        commission = 0.15*(trades-10000)+0.20*(10000-1000)+0.25*1000
    else:
        commission = 0.10*(trades-20000)+0.15*(20000-10000)+0.20*(10000-1000)+0.25*1000
    CME_NFA_commission = 0.22*trades
    spread=0.50*1*trades
    return (commission+CME_NFA_commission+spread)/trades

def flat_to_long(price):
    global profit, status, in_price, trade, long_profit, short_profit, d_price, in_tick, current_idx,current_trade,money,position,lv
    status = 2
    in_price = price
    in_tick = 0
    current_trade = {
        'type': 'long',
        'start_time': current_idx,
        'start_price': in_price
    }
    position = lv*money/price

def flat_to_short(price):
    global profit, status, in_price, trade, long_profit, short_profit, d_price, in_tick, current_idx,current_trade,money,position,lv
    status = 0
    in_price = price
    in_tick = 0
    current_trade = {
        'type': 'short',
        'start_time': current_idx,
        'start_price': in_price
    }
    position = lv*money/price

def long_to_flat(d_price):
    global profit, status, in_price, trade, long_profit, short_profit,long_win,long_lose,long_win_time,long_lose_time, historical_max, max_back, current_idx,current_trade, in_tick,money,position,lv
    profit += (d_price - in_price) * position
    money += (d_price - in_price) * position
    if money > historical_max:
        historical_max = money
    if money < historical_max:
        callback = (historical_max - money)/historical_max
        if callback > max_back:
            max_back = callback
    long_profit += (d_price - in_price) * position
    if d_price - in_price>0:
        long_win += d_price - in_price
        long_win_time += 1
    else:
        long_lose += d_price - in_price
        long_lose_time += 1
    status = 1
    trade +=1
    trade_logs.append({
        'type': current_trade['type'],
        'start_time': current_trade['start_time'],
        'end_time': current_idx,
        'duration': (current_idx - current_trade['start_time']).total_seconds() / 60,  # 单位：分钟
        'profit': (d_price - in_price) * position/10000
    })
    in_tick = 0

def short_to_flat(d_price):
    global profit, status, in_price, trade, long_profit, short_profit,short_win,short_lose,short_win_time,short_lose_time, historical_max, max_back, current_idx,current_trade, in_tick,money,position,lv
    profit -= (d_price - in_price) * position
    money -= (d_price - in_price) * position
    if money > historical_max:
        historical_max = money
    if money < historical_max:
        callback = (historical_max - money)/historical_max
        if callback > max_back:
            max_back = callback
    short_profit -= (d_price - in_price) * position
    if d_price - in_price < 0:
        short_win -= d_price - in_price
        short_win_time += 1
    else:
        short_lose -= d_price - in_price
        short_lose_time += 1
    status = 1
    trade +=1
    trade_logs.append({
        'type': current_trade['type'],
        'start_time': current_trade['start_time'],
        'end_time': current_idx,
        'duration': (current_idx - current_trade['start_time']).total_seconds() / 60,
        'profit': (- d_price + in_price) * position/10000
    })
    in_tick = 0
    
def flat_to_wait():
    global waittime,status
    waittime=0
    status=3

def long_to_wait():
    global waittime1,status
    waittime1=0
    status=4

def check_status():
    global status, tag, next_high, next_low, d_price, in_market, time_to_short, time_to_hold, time_to_long, current_idx, current_trade, varing_rate, close3d, close1d, flat_price,waittime,open_price
    global in_tick,pre_event,post_event
    if status == 1:
        if in_market == 0:
            if varing_rate < 2.2e-7:
                if tag == 0:
                    #if next_high<d_price:
                        flat_to_short(d_price)
                        time_to_short += 1
                elif tag == 2:
                    #if next_low>d_price:
                    if 'flat_price' in globals():
                        if low_price<=flat_price*(1-(0.010/100)):
                            #flat_to_long(flat_price*(1-(0.00/100)))
                            #if low_price-d_price>0:
                                #print(f'差异较大 d_price={d_price} low_price={low_price}')
                            flat_to_wait()
                            #flat_to_long(d_price)
                            #time_to_long += 1
                    else:
                        flat_to_long(d_price)
                        time_to_long += 1
                elif tag == 1:
                        time_to_hold += 1
    elif status==3:
        waittime+=1
        if waittime>0:
            flat_to_long(open_price)
            time_to_long += 1


for idx in tqdm(df.index, desc="Processing rows"):
    tag     = df.at[idx, 'prediction_tag']
    close_price = df.at[idx, 'close']
    open_price = df.at[idx, 'open']
    d_price = df.at[idx, 'd_price']
    high_price = df.at[idx, 'high']
    low_price = df.at[idx, 'low']
    next_high = df.at[idx, 'next_high']
    next_low = df.at[idx, 'next_low']
    in_market = df.at[idx, 'in_market']
    current_idx = df.at[idx, 'datetime']
    varing_rate = df.at[idx, 'volweek_raw']
    close3d = df.at[idx, 'close3d']
    close1d = df.at[idx, 'close_1380']
    pre_event = df.at[idx, 'pre_event']
    post_event = df.at[idx, 'post_event']
    #vwap=df.at[idx, 'vwap']
    vwap=df.at[idx, 'vwap_0930']
    #if idx==df.index[-1]:
        #long_to_flat(d_price)
    # —— 根据当前状态 status 与当前行的 tag 来决定新的 profit / status / in_price ——
    valve=0.00/100
    if status == 1:
        in_tick += 1
        if in_market==1:
            if close_price>vwap*(1+valve):
                flat_to_long(d_price)
            elif close_price<vwap*(1-valve):
                flat_to_short(d_price)



    #如果close小于'vwap'
    elif status == 0:
        in_tick += 1
        if in_market==1:
            if close_price>vwap*(1+valve):
                short_to_flat(d_price)
            if close_price>vwap*(1+valve):
                flat_to_long(d_price)
        else:
            short_to_flat(d_price)



    elif status == 2:
        in_tick += 1
        if in_market==1:
            if close_price<vwap*(1-valve):
                long_to_flat(d_price)
            if close_price<vwap*(1-valve):
                flat_to_short(d_price)
        else:
            long_to_flat(d_price)
    

profit=profit
long_profit=long_profit
short_profit=short_profit
monthly_trades=trade/28.8
average_cost_MNQ=commission_calculation_MNQ(monthly_trades)
average_cost_NQ=commission_calculation_NQ(monthly_trades)


print('long profit: ',long_profit)
print('long_win: ',long_win)
print('long_lose: ',long_lose)
print('long_win_time: ',long_win_time)
print('long_lose_time: ',long_lose_time)
print(' ')

print('short profit: ',short_profit)
print('short_win: ',short_win)
print('short_lose: ',short_lose)
print('short_win_time: ',short_win_time)
print('short_lose_time: ',short_lose_time)
print(' ')

print('total trade: ',trade)
print('time to long: ',time_to_long)
print('time to hold: ',time_to_hold)
print('time to short: ',time_to_short)
print('total profit: ',profit)
print('max_back: ',max_back)
print('money: ', money)
print('total consume MNQ: ',trade*average_cost_MNQ/43464.50)
print('total consume NQ: ',trade*average_cost_NQ/434645.0)

df_trades = pd.DataFrame(trade_logs)
df_trades.to_csv('trade_log.csv', index=False)

# 2. 每日汇总日志
# 根据“前一日17:30–当日17:30”划分
def assign_day(ts: pd.Timestamp):
    # 如果时间在当日17:30之后，就算作下一天
    cutoff = ts.replace(hour=17, minute=30, second=0)
    if ts >= cutoff:
        return (ts + timedelta(days=1)).date()
    else:
        return ts.date()

df_trades['day'] = df_trades['end_time'].apply(assign_day)
daily = df_trades.groupby('day').agg(
    trades_opened = ('type','count'),
    max_profit    = ('profit','max'),
    max_loss      = ('profit','min'),
    win_count     = ('profit', lambda x: (x>0).sum()),
    loss_count    = ('profit', lambda x: (x<0).sum()),
    total_pl      = ('profit','sum')
).reset_index()
daily['cum_pl'] = daily['total_pl'].cumsum()
daily['cum_trades'] = daily['trades_opened'].cumsum()


daily.to_csv('daily_log.csv', index=False)

fig, ax1 = plt.subplots()
ax1.plot(daily['day'], daily['cum_pl'], label='Cumulative P&L')
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative P&L')
    
ax2 = ax1.twinx()
ax2.plot(daily['day'], daily['cum_trades'], label='Trades Opened',color='red')
ax2.set_ylabel('Trades Opened')
    
# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
# 优化布局与日期显示
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
# 循环结束后，df 的这三列就已经记录了每一步的变化
# 如果你只关心最后的 profit，可以直接用外部变量 profit；否则可以看 df['profit'].iloc[-1]。