from env import *  
import argparse
import torch
import pandas as pd
import numpy as np
from datetime import time,timedelta
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from types import SimpleNamespace
import matplotlib.pyplot as plt

labeled_result_path= data_base + "/type_p1/1.9+-0.095.pkl"
df = pd.read_pickle(labeled_result_path)
df['in_market'] = df['datetime'].dt.time.between(time(7, 30), time(16, 0)).astype(int)
df['d_price'] = df['open'].shift(-1)
#df['d_price'] = df['close']
df['next_high'] = df['high'].shift(-1)
df['next_low'] = df['low'].shift(-1)
df = df.iloc[:-1]
#df.loc[:, 'prediction_tag'] = 2
df.iloc[-1, df.columns.get_loc('prediction_tag')] = 1

long_win_valve=0.05/100
long_lose_valve=0.05/100
short_win_valve=0.05/100
short_lose_valve=0.05/100
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
historical_max=0
max_back=0
time_to_long=0
time_to_short=0
time_to_hold=0
tick_life_time = 100


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

def flat_to_long():
    global profit, status, in_price, trade, long_profit, short_profit, d_price, in_tick,current_trade
    status = 2
    in_price = d_price
    in_tick = 0
    current_trade = {
        'type': 'long',
        'start_time': current_idx,
        'start_price': in_price
    }

def flat_to_short():
    global profit, status, in_price, trade, long_profit, short_profit, d_price, in_tick,current_trade
    status = 0
    in_price = d_price
    in_tick = 0
    current_trade = {
        'type': 'short',
        'start_time': current_idx,
        'start_price': in_price
    }

def long_to_flat(d_price):
    global profit, status, in_price, trade, long_profit, short_profit,long_win,long_lose,long_win_time,long_lose_time, historical_max, max_back,trade_logs, current_idx,current_trade, in_tick
    profit += (d_price - in_price)
    if profit > historical_max:
        historical_max = profit
    if profit < historical_max:
        callback = historical_max - profit
        if callback > max_back:
            max_back = callback
    long_profit += (d_price - in_price)
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
        'profit': d_price - in_price
    })

def short_to_flat(d_price):
    global profit, status, in_price, trade, long_profit, short_profit,short_win,short_lose,short_win_time,short_lose_time, historical_max, max_back,trade_logs, current_idx,current_trade, in_tick
    profit -= (d_price - in_price)
    if profit > historical_max:
        historical_max = profit
    if profit < historical_max:
        callback = historical_max - profit
        if callback > max_back:
            max_back = callback
    short_profit -= (d_price - in_price)
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
        'profit': - d_price + in_price
    })

def check_status():
    global status, tag, next_high, next_low, d_price, in_market, time_to_short, time_to_hold, time_to_long
    if status == 1:
        if in_market == 0:
            if tag == 0:
                #if next_high<d_price:
                    flat_to_short()
                    time_to_short += 1
            elif tag == 2:
                #if next_low>d_price:
                    flat_to_long()
                    time_to_long += 1
            elif tag == 1:
                    time_to_hold += 1


for idx in tqdm(df.index, desc="Processing rows"):
    tag     = df.at[idx, 'prediction_tag']
    d_price = df.at[idx, 'd_price']
    high_price = df.at[idx, 'high']
    low_price = df.at[idx, 'low']
    next_high = df.at[idx, 'next_high']
    next_low = df.at[idx, 'next_low']
    in_market = df.at[idx, 'in_market']
    current_idx = df.at[idx, 'datetime']
    
    
    # —— 根据当前状态 status 与当前行的 tag 来决定新的 profit / status / in_price ——
    if status == 1:
        check_status()

    elif status == 0:
        in_tick += 1
        if( (high_price-in_price > in_price*short_lose_valve) & (in_price - low_price > in_price*short_win_valve) ): #reaches both win and lose
            short_to_flat(d_price)
            check_status()
        elif(in_price - low_price > in_price*short_win_valve): #reaches win only
            short_to_flat(d_price)
            check_status()
        #elif(high_price-in_price > in_price*short_lose_valve): #reaches lose only
            #short_to_flat(in_price+in_price*short_lose_valve)
            #check_status()
        elif(in_tick>=tick_life_time):
            short_to_flat(d_price)
            check_status()


    elif status == 2:
        in_tick += 1
        if( (high_price-in_price > in_price*long_win_valve) & (in_price - low_price > in_price*long_lose_valve) ): #reaches both win and lose
            long_to_flat(d_price)
            check_status()
        elif(high_price-in_price > in_price*long_win_valve): #reaches win only
            long_to_flat(d_price)
            check_status()
        #elif(in_price - low_price > in_price*long_lose_valve): #reaches lose only
            #long_to_flat(in_price-in_price*long_lose_valve)
            #check_status()
        elif(in_tick>=tick_life_time):
            long_to_flat(d_price)
            check_status()

profit=profit/ref_price
max_back=max_back/ref_price
long_profit=long_profit/ref_price
short_profit=short_profit/ref_price
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