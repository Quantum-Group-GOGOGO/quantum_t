from env import *  
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from types import SimpleNamespace

labeled_result_path= data_base + "/type_p1/1.9+-0.095.pkl"
df = pd.read_pickle(labeled_result_path)
df['test'] = df['datetime'].shift(-1)
df['d_price'] = df['open'].shift(-1)
#df['d_price'] = df['close']
df['next_high'] = df['high'].shift(-1)
df['next_low'] = df['low'].shift(-1)
df = df.iloc[:-1]
#df.loc[:, 'prediction_tag'] = 2
df.iloc[-1, df.columns.get_loc('prediction_tag')] = 1

valve=0.04/100
ref_price=df['close'].iloc[1]
profit = 0.0
status = 1   # 起始状态
in_price = 0.0
trade=0
long_profit=0
short_profit=0
d_price = 0
in_tick = 0
tick_life_time = 20
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
    spread=5.00*2*trades
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
    spread=0.50*2*trades
    return (commission+CME_NFA_commission+spread)/trades

def flat_to_long():
    global profit, status, in_price, trade, long_profit, short_profit, d_price, in_tick
    status = 2
    in_price = d_price
    in_tick = 0

def flat_to_short():
    global profit, status, in_price, trade, long_profit, short_profit, d_price, in_tick
    status = 0
    in_price = d_price
    in_tick = 0

def long_to_flat(d_price):
    global profit, status, in_price, trade, long_profit, short_profit
    profit += (d_price - in_price)
    long_profit += (d_price - in_price)
    status = 1
    trade +=1

def short_to_flat(d_price):
    global profit, status, in_price, trade, long_profit, short_profit
    profit -= (d_price - in_price)
    short_profit -= (d_price - in_price)
    status = 1
    trade +=1

def check_status():
    global status, tag, next_high, next_low, d_price
    if status == 1:
        if tag == 0:
            #if next_high<d_price:
                flat_to_short()
        elif tag == 2:
            #if next_low>d_price:
                flat_to_long()


for idx in tqdm(df.index, desc="Processing rows"):
    tag     = df.at[idx, 'prediction_tag']
    d_price = df.at[idx, 'd_price']
    high_price = df.at[idx, 'high']
    low_price = df.at[idx, 'low']
    next_high = df.at[idx, 'next_high']
    next_low = df.at[idx, 'next_low']
    
    
    # —— 根据当前状态 status 与当前行的 tag 来决定新的 profit / status / in_price ——
    if status == 1:
        check_status()

    elif status == 0:
        in_tick += 1
        if(high_price-in_price > in_price*valve):
            short_to_flat(in_price+in_price*valve)
            check_status()
        elif(in_price - low_price > in_price*valve):
            short_to_flat(in_price-in_price*valve)
            check_status()
        elif(in_tick>=tick_life_time):
            short_to_flat(d_price)
            check_status()


    elif status == 2:
        in_tick += 1
        if(high_price-in_price > in_price*valve):
            long_to_flat(in_price+in_price*valve)
            check_status()
        elif(in_price - low_price > in_price*valve):
            long_to_flat(in_price-in_price*valve)
            check_status()
        elif(in_tick>=tick_life_time):
            long_to_flat(d_price)
            check_status()

profit=profit/ref_price
long_profit=long_profit/ref_price
short_profit=short_profit/ref_price
monthly_trades=trade/28.8
average_cost_MNQ=commission_calculation_MNQ(monthly_trades)
average_cost_NQ=commission_calculation_NQ(monthly_trades)

print('total profit: ',profit)
print('long profit: ',long_profit)
print('short profit: ',short_profit)

print('total trade: ',trade)
print('total consume MNQ: ',trade*average_cost_MNQ/43464.50)
print('total consume NQ: ',trade*average_cost_NQ/434645.0)

# 循环结束后，df 的这三列就已经记录了每一步的变化
# 如果你只关心最后的 profit，可以直接用外部变量 profit；否则可以看 df['profit'].iloc[-1]。