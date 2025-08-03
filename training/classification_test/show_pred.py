from env import *  
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datetime import datetime,timedelta,time
data_path = data_base + "/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_with_predictions_120to80_2LSTM_future3.pkl"
labeled_result_path= data_base + "/type_p1/1.9+-0.095.pkl"
# 创建测试集的 Dataset 和 DataLoader
df = pd.read_pickle(data_path)
#df.drop(df.columns.difference(['tag','tags_in', 'tags_flat', 'tags_de','prediction1', 'prediction2', 'prediction3']), axis=1, inplace=True)
df['log_ret'] = (np.log(df['close'] / df['close'].shift(1)))**2
df['volweek_raw'] = df['log_ret'].rolling(1440*7).mean()
df['close3d'] = df['close'].rolling(1440*3).mean()
df['av']=(df['high']+df['low']+df['close'])/3
df['PV']=df['av']*df['volumeO']
period=191
#period=30
df['PVc']=df['PV'].rolling(window=period, min_periods=1).sum()
df['Vc']=df['volumeO'].rolling(window=period, min_periods=1).sum()
df['vwap']=df['PVc']/df['Vc']

# 3. 定义重置时点
t1 = time(9, 30)
t2 = time(18, 0)
# ——— 9:30 开始累积的 VWAP ———
# 3.1 计算“会话日期”：若时间 < 9:30，就归到前一天
session_0930 = df['datetime'].apply(
    lambda dt: dt.date() if dt.time() >= t1 else (dt.date() - timedelta(days=1))
)
# 3.2 只在 9:30 及之后保留 PV 和 volume，其它置 0
pv_0930  = df['PV'].where(df['datetime'].dt.time >= t1, 0)
vol_0930 = df['volumeO'].where(df['datetime'].dt.time >= t1, 0)
# 3.3 分组累加
cum_pv_0930  = pv_0930.groupby(session_0930).cumsum()
cum_vol_0930 = vol_0930.groupby(session_0930).cumsum()
# 3.4 计算 VWAP
df['vwap_0930'] = cum_pv_0930 / cum_vol_0930

# ——— 18:00 开始累积的夜盘 VWAP ———
# 4.1 定义夜盘区间：18:00–次日 9:30
mask_night = (
    (df['datetime'].dt.time >= t2) |
    (df['datetime'].dt.time < t1)
)
# 4.2 会话日期同理：若时间 < 18:00，就归到前一天
session_1800 = df['datetime'].apply(
    lambda dt: dt.date() if dt.time() >= t2 else (dt.date() - timedelta(days=1))
)
# 4.3 只在夜盘区间保留 PV 和 volume
pv_1800  = df['PV'].where(mask_night, 0)
vol_1800 = df['volumeO'].where(mask_night, 0)
# 4.4 分组累加
cum_pv_1800  = pv_1800.groupby(session_1800).cumsum()
cum_vol_1800 = vol_1800.groupby(session_1800).cumsum()
# 4.5 计算 VWAP
df['vwap_1800'] = cum_pv_1800 / cum_vol_1800
import matplotlib.pyplot as plt

# 假设你的 DataFrame 名为 df，包含 'datetime' 列和 'volweek_raw' 列
# 如果 'datetime' 已设为索引，可改为 x = df.index
x = df['datetime'] if 'datetime' in df.columns else df.index
y = df['volweek_raw']



total = len(df)
start = int(total * 0.7)
end =int(total * 0.999)
#df = df.iloc[start:-1]
df = df.iloc[start:end]
print(df.tail())
#df['prediction_tag'] = df[['prediction1', 'prediction2', 'prediction3']].idxmax(axis=1).map({'prediction1': 0, 'prediction2': 1, 'prediction3': 2})
cond1 = (df['prediction2'] > -1.9) & \
        (df['prediction2'] > df['prediction1']-0.5) & \
        (df['prediction2'] > df['prediction3']-0.5)
cond2 = df['prediction1'] >= df['prediction3']
#choices = [1, 0]
choices = [2, 2]
df['prediction_tag'] = np.select([cond1, cond2], choices, default=2)
print(df.head())
print(df.tail())
#contingency = pd.crosstab(df['prediction_tag'], df['tag'])
#print(contingency)

#print(contingency.iloc[0,0]+contingency.iloc[-1,-1]-contingency.iloc[-1,0]-contingency.iloc[0,-1])


df.to_pickle(labeled_result_path)


