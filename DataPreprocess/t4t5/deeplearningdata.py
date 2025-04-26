import pandas as pd
import numpy as np
from tqdm import tqdm

#data_base='/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
#T4_data_path=data_base+'/type4/Nasdaq_qqq_align_labeled_base_evaluated.pkl'
T4_data_path = data_base + '/type4/Nasdaq_qqq_align_labeled_base_evaluated_history.pkl'

T5_data_path=data_base+'/type5/Nasdaq_qqq_align_labeled_base_evaluated.pkl'
T5_data_path_test=data_base+'/type5/Nasdaq_qqq_align_labeled_base_evaluated_test.pkl'

df=pd.read_pickle(T4_data_path)
#df=df.iloc[0:20000]
# 每周有 7 天，每天占 1/7
seconds_in_week = 7 * 24 * 60 * 60  # 每周的总秒数

# 定义一个函数来计算 week_fraction
def calculate_week_fraction(dt):
    # 一周中的第几天（周一是 0，周日是 6）
    day_of_week = dt.weekday()  # 0 到 6
    
    # 当天已经经过的秒数
    seconds_in_day = dt.hour * 3600 + dt.minute * 60 + dt.second
    
    # 计算该时刻在本周中的秒数
    seconds_in_week_at_this_time = day_of_week * 24 * 3600 + seconds_in_day
    
    # 计算该时刻占一周的比例
    return seconds_in_week_at_this_time / seconds_in_week

df['week_fraction'] = df['datetime'].apply(calculate_week_fraction)

#计算sin和cos的值
# 计算 week_fraction_sin 和 week_fraction_cos
df['week_fraction_sin'] = 0.5*(np.sin(df['week_fraction'] * 2 * np.pi)+1)
df['week_fraction_cos'] = 0.5*(np.cos(df['week_fraction'] * 2 * np.pi)+1)

#把SinT和CosT也投射到0到1
df['sinT']=0.5*(df['sinT']+1)
df['cosT']=0.5*(df['cosT']+1)

#计算从2000年1月1日到现在的分钟数
# 定义基准时间 2000年1月1日00:00
reference_time = pd.Timestamp('2000-01-01 00:00:00')

# 计算每个时间点距离 2000 年 1 月 1 日 00:00 的分钟数
df['absolute_time'] = (df['datetime'] - reference_time).dt.total_seconds() / 60

#加权平均volume
df['volume_10'] = df['volume'].rolling(window=10, min_periods=1).mean()
df['volume_60'] = df['volume'].rolling(window=60, min_periods=1).mean()
df['volume_240'] = df['volume'].rolling(window=240, min_periods=1).mean()
df['volume_1380'] = df['volume'].rolling(window=1830, min_periods=1).mean()

#算年平均
#df['volume_YEAR'] = df['volume'].rolling(window=347760, min_periods=1).mean()
window_size = 347760
# 计算滚动加和
#df['volume_YEAR'] = df['volume'].rolling(window=window_size, min_periods=window_size).sum()
# 初始化一个空列表来保存滚动和的结果
rolling_sum = [None] * len(df)

# 计算第一个窗口的和
current_sum = df['volume'].iloc[:window_size].sum()
rolling_sum[window_size - 1] = current_sum

# 使用滑动和的方法计算后续窗口的值
for i in range(window_size, len(df)):
    current_sum += df['volume'].iloc[i]  # 加入新的点
    current_sum -= df['volume'].iloc[i - window_size]  # 减去离开窗口的点
    rolling_sum[i] = current_sum

# 将结果添加到 DataFrame 中
df['volume_YEAR'] = rolling_sum

# 用第 347760 个点的值填充前面的 NaN
df['volume_YEAR'] = df['volume_YEAR'].fillna(df['volume_YEAR'].iloc[window_size - 1])

#加权平均close
df['close_10'] = df['close'].rolling(window=10, min_periods=1).mean()
df['close_60'] = df['close'].rolling(window=60, min_periods=1).mean()
df['close_240'] = df['close'].rolling(window=240, min_periods=1).mean()
df['close_1380'] = df['close'].rolling(window=1830, min_periods=1).mean()
#df['close_YEAR'] = df['close'].rolling(window=347760, min_periods=1).mean()


# 丢弃 'datetime' 列
df = df.drop(columns=['datetime','open','high','low'], errors='ignore')
# 小型切片用于测试
print(df.head())
# 找到第一个 NaN 的位置
nan_location = df.isna().stack().idxmax() if df.isna().values.any() else None

if nan_location:
    row, col = nan_location
    print(f"第一个 NaN 值位于: 行 {row}，列 '{col}'")
else:
    print("DataFrame 中没有 NaN 值。")
df.to_pickle(T5_data_path)
df.iloc[0:20000].to_pickle(T5_data_path_test)