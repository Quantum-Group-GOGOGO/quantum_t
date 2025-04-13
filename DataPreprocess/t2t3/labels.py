#1 时间标签
#2 大事件标签
#3 事件前标签
#4 事件后标签

#5 时间断点标签
#6 时间断点前标签
#7 时间断点后标签

#8 不带断点和事件评价(直接往后数N个点)
#9 带断点和事件评价(收敛到断点处和事件处)

import printh as ph
import pandas as pd
import numpy as np

# 数据读取
data_base='/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
T2_data_path=data_base+'/type2/Nasdaq_qqq_align_base.pkl'
df = pd.read_pickle(T2_data_path)
printH=ph.PrintH(df)
printH.add_hidden_column('high')
printH.add_hidden_column('low')
print(df.head())
#1 时间标签
# 计算当天的总秒数
seconds_in_day = 24 * 60 * 60

# 提取当前时间在当天的秒数
df['seconds_since_midnight'] = df['datetime'].dt.hour * 3600 + df['datetime'].dt.minute * 60 + df['datetime'].dt.second

# 归一化秒数到 [0, 1]
df['time_fraction'] = df['seconds_since_midnight'] / seconds_in_day

# 删除中间列（可选）
df.drop(columns=['seconds_since_midnight'], inplace=True)


#计算正弦
# 新增一列，计算 sin(time_fraction * 2 * pi)
df['sinT'] = np.sin(df['time_fraction'] * 2 * np.pi)
#计算余弦
# 新增一列，计算 cos(time_fraction * 2 * pi)
df['cosT'] = np.cos(df['time_fraction'] * 2 * np.pi)

printH.add_hidden_column('time_fraction')
printH.add_hidden_column('sinT')
printH.add_hidden_column('cosT')


# 大事件标签
# 初始化 'event' 列为 0
df['event'] = 0


# 对整个 DataFrame 进行 shift 操作，得到前后时间点的 close 值
df['close_prev'] = df['close'].shift(1)
df['close_next'] = df['close'].shift(-1)

# 筛选出时间为 08:30:00 的行
df_0830 = df[df['datetime'].dt.time == pd.to_datetime('08:30:00').time()]

# 计算前后 close 值的变化百分比，并更新 'event' 列
df.loc[df_0830.index, 'event'] = np.where(
    abs(df_0830['close_next'] - df_0830['close_prev']) / df_0830['close_prev'] > 0.005, 1, 0
)
# 第一行自动填充成1
df.iloc[0, df.columns.get_loc('event')] = 1
# 最后一行自动填充成1
df.iloc[-1, df.columns.get_loc('event')] = 1
# 删除临时列
df.drop(columns=['close_prev', 'close_next'], inplace=True)

print('Finish Event Label')
# 生成双向事件间隔标记
# 计算 pre_event
df['pre_event'] = (df['datetime'] - df[df['event'] == 1]['datetime'].reindex(df.index, method='ffill')).dt.total_seconds() / 60

# 计算 post_event
df['post_event'] = (df[df['event'] == 1]['datetime'].reindex(df.index, method='bfill') - df['datetime']).dt.total_seconds() / 60

# 确保 event=1 的行的 pre_event 和 post_event 都为 0
df.loc[df['event'] == 1, ['pre_event', 'post_event']] = 0

# 填充所有 NaN 为 0（如果有任何遗漏）
df['pre_event'] = df['pre_event'].fillna(0)
df['post_event'] = df['post_event'].fillna(0)

# 求时间断点

# 计算前一个时间点的时间差
df['time_diff'] = df['datetime'].diff().abs()

# 计算后一个时间点的时间差
df['time_diff_next'] = df['datetime'].diff(-1).abs()

# 生成 time_break_flag 列
df['time_break_flag'] = np.where(
    (df['time_diff'] > pd.Timedelta(hours=4)) | (df['time_diff_next'] > pd.Timedelta(hours=4)),
    1,
    0
)
# 第一行自动填充成1
df.iloc[0, df.columns.get_loc('time_break_flag')] = 1
# 最后一行自动填充成1
df.iloc[-1, df.columns.get_loc('time_break_flag')] = 1

# 求时间断点的距离
# 删除临时列
df.drop(columns=['time_diff', 'time_diff_next'], inplace=True)

# 生成双向事件间隔标记
# 计算 pre_break
df['pre_break'] = (df['datetime'] - df[df['time_break_flag'] == 1]['datetime'].reindex(df.index, method='ffill')).dt.total_seconds() / 60

# 计算 post_break
df['post_break'] = (df[df['time_break_flag'] == 1]['datetime'].reindex(df.index, method='bfill') - df['datetime']).dt.total_seconds() / 60

# 确保 time_break_flag=1 的行的 pre_break 和 post_break 都为 0
df.loc[df['time_break_flag'] == 1, ['pre_break', 'post_break']] = 0

# 填充所有 NaN 为 0（如果有任何遗漏）
df['pre_break'] = df['pre_break'].fillna(0)
df['post_break'] = df['post_break'].fillna(0)
printH.print()
T3_data_path=data_base+'/type3/Nasdaq_qqq_align_labeled_base.pkl'
df.to_pickle(T3_data_path)
T3_data_path=data_base+'/type3/Nasdaq_qqq_align_labeled_base_test.pkl'
df.iloc[0:2000].to_pickle(T3_data_path)