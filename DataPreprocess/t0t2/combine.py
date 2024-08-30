import pandas as pd
import numpy as np
from tqdm import tqdm
data_base='/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
QQQ_path=data_base+'/Type0/QQQ/QQQ_BASE_T.pkl'
NQ_path=data_base+'/Type0/NQ/NQ_BASE.pkl'
matrix_path=data_base+'/Type2/ratio_matrix.pkl'
QQQ = pd.read_pickle(QQQ_path)
NQ = pd.read_pickle(NQ_path)
matrix = pd.read_pickle(matrix_path)
matrix = matrix.rename_axis('datetime')
matrix = matrix.reset_index()
matrix.rename(columns={'open_ratio': 'open', 'high_ratio': 'high', 'low_ratio': 'low', 'close_ratio': 'close', 'volume_ratio': 'volume'}, inplace=True)
#print(QQQ.head())
#print(NQ.head())
#print(QQQ.tail())
#print(NQ.tail(100))
#print(matrix.head())
#print(matrix.tail())
#print(matrix.columns)
#full_time_index = pd.date_range(start=matrix['datetime'].min(), end=matrix['datetime'].max(), freq='min')
#full_matrix = pd.DataFrame(full_time_index, columns=['datetime'])
#print(full_matrix.head())
#print(full_matrix.tail())

# 合并两个 DataFrame，保留共有的 datetime
merged_df = pd.merge(NQ, matrix, on='datetime', suffixes=('_NQ', '_matrix'))

# 生成新的 各个 列，计算 NQ 的 各个 除以 matrix 的 各个列
merged_df['open'] = merged_df['open_NQ'] / merged_df['open_matrix']
merged_df['low'] = merged_df['low_NQ'] / merged_df['low_matrix']
merged_df['high'] = merged_df['high_NQ'] / merged_df['high_matrix']
merged_df['close'] = merged_df['close_NQ'] / merged_df['close_matrix']
merged_df['volume'] = merged_df['volume_NQ'] / merged_df['volume_matrix']

# 选择需要的列生成新的 DataFrame NQ_align
NQ_align = merged_df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
#print(NQ_align.tail())
# 此处的NQ_align就是通过NQ转变来的对齐过QQQ的数值

#--------华丽的分割线--------

# 合并两个 DataFrame，取并集，保留共有的 datetime
Nasdaq_qqq_align = pd.merge(NQ_align, QQQ, on='datetime', how='outer', suffixes=('_NQ', '_QQQ'))
#Nasdaq_qqq_align = Nasdaq_qqq_align.iloc[10000:20000]
# 优先使用 NQ_align 的数据，如果 NQ_align 中没有数据，则使用 QQQ 的数据
for column in ['open', 'high', 'low', 'close', 'volume']:
    Nasdaq_qqq_align[column] = Nasdaq_qqq_align[column + '_NQ'].combine_first(Nasdaq_qqq_align[column + '_QQQ'])

# 删除不再需要的多余列
Nasdaq_qqq_align = Nasdaq_qqq_align[['datetime', 'open', 'high', 'low', 'close', 'volume']]

# 空缺行填充

# 定义4小时的时间间隔
time_gap = pd.Timedelta(hours=4)

# 创建最终数据的列表
all_data = []

# 添加第一个时间点的数据信息
all_data.append(Nasdaq_qqq_align.iloc[0].to_dict())

# 遍历原始 DataFrame 的行
for i in tqdm(range(len(Nasdaq_qqq_align) - 1)):
    current_row = Nasdaq_qqq_align.iloc[i]
    next_row = Nasdaq_qqq_align.iloc[i + 1]
    
    # 检查两个时间点之间的差值是否小于4小时
    if next_row['datetime'] - current_row['datetime'] <= time_gap:
        # 生成缺失的时间点
        missing_times = pd.date_range(
            start=current_row['datetime'] + pd.Timedelta(minutes=1),
            end=next_row['datetime'] - pd.Timedelta(minutes=1),
            freq='min'
        )
        
        # 将新生成的时间点和对应的填充数据添加到列表
        for time_point in missing_times:
            all_data.append({
                'datetime': time_point,
                'open': current_row['open'],
                'high': current_row['high'],
                'low': current_row['low'],
                'close': current_row['close'],
                'volume': 0.0
            })
    
    # 添加下一行数据
    all_data.append(next_row.to_dict())
# 使用 pd.DataFrame 直接构建最终的 DataFrame
final_df = pd.DataFrame(all_data)

# 显示结果
print(final_df)
final_df.to_pickle(data_base+'/type2/Nasdaq_qqq_align_base.pkl')