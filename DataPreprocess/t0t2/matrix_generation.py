import pandas as pd
import numpy as np
from tqdm import tqdm
data_base='/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
QQQ_path=data_base+'/Type0/QQQ/QQQ_BASE_T.pkl'
NQ_path=data_base+'/Type0/NQ/NQ_BASE.pkl'
QQQ = pd.read_pickle(QQQ_path)
NQ = pd.read_pickle(NQ_path)
NQ = NQ[NQ['volume'] != 0]
print(QQQ.head())
print(QQQ.tail())
print(NQ.head())
print(NQ.tail())
print('Merging!!!!!!!!!!!!!!!!!!')
# 合并两个 DataFrame，取交集
merged_df = pd.merge(NQ, QQQ, on='datetime', suffixes=('_NQ', '_QQQ'))
# 计算 ratio 列
merged_df['open_ratio'] = merged_df['open_NQ'] / merged_df['open_QQQ']
merged_df['high_ratio'] = merged_df['high_NQ'] / merged_df['high_QQQ']
merged_df['low_ratio'] = merged_df['low_NQ'] / merged_df['low_QQQ']
merged_df['close_ratio'] = merged_df['close_NQ'] / merged_df['close_QQQ']
merged_df['volume_ratio'] = merged_df['volume_NQ'] / merged_df['volume_QQQ']

# 选择需要的列
result_df = merged_df[['datetime', 'open_ratio', 'high_ratio', 'low_ratio', 'close_ratio', 'volume_ratio']]
print(result_df.head())
print(result_df.tail())

print('Averaging!!!!!!!!!!!')

def moving_averaging(df,column_name,window_size=5):
    # 计算移动平均
    df.loc[:, column_name+'_fm']=df[column_name].rolling(window=window_size).mean()
    df.loc[:, column_name+'_bm']=df[column_name].rolling(window=window_size).mean().shift(-(window_size-1))
    #填充最前面和最后面边缘数据产生的NAN
    #df[column_name+'_fm'].iloc[:(window_size-1)] = df[column_name].iloc[:(window_size-1)]
    #df[column_name+'_bm'].iloc[-(window_size-1):] = df[column_name].iloc[-(window_size-1):]
    df.loc[df.index[:(window_size-1)], column_name+'_bm'] = df.loc[df.index[:(window_size-1)], column_name]
    df.loc[df.index[-(window_size-1):], column_name+'_bm'] = df.loc[df.index[-(window_size-1):], column_name]


moving_averaging(result_df,'open_ratio')
moving_averaging(result_df,'high_ratio')
moving_averaging(result_df,'low_ratio')
moving_averaging(result_df,'close_ratio')
moving_averaging(result_df,'volume_ratio')
#生成新的datetime序列
result_df['datetime_1'] = result_df['datetime']
result_df['datetime_2'] = result_df['datetime']


print(result_df.head())
print(result_df.tail())
print('EXPANDING!!!!!!!!!!!!!!!!!!')
# 生成从第一个时间到最后一个时间的每分钟时间序列
full_time_index = pd.date_range(start=result_df['datetime'].min(), end=result_df['datetime'].max(), freq='min')


# 建立一个新的 DataFrame，时间序列作为索引
full_df = pd.DataFrame(index=full_time_index)

# 将原始数据合并到新的 DataFrame 中
full_df = full_df.merge(result_df.set_index('datetime'), left_index=True, right_index=True, how='left')

print(full_df.head())
print(full_df.tail())

print('INTERPOLATION')
def forward_fill(df):
    #填充前置列
    df['datetime_1']=df['datetime_1'].ffill()
    # 对所有列名以 '_fm' 结尾的列进行前向填充
    for column in df.columns:
        if column.endswith('_fm'):
            df[column] = df[column].ffill()
def backward_fill(df):
    #填充后置列
    df['datetime_2']=df['datetime_2'].bfill()
    # 对所有列名以 '_bm' 结尾的列进行前向填充
    for column in df.columns:
        if column.endswith('_bm'):
            df[column] = df[column].bfill()

forward_fill(full_df)
backward_fill(full_df)

def interpolate_open_ratio(row,index_name):
    if pd.isna(row[index_name]):
        t = row.name
        t1 = row['datetime_1']
        t2 = row['datetime_2']
        return row[index_name+'_fm'] + (row[index_name+'_bm'] - row[index_name+'_fm']) * (t - t1).total_seconds() / (t2 - t1).total_seconds()
    return row[index_name]

def interpolation_column(index_Name):
    full_df[index_Name] = full_df.progress_apply(interpolate_open_ratio, axis=1,index_name=index_Name)

# 初始化 tqdm
tqdm.pandas()
# 应用插值函数
interpolation_column('open_ratio')
interpolation_column('high_ratio')
interpolation_column('low_ratio')
interpolation_column('close_ratio')
interpolation_column('volume_ratio')


# 需要删除的列名列表
columns_to_drop = ['open_ratio_fm', 'open_ratio_bm','high_ratio_fm', 'high_ratio_bm','low_ratio_fm', 'low_ratio_bm','close_ratio_fm', 'close_ratio_bm','volume_ratio_fm', 'volume_ratio_bm']
full_df.drop(columns=columns_to_drop, inplace=True)
columns_to_drop = ['datetime_1','datetime_2']
full_df.drop(columns=columns_to_drop, inplace=True)
print(full_df.head())
print(full_df.tail())
a=600
print(full_df.iloc[a:a+5])
full_df.to_pickle(data_base+'/type2/ratio_matrix.pkl')