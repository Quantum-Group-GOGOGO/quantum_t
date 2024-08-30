import pandas as pd
data_base='/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
data_path=data_base+'/raw/QQQ/QQQ_full_1min_adjsplitdiv.csv'
QQQ_path=data_base+'/Type0/QQQ/QQQ_BASE_T.pkl'
# 读取CSV文件
df = pd.read_csv(data_path, header=None)

# 重命名列（如果需要）
df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

# 转换数据类型
df['datetime'] = pd.to_datetime(df['datetime'])

# 过滤掉2024年4月30日之后的数据
filter_date_1 = pd.Timestamp('2000-04-17')
filter_date_2 = pd.Timestamp('2024-05-01')
df=df[df['datetime'] >= filter_date_1]
df=df[df['datetime'] <= filter_date_2]

# 显示DataFrame的前几行
print(df.head())
print(df.tail())
df.to_pickle(QQQ_path) 