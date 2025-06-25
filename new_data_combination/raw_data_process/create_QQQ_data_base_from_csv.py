import pandas as pd
from env import *
# 这个脚本理论上只要运行一次，它会从历史数据的csv中读取数据并转化成pkl的数据保存，这个pkl的数据是raw QQQ数据的原始先前行，
csv_data_path=data_base+'/raw/QQQ/QQQ_full_1min_adjsplitdiv.csv'
QQQ_type0_path=live_data_base+'/type0/QQQ/QQQ_BASE.pkl'
cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']

df = pd.read_csv(
    csv_data_path,           # 或者你的文件路径
    header=None,          # 表示文件中没有表头行
    names=cols,           # 给每列命名
    parse_dates=['datetime'], # 自动把 time 列解析成 datetime 类型
    index_col='datetime'      # （可选）把 time 列设为索引
)

# 查看前几行，确认类型
print(df.head())
print(df.tail())
print(df.index.dtype)
df.to_pickle(QQQ_type0_path)