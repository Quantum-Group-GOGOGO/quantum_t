import pandas as pd
from env import *

# 1. 读取原来的 pkl 文件
QQQ_type0_path=live_data_base+'/type0/QQQ/'
current_filename = 'QQQ_BASE.pkl'
file_path = QQQ_type0_path+current_filename
df = pd.read_pickle(file_path)

# 2. 删除最后 1000 行
#    如果 df 一共少于或等于 1000 行，就得到一个空的 DataFrame
if len(df) > 16000:
    df_trimmed = df.iloc[:-16000]
else:
    df_trimmed = df.iloc[0:0]  # 返回空 DataFrame，保留原来的列名和索引

# 3. 将截断后的 DataFrame 覆盖写回原文件
df_trimmed.to_pickle(file_path)