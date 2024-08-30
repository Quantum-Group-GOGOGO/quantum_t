import pandas as pd
import numpy as np
from tqdm import tqdm

data_base='/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
T4_data_path=data_base+'/type4/Nasdaq_qqq_align_labeled_base_evaluated_300.pkl'

T5_data_path=data_base+'/type5/Nasdaq_qqq_align_labeled_base_evaluated_300_001.pkl'
T5_data_path_test=data_base+'/type5/Nasdaq_qqq_align_labeled_base_evaluated_300_001_test.pkl'

df=pd.read_pickle(T4_data_path)
# 丢弃 'datetime' 列
df = df.drop(columns=['datetime'], errors='ignore')
# 小型切片用于测试

df.to_pickle(T5_data_path)
df.iloc[0:20000].to_pickle(T5_data_path_test)