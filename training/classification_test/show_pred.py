from env import *  
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
data_path = data_base + "/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_with_predictions_120to80_2LSTM_future2.pkl"
labeled_result_path= data_base + "/type_p1/1.9+-0.095.pkl"
# 创建测试集的 Dataset 和 DataLoader
df = pd.read_pickle(data_path)
#df.drop(df.columns.difference(['tag','tags_in', 'tags_flat', 'tags_de','prediction1', 'prediction2', 'prediction3']), axis=1, inplace=True)

total = len(df)
start = int(total * 0.9)
df = df.iloc[start:-1]


#df['prediction_tag'] = df[['prediction1', 'prediction2', 'prediction3']].idxmax(axis=1).map({'prediction1': 0, 'prediction2': 1, 'prediction3': 2})
cond1 = (df['prediction2'] > -1.9) & \
        (df['prediction2'] > df['prediction1']) & \
        (df['prediction2'] > df['prediction3'])
cond2 = df['prediction1'] >= df['prediction3']
choices = [1, 0]
df['prediction_tag'] = np.select([cond1, cond2], choices, default=2)
print(df.head())
print(df.tail())
contingency = pd.crosstab(df['prediction_tag'], df['tag'])
print(contingency)

print(contingency.iloc[0,0]+contingency.iloc[-1,-1]-contingency.iloc[-1,0]-contingency.iloc[0,-1])


df.to_pickle(labeled_result_path)

