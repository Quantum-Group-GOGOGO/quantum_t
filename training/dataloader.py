import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# 创建一个自定义 Dataset 类
class TimeSeriesDataset(Dataset):
    def __init__(self, close, other_data, sequence_length, close_10_length, close_100_length):
        self.close = close
        self.other_data = other_data
        self.sequence_length = sequence_length
        self.close_10_length = close_10_length
        self.close_100_length = close_100_length

    def __len__(self):
        return len(self.close) - self.sequence_length - (self.close_10_length - 1) * 10 - (self.close_100_length - 1) * 100

    def __getitem__(self, idx):
        # 提取 close_1 (前300个点加当前点)
        close_1 = self.close[idx:idx + self.sequence_length + 1]

        # 提取 close_10 (从当前开始，每10个点取1个，共100个点)
        close_10 = self.close[idx:idx + self.close_10_length * 10:10]

        # 提取 close_100 (从当前开始，每100个点取1个，共100个点)
        close_100 = self.close[idx:idx + self.close_100_length * 100:100]

        # 提取协变量数据，只取当前点
        other_data_current = self.other_data[idx + self.sequence_length]

        return (close_1, close_10, close_100, other_data_current)


# 将下面的代码放在 main 块中
if __name__ == "__main__":
    # 假设你已经有一个加载好的 DataFrame 'df'
    data_base='/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
    T4_data_path=data_base+'/type5/Nasdaq_qqq_align_labeled_base_evaluated_300_001_test.pkl'
    df = pd.read_pickle(T4_data_path)
    
    # 丢弃 'datetime' 列
    #df = df.drop(columns=['datetime'], errors='ignore')
    
    # 提取 'close' 列
    close = df['close'].values
    print('hello1')
    
    # 定义时间序列长度
    sequence_length = 300  # 前300个点，加上当前点一共301个点
    close_10_length = 100  # 每10个点取一个，取100个点
    close_100_length = 100  # 每100个点取一个，取100个点
    print('hello2')
    
    # 创建 Dataset 和 DataLoader
    other_features = [col for col in df.columns if col != 'close']
    other_data = df[other_features].values
    print('hello3')
    dataset = TimeSeriesDataset(close, other_data, sequence_length, close_10_length, close_100_length)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    print('hello4')
    
    # 查看一个样本
    sample_close_1, sample_close_10, sample_close_100, sample_other = next(iter(dataloader))
    print("Close_1 Sequence:", sample_close_1.shape)
    print("Close_10 Sequence:", sample_close_10.shape)
    print("Close_100 Sequence:", sample_close_100.shape)
    print("Other Data:", sample_other.shape)