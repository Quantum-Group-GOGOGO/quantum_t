import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.special import expit  # 使用expit函数实现sigmoid

# 创建一个自定义 Dataset 类
class TimeSeriesLSTM1Dataset(Dataset):
    def __init__(self, df, sequence_length_1, sequence_length_10, sequence_length_60, sequence_length_240, sequence_length_1380):
        # 提取 'close' 等列并转换为 float32
        self.close = df['close'].values.astype(np.float32)
        self.close_10 = df['close_10'].values.astype(np.float32)
        self.close_60 = df['close_60'].values.astype(np.float32)
        self.close_240 = df['close_240'].values.astype(np.float32)
        self.close_1380 = df['close_1380'].values.astype(np.float32)

        # 提取 'volume' 等列并转换为 float32
        self.volume = df['volume'].values.astype(np.float32)
        self.volume_10 = df['volume_10'].values.astype(np.float32)
        self.volume_60 = df['volume_60'].values.astype(np.float32)
        self.volume_240 = df['volume_240'].values.astype(np.float32)
        self.volume_1380 = df['volume_1380'].values.astype(np.float32)

        # 提取其他数据并转换为 float32
        excluded_columns = ['close', 'close_10', 'close_60', 'close_240', 'close_1380', 'volume', 'volume_10', 'volume_60', 'volume_240', 'volume_1380', 'evaluation_30', 'evaluation_60', 'evaluation_120', 'evaluation_300', 'evaluation_480']
        other_features = [col for col in df.columns if col not in excluded_columns]
        self.other_data = df[other_features].values.astype(np.float32)

        # 提取评测数据并转换为 float32
        evaluation_columns = ['evaluation_30', 'evaluation_60', 'evaluation_120', 'evaluation_300', 'evaluation_480']
        evaluation_index = [col for col in df.columns if col in evaluation_columns]
        self.evaluation_data = df[evaluation_index].values.astype(np.float32)

        self.sequence_length_1 = sequence_length_1
        self.sequence_length_10 = sequence_length_10
        self.sequence_length_60 = sequence_length_60
        self.sequence_length_240 = sequence_length_240
        self.sequence_length_1380 = sequence_length_1380

    def __len__(self):
        # 确保所有序列都足够长，至少满足最大跳度
        return len(self.close) - self.sequence_length_1380 * 1380 - 1
        #return 1000

    def __getitem__(self, idx):
        # 计算尾端索引为idx往后数 sequence_length_1380 * 1380 的位置
        end_idx = idx + self.sequence_length_1380 * 1380

        # 根据尾端索引往前取对应长度，确保所有序列尾端对齐
        close_1 = self.close[end_idx - self.sequence_length_1 + 1:end_idx + 1]
        close_10 = self.close_10[end_idx - (self.sequence_length_10 - 1) * 10:end_idx + 1:10]
        close_60 = self.close_60[end_idx - (self.sequence_length_60 - 1) * 60:end_idx + 1:60]
        close_240 = self.close_240[end_idx - (self.sequence_length_240 - 1) * 240:end_idx + 1:240]
        close_1380 = self.close_1380[end_idx - (self.sequence_length_1380 - 1) * 1380:end_idx + 1:1380]

        # 对时间序列进行归一化
        close_1 = self.normalize_series1(close_1)
        close_10 = self.normalize_series10(close_10)
        close_60 = self.normalize_series(close_60)
        close_240 = self.normalize_series(close_240)
        close_1380 = self.normalize_series(close_1380)

        # 根据尾端索引往前取对应长度，确保所有序列尾端对齐
        volume_1 = self.volume[end_idx - self.sequence_length_1 + 1:end_idx + 1]
        volume_10 = self.volume_10[end_idx - (self.sequence_length_10 - 1) * 10:end_idx + 1:10]
        volume_60 = self.volume_60[end_idx - (self.sequence_length_60 - 1) * 60:end_idx + 1:60]
        volume_240 = self.volume_240[end_idx - (self.sequence_length_240 - 1) * 240:end_idx + 1:240]
        volume_1380 = self.volume_1380[end_idx - (self.sequence_length_1380 - 1) * 1380:end_idx + 1:1380]

        # 加载协变量
        other_data_current = self.other_data[end_idx]

        #加评测值
        evaluation_data_current = self.evaluation_data[end_idx]

        return (close_1, volume_1)
    
    def normalize_series(self, series):
        # 使用最后一个点作为基准进行归一化
        last_value = series[-1]
        # normalized_series = expit(25*(series - last_value)/last_value)  # 使用sigmoid函数归一化 4%变化对应sigmoid(1) 所以乘25倍
        normalized_series = 25*(series - last_value)/last_value  # 标准差归一化 不严格归一化
        return normalized_series

    def normalize_series1(self, series):
        # 使用最后一个点作为基准进行归一化
        last_value = series[-1]
        #normalized_series = expit(100*(series - last_value)/last_value)  # 使用sigmoid函数归一化 1%变化对应sigmoid(1) 所以乘100倍
        normalized_series = 100*(series - last_value)/last_value  # 标准差归一化 不严格归一化
        return normalized_series
    
    def normalize_series10(self, series):
        # 使用最后一个点作为基准进行归一化
        last_value = series[-1]
        #normalized_series = expit(100*(series - last_value)/last_value)  # 使用sigmoid函数归一化 1%变化对应sigmoid(1) 所以乘100倍
        normalized_series = 100*(series - last_value)/last_value  # 标准差归一化 不严格归一化
        return normalized_series