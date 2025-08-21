import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.special import expit  # 使用expit函数实现sigmoid

# 创建一个自定义 Dataset 类
class TimeSeriesLSTMTSDataset(Dataset):
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

        self.high = df['high'].values.astype(np.float32)
        self.low = df['low'].values.astype(np.float32)
        self.open = df['open'].values.astype(np.float32)
        # 提取评测数据并转换为 float32
        #evaluation_columns = ['evaluation_30', 'evaluation_60', 'evaluation_120', 'evaluation_300', 'evaluation_480']
        #evaluation_columns = ['tags_in', 'tags_flat', 'tags_de']
        evaluation_columns = ['tag']
        evaluation_index = [col for col in df.columns if col in evaluation_columns]
        self.evaluation_data = df[evaluation_index].values.astype(np.float32)

        # 提取其他数据并转换为 float32
        auxiliary_columns = ['sinT', 'cosT', 'pre_event', 'post_event', 'pre_break', 'post_break', 'week_fraction_sin', 'week_fraction_cos', 'absolute_time']
        auxiliary_index = [col for col in df.columns if col in auxiliary_columns]
        self.auxiliary_data = df[auxiliary_index].values.astype(np.float32)

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
        # 计算尾端索引
        end_idx = idx + self.sequence_length_1380 * 1380

        # ---- 价格序列（按不同步长） ----
        high_1  = self.high[end_idx - self.sequence_length_1 + 1 : end_idx + 1]
        low_1   = self.low[end_idx - self.sequence_length_1 + 1 : end_idx + 1]
        open_1  = self.open[end_idx - self.sequence_length_1 + 1 : end_idx + 1]
        close_1 = self.close[end_idx - self.sequence_length_1 + 1 : end_idx + 1]

        close_10   = self.close_10[  end_idx - (self.sequence_length_10  - 1) * 10   : end_idx + 1 : 10]
        close_60   = self.close_60[  end_idx - (self.sequence_length_60  - 1) * 60   : end_idx + 1 : 60]
        close_240  = self.close_240[ end_idx - (self.sequence_length_240 - 1) * 240  : end_idx + 1 : 240]
        close_1380 = self.close_1380[end_idx - (self.sequence_length_1380- 1) * 1380 : end_idx + 1 : 1380]

        # ---- 归一化（全部保证为 float32 ndarray）----
        high_1  = np.asarray(self.normalize_asyseries_1(high_1,  close_1),  dtype=np.float32)
        low_1   = np.asarray(self.normalize_asyseries_1(low_1,   close_1),  dtype=np.float32)
        open_1  = np.asarray(self.normalize_asyseries_1(open_1,  close_1),  dtype=np.float32)
        close_1 = np.asarray(self.normalize_series1(close_1),               dtype=np.float32)
        close_10   = np.asarray(self.normalize_series10(close_10),          dtype=np.float32)
        close_60   = np.asarray(self.normalize_series60(close_60),          dtype=np.float32)
        close_240  = np.asarray(self.normalize_series240(close_240),        dtype=np.float32)
        close_1380 = np.asarray(self.normalize_series1380(close_1380),      dtype=np.float32)

        # ---- 成交量序列 ----
        volume_1   = self.volume[      end_idx - self.sequence_length_1 + 1 : end_idx + 1]
        volume_10  = self.volume_10[   end_idx - (self.sequence_length_10  - 1) * 10   : end_idx + 1 : 10]
        volume_60  = self.volume_60[   end_idx - (self.sequence_length_60  - 1) * 60   : end_idx + 1 : 60]
        volume_240 = self.volume_240[  end_idx - (self.sequence_length_240 - 1) * 240  : end_idx + 1 : 240]
        volume_1380= self.volume_1380[ end_idx - (self.sequence_length_1380- 1) * 1380 : end_idx + 1 : 1380]

        volume_1    = np.asarray(volume_1,    dtype=np.float32)
        volume_10   = np.asarray(volume_10,   dtype=np.float32)
        volume_60   = np.asarray(volume_60,   dtype=np.float32)
        volume_240  = np.asarray(volume_240,  dtype=np.float32)
        volume_1380 = np.asarray(volume_1380, dtype=np.float32)

        # ---- 协变量 / 评测 ----
        # evaluation_data: 如果是 ['tag']，则 end_idx 后得到形状 (1,)
        evaluation_data_current = self.evaluation_data[end_idx]
        evaluation_data_current = np.asarray(evaluation_data_current, dtype=np.float32).reshape(-1)

        auxiliary_data_current  = self.auxiliary_data[end_idx]
        auxiliary_data_current  = np.asarray(auxiliary_data_current,  dtype=np.float32).reshape(-1)

        # ---- 统一转成 torch.float32 ----
        def to_t(x):
            # 所有 numpy 数组 / 标量都转张量；若不小心传入了“类型对象”，这里会报清楚的错
            if isinstance(x, type):
                raise TypeError(f"Detected a dtype/type object in dataset output: {x}. "
                                f"Return values (arrays/scalars), not types.")
            if isinstance(x, np.generic):  # numpy 标量
                x = np.asarray(x, dtype=np.float32)
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(torch.float32)
            if isinstance(x, (float, int)):
                return torch.tensor(x, dtype=torch.float32)
            # 其它类型（极少见）统一兜底
            return torch.as_tensor(x, dtype=torch.float32)

        return (
            to_t(close_1), to_t(close_10), to_t(close_60), to_t(close_240), to_t(close_1380),
            to_t(volume_1), to_t(volume_10), to_t(volume_60), to_t(volume_240), to_t(volume_1380),
            to_t(evaluation_data_current), to_t(auxiliary_data_current),
            to_t(high_1), to_t(low_1), to_t(open_1),
        )
        
    def normalize_asyseries_1(self, series, asyseries):
        # 使用最后一个点作为基准进行归一化
        last_value = asyseries[-1]
        # normalized_series = expit(25*(series - last_value)/last_value)  # 使用sigmoid函数归一化 4%变化对应sigmoid(1) 所以乘25倍
        normalized_series = 100*(series - last_value)/last_value  # 标准差归一化 不严格归一化
        return normalized_series
    
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
    
    def normalize_series60(self, series):
        # 使用最后一个点作为基准进行归一化
        last_value = series[-1]
        #normalized_series = expit(25*(series - last_value)/last_value)  # 使用sigmoid函数归一化 1%变化对应sigmoid(1) 所以乘100倍
        normalized_series = 25*(series - last_value)/last_value  # 标准差归一化 不严格归一化
        return normalized_series
    
    def normalize_series240(self, series):
        # 使用最后一个点作为基准进行归一化
        last_value = series[-1]
        #normalized_series = expit(25*(series - last_value)/last_value)  # 使用sigmoid函数归一化 1%变化对应sigmoid(1) 所以乘100倍
        normalized_series = 25*(series - last_value)/last_value  # 标准差归一化 不严格归一化
        return normalized_series
    
    def normalize_series1380(self, series):
        # 使用最后一个点作为基准进行归一化
        last_value = series[-1]
        #normalized_series = expit(25*(series - last_value)/last_value)  # 使用sigmoid函数归一化 1%变化对应sigmoid(1) 所以乘100倍
        normalized_series = 25*(series - last_value)/last_value  # 标准差归一化 不严格归一化
        return normalized_series