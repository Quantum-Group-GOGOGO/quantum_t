from dataloader_LSTM1 import TimeSeriesLSTM1Dataset
from torch.utils.data import DataLoader
import pandas as pd

# 将下面的代码放在 main 块中
if __name__ == "__main__":
    # 假设你已经有一个加载好的 DataFrame 'df'
    data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
    T6_data_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1.pkl'

    df = pd.read_pickle(T6_data_path)

    # 定义时间序列长度
    sequence_length_1 = 120
    #sequence_length_10 = 100
    #sequence_length_60 = 60
    #sequence_length_240 = 60
    #sequence_length_1380 = 60

    # 创建 Dataset 和 DataLoader
    dataset = TimeSeriesLSTM1Dataset(df, sequence_length_1, sequence_length_10,
                                sequence_length_60, sequence_length_240, sequence_length_1380)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    # 查看一个样本
    sample_close_1, sample_volume_1 = next(iter(dataloader))
    print("Close_1 Sequence:", sample_close_1.shape)
    print("Volume_1 Sequence:", sample_volume_1.shape)
    #print("Close_10 Sequence:", sample_close_10.shape)
    #print("Close_60 Sequence:", sample_close_60.shape)
    #print("Close_240 Sequence:", sample_close_240.shape)
    #print("Close_1380 Sequence:", sample_close_1380.shape)
    #print("Other Data:", sample_other.shape)
    #print("Evaluation Data:", sample_evaluation.shape)