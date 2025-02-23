from LSTM1 import LSTMAutoencoder
import torch
import pandas as pd
import matplotlib.pyplot as plt
from dataloader_LSTMTS import TimeSeriesLSTMTSDataset
from torch.utils.data import DataLoader

# 读取测试集数据
data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
#data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
test_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_test.pkl'

# 定义时间序列长度
sequence_length_1 = 120
sequence_length_10 = 120
sequence_length_60 = 120
sequence_length_240 = 120
sequence_length_1380 = 120

# 创建测试集的 Dataset 和 DataLoader
test_df = pd.read_pickle(test_path)
test_dataset = TimeSeriesLSTMTSDataset(test_df, sequence_length_1, sequence_length_10,
                                      sequence_length_60, sequence_length_240, sequence_length_1380)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

# 定义模型参数
input_size = 2  # 每个时间步的特征数量（两个变量：Close_1 和 Volume_1）
hidden_size = 60
num_layers = 2
encoded_size = 40

# 实例化模型
model = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size)

# 加载训练好的模型参数
model_path = data_base + '/models/lstm1_encoder/LSTMAutoencoder_trained2.pth'
model.load_state_dict(torch.load(model_path))
model.eval()  # 切换到评估模式

# 获取一个测试样本并进行重构
with torch.no_grad():
    # 从测试集中获取一个样本
    (close_1, close_10, close_60, close_240, close_1380,
     volume_1, volume_10, volume_60, volume_240, volume_1380, evaluation_data_current) = next(iter(test_dataloader))

    # 合并输入特征（close 和 volume）
    x_sample = torch.cat((close_10.unsqueeze(-1), volume_10.unsqueeze(-1)), dim=2).float()  # (1, sequence_length, input_size)

    # 通过模型进行前向传播，得到重构的输出
    reconstructed_output = model(x_sample)

    # 获取第一个样本的原始和重构序列（由于 batch_size=1，取出第一个即可）
    original_sequence = x_sample.squeeze(0).numpy()  # 转换为 numpy 数组以便绘图
    reconstructed_sequence = reconstructed_output.squeeze(0).numpy()

    # 绘制原始序列和重构序列
    plt.figure(figsize=(12, 6))

    # 绘制 Close_1 原始和重构序列的曲线
    plt.subplot(2, 1, 1)
    plt.plot(original_sequence[:, 0], label='Original Close_10', color='blue')
    plt.plot(reconstructed_sequence[:, 0], label='Reconstructed Close_10', color='red', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Close_10 Value')
    plt.title('Original vs Reconstructed Close_10')
    plt.legend()

    # 绘制 Volume_1 原始和重构序列的曲线
    plt.subplot(2, 1, 2)
    plt.plot(original_sequence[:, 1], label='Original Volume_10', color='blue')
    plt.plot(reconstructed_sequence[:, 1], label='Reconstructed Volume_10', color='red', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Volume_10 Value')
    plt.title('Original vs Reconstructed Volume_10')
    plt.legend()

    # 显示图像
    plt.tight_layout()
    plt.show()