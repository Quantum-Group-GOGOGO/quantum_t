import torch
import pandas as pd
import matplotlib.pyplot as plt
from LSTM1 import LSTMEncoder  # 只加载 Encoder
from dataloader_LSTMTS import TimeSeriesLSTMTSDataset
from torch.utils.data import DataLoader

# 读取测试集数据
data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
test_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_test.pkl'

# **统一时间序列长度**
sequence_length = 120

# 创建测试集 Dataset 和 DataLoader
test_df = pd.read_pickle(test_path)
test_dataset = TimeSeriesLSTMTSDataset(test_df, sequence_length, sequence_length, sequence_length, sequence_length, sequence_length)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

# **模型参数**
input_size = 1  # 每个时间步只有一个变量 (单独处理 close 和 volume)
hidden_size = 60
num_layers = 2
encoded_size = 40

# **创建共享的 LSTMEncoder**
encoder = LSTMEncoder(input_size, hidden_size, num_layers, encoded_size)

# **加载训练好的模型权重**
model_path = data_base + '/models/lstm1_encoder/LSTMAutoencoder_trained2.pth'
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
encoder.load_state_dict(state_dict, strict=False)
encoder.eval()  # 设为评估模式

# **获取一个测试样本**
with torch.no_grad():
    # **取一个随机样本**
    (close_1, close_10, close_60, close_240, close_1380,
     volume_1, volume_10, volume_60, volume_240, volume_1380, evaluation_data_current) = next(iter(test_dataloader))

    # **合并 10 个时间序列**
    sequences = [close_1, close_10, close_60, close_240, close_1380,
                 volume_1, volume_10, volume_60, volume_240, volume_1380]

    # **逐个编码 10 个序列**
    encoded_vectors = []
    for seq in sequences:
        seq_encoded = encoder(seq.unsqueeze(-1).float())  # 形状: (1, 40)
        encoded_vectors.append(seq_encoded)

    # **最终拼接为 (1, 400) 维向量**
    final_encoded_vector = torch.cat(encoded_vectors, dim=-1)  # 形状: (1, 400)

    # **打印最终编码向量**
    print("Encoded Vector (Shape: 1 × 400):")
    print(final_encoded_vector.squeeze(0).numpy())  # 转换为 NumPy 并打印

    # **绘制 10 条原始时间序列**
    plt.figure(figsize=(12, 8))
    time_steps = list(range(sequence_length))  # 统一长度 120

    # **绘制 5 个 close**
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, close_1.squeeze(0).numpy(), label='Close_1', linestyle='-')
    plt.plot(time_steps, close_10.squeeze(0).numpy(), label='Close_10', linestyle='--')
    plt.plot(time_steps, close_60.squeeze(0).numpy(), label='Close_60', linestyle='-.')
    plt.plot(time_steps, close_240.squeeze(0).numpy(), label='Close_240', linestyle=':')
    plt.plot(time_steps, close_1380.squeeze(0).numpy(), label='Close_1380', linestyle='-.')
    plt.xlabel('Time Step')
    plt.ylabel('Close Price')
    plt.title('Original Close Time Series')
    plt.legend()

    # **绘制 5 个 volume**
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, volume_1.squeeze(0).numpy(), label='Volume_1', linestyle='-')
    plt.plot(time_steps, volume_10.squeeze(0).numpy(), label='Volume_10', linestyle='--')
    plt.plot(time_steps, volume_60.squeeze(0).numpy(), label='Volume_60', linestyle='-.')
    plt.plot(time_steps, volume_240.squeeze(0).numpy(), label='Volume_240', linestyle=':')
    plt.plot(time_steps, volume_1380.squeeze(0).numpy(), label='Volume_1380', linestyle='-.')
    plt.xlabel('Time Step')
    plt.ylabel('Volume')
    plt.title('Original Volume Time Series')
    plt.legend()

    # **显示图像**
    plt.tight_layout()
    plt.show()
