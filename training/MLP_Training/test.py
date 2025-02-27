import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader_LSTMTS import TimeSeriesLSTMTSDataset
from LSTM1 import LSTMEncoder  # 只加载 Encoder
from MLP import MLPRegressor
import pandas as pd

# **超参数**
input_dim = 40 * 5  # 输入维度（40 × 5）
output_dim = 5  # 输出 5 个 evaluation 值
hidden_dim = 128  # 隐藏层大小
num_epochs = 20
batch_size = 32
learning_rate = 0.001

# 定义时间序列长度
sequence_length_1 = 120
sequence_length_10 = 120
sequence_length_60 = 120
sequence_length_240 = 120
sequence_length_1380 = 120

# **数据集路径**
data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
test_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_test.pkl'

# 创建测试集的 Dataset 和 DataLoader
test_df = pd.read_pickle(test_path)
test_dataset = TimeSeriesLSTMTSDataset(test_df, sequence_length_1, sequence_length_10,
                                      sequence_length_60, sequence_length_240, sequence_length_1380)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

# **加载 MLP 模型**
mlp_model = MLPRegressor(input_dim, hidden_dim, output_dim)
mlp_model.load_state_dict(torch.load(data_base + "/models/mlp_regressor.pth"))
mlp_model.eval()

# **获取一个测试样本**
(close_1, close_10, close_60, close_240, close_1380,
 volume_1, volume_10, volume_60, volume_240, volume_1380, evaluation_values) = next(iter(test_dataloader))

# 定义模型参数
input_size = 2  # 每个时间步的特征数量（两个变量：Close_1 和 Volume_1）
hidden_size = 60
num_layers = 2
encoded_size = 40
encoder = LSTMEncoder(input_size, hidden_size, num_layers, encoded_size)

# **编码 10 个时间序列**
encoded_vectors = []
for seq in [close_1, close_10, close_60, close_240, close_1380,
            volume_1, volume_10, volume_60, volume_240, volume_1380]:
    seq_encoded = encoder(seq.unsqueeze(-1).float())  # 形状: (1, 40)
    encoded_vectors.append(seq_encoded)

# **拼接编码向量**
final_encoded_vector = torch.cat(encoded_vectors, dim=-1)  # 形状: (1, 200)

# **MLP 预测**
predicted_evaluation = mlp_model(final_encoded_vector)

# **打印真实值 vs 预测值**
print("真实 Evaluation:", evaluation_values.numpy())
print("预测 Evaluation:", predicted_evaluation.detach().numpy())