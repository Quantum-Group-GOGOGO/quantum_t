import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader_LSTMTS import TimeSeriesLSTMTSDataset
from LSTM1 import LSTMAutoencoder  # 只加载 Encoder
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
encoder1 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size)
encoder2 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size)
encoder3 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size)
encoder4 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size)
encoder5 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size)

model_path = data_base + '/models/lstm1_encoder/LSTMAutoencoder_trained2.pth'
state_dict = torch.load(model_path, map_location=device, weights_only=True)
encoder1.load_state_dict(state_dict, strict=True)
encoder2.load_state_dict(state_dict, strict=True)
encoder3.load_state_dict(state_dict, strict=True)
encoder4.load_state_dict(state_dict, strict=True)
encoder5.load_state_dict(state_dict, strict=True)

encoded1 = encoder1.encoder(torch.stack([close_1, volume_1], dim=-1))
encoded2 = encoder2.encoder(torch.stack([close_10, volume_10], dim=-1))
encoded3 = encoder3.encoder(torch.stack([close_60, volume_60], dim=-1))
encoded4 = encoder4.encoder(torch.stack([close_240, volume_240], dim=-1))
encoded5 = encoder5.encoder(torch.stack([close_1380, volume_1380], dim=-1))
# 拼接 5 组 LSTM 输出
mlp_input = torch.cat([encoded1, encoded2, encoded3, encoded4, encoded5], dim=1)

# **MLP 预测**
predicted_evaluation = mlp_model(mlp_input)

# **打印真实值 vs 预测值**
print("真实 Evaluation:", evaluation_values.numpy())
print("预测 Evaluation:", predicted_evaluation.detach().numpy())