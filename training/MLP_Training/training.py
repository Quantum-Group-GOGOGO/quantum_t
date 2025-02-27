import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader_LSTMTS import TimeSeriesLSTMTSDataset
from LSTM1 import LSTMEncoder  # 只加载 Encoder
from MLP import MLPRegressor
import pandas as pd



# **数据集路径**
data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
#data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
train_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_train.pkl'

# **统一时间序列长度**
sequence_length = 120

# **加载训练数据**
train_df = pd.read_pickle(train_path)
train_dataset = TimeSeriesLSTMTSDataset(train_df, sequence_length, sequence_length, sequence_length, sequence_length, sequence_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# **加载共享的 LSTMEncoder**
input_size = 2
hidden_size = 60
num_layers = 2
encoded_size = 40

# **创建共享的 LSTMEncoder**
encoder1 = LSTMEncoder(input_size, hidden_size, num_layers, encoded_size)
encoder2 = LSTMEncoder(input_size, hidden_size, num_layers, encoded_size)
encoder3 = LSTMEncoder(input_size, hidden_size, num_layers, encoded_size)
encoder4 = LSTMEncoder(input_size, hidden_size, num_layers, encoded_size)
encoder5 = LSTMEncoder(input_size, hidden_size, num_layers, encoded_size)

# **加载训练好的模型权重**
model_path = data_base + '/models/lstm1_encoder/LSTMAutoencoder_trained2.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(model_path, map_location=device)
encoder1.load_state_dict(state_dict, strict=False)
encoder2.load_state_dict(state_dict, strict=False)
encoder3.load_state_dict(state_dict, strict=False)
encoder4.load_state_dict(state_dict, strict=False)
encoder5.load_state_dict(state_dict, strict=False)
encoder1.eval()  # 设为评估模式
encoder2.eval()  # 设为评估模式
encoder3.eval()  # 设为评估模式
encoder4.eval()  # 设为评估模式
encoder5.eval()  # 设为评估模式

# **超参数**
input_dim = 40 * 5  # 输入维度（40 × 5）
output_dim = 5  # 输出 5 个 evaluation 值
hidden_dim = 128  # 隐藏层大小
num_epochs = 20
batch_size = 32
learning_rate = 0.001

# **实例化 MLP**
mlp_model = MLPRegressor(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)
