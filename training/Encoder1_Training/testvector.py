import torch
import pandas as pd
import matplotlib.pyplot as plt
from LSTM1 import LSTMEncoder  # 只加载 Encoder
from dataloader_LSTMTS import TimeSeriesLSTMTSDataset
from torch.utils.data import DataLoader

# 读取测试集数据
#data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
test_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_test.pkl'

# **统一时间序列长度**
sequence_length = 120

# 创建测试集 Dataset 和 DataLoader
test_df = pd.read_pickle(test_path)
test_dataset = TimeSeriesLSTMTSDataset(test_df, sequence_length, sequence_length, sequence_length, sequence_length, sequence_length)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

# **模型参数**
input_size = 2  # 每个时间步只有一个变量 (单独处理 close 和 volume)
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

# **获取一个测试样本**
with torch.no_grad():
    # **取一个随机样本**
    (close_1, close_10, close_60, close_240, close_1380,
     volume_1, volume_10, volume_60, volume_240, volume_1380, evaluation_data_current) = next(iter(test_dataloader))

    # **逐个编码 10 个序列**
    encoded_vectors = []

    # 合并输入特征（close 和 volume）
    x_sample1 = torch.cat((close_1.unsqueeze(-1), volume_1.unsqueeze(-1)), dim=2).float()  # (1, sequence_length, input_size)
    x_sample2 = torch.cat((close_10.unsqueeze(-1), volume_10.unsqueeze(-1)), dim=2).float()  # (1, sequence_length, input_size)
    x_sample3 = torch.cat((close_60.unsqueeze(-1), volume_60.unsqueeze(-1)), dim=2).float()  # (1, sequence_length, input_size)
    x_sample4 = torch.cat((close_240.unsqueeze(-1), volume_240.unsqueeze(-1)), dim=2).float()  # (1, sequence_length, input_size)
    x_sample5 = torch.cat((close_1380.unsqueeze(-1), volume_1380.unsqueeze(-1)), dim=2).float()  # (1, sequence_length, input_size)

    vector1 = encoder1(x_sample1)
    vector2 = encoder2(x_sample2)
    vector3 = encoder3(x_sample3)
    vector4 = encoder4(x_sample4)
    vector5 = encoder5(x_sample5)

    encoded_vectors.append(vector1)
    encoded_vectors.append(vector2)
    encoded_vectors.append(vector3)
    encoded_vectors.append(vector4)
    encoded_vectors.append(vector5)
    # **最终拼接为 (1, 400) 维向量**
    final_encoded_vector = torch.cat(encoded_vectors, dim=-1)  # 形状: (1, 400)

