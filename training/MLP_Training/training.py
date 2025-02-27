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
input_size = 1
hidden_size = 60
num_layers = 2
encoded_size = 40

encoder = LSTMEncoder(input_size, hidden_size, num_layers, encoded_size)
model_path = data_base + '/models/lstm1_encoder/LSTMAutoencoder_trained2.pth'
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
encoder.load_state_dict(state_dict, strict=False)
encoder.eval()  # 设为评估模式

# **实例化 MLP**
mlp_model = MLPRegressor(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)

# **训练 MLP**
for epoch in range(num_epochs):
    total_loss = 0
    for (close_1, close_10, close_60, close_240, close_1380,
         volume_1, volume_10, volume_60, volume_240, volume_1380, evaluation_values) in train_dataloader:

        # **逐个编码 10 个时间序列**
        encoded_vectors = []
        for seq in [close_1, close_10, close_60, close_240, close_1380,
                    volume_1, volume_10, volume_60, volume_240, volume_1380]:
            seq_encoded = encoder(seq.unsqueeze(-1).float())  # 形状: (batch_size, 40)
            encoded_vectors.append(seq_encoded)

        # **最终拼接为 (batch_size, 400)**
        final_encoded_vector = torch.cat(encoded_vectors, dim=-1)  # 形状: (batch_size, 400)

        # **前向传播**
        predicted_eval = mlp_model(final_encoded_vector)  # 输出形状: (batch_size, 5)

        # **计算损失**
        loss = criterion(predicted_eval, evaluation_values)  # 计算 MSELoss
        total_loss += loss.item()

        # **反向传播**
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

# **保存训练好的 MLP 模型**
torch.save(mlp_model.state_dict(), data_base + "/models/mlp_regressor.pth")
print("MLP 训练完成，模型已保存！")