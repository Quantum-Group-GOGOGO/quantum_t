import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader_LSTMTS import TimeSeriesLSTMTSDataset
from LSTM1 import LSTMAutoencoder  # 只加载 Encoder
from MLP import MLPRegressor
import pandas as pd
from tqdm import tqdm


# **数据集路径**
data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
#data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
train_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_train.pkl'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **统一时间序列长度**
sequence_length = 120
batch_size = 32
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
encoder1 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size).to(device)
encoder2 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size).to(device)
encoder3 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size).to(device)
encoder4 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size).to(device)
encoder5 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size).to(device)

# **加载训练好的模型权重**
model_path = data_base + '/models/lstm1_encoder/LSTMAutoencoder_trained2.pth'
state_dict = torch.load(model_path, map_location=device, weights_only=True)
encoder1.load_state_dict(state_dict, strict=True)
encoder2.load_state_dict(state_dict, strict=True)
encoder3.load_state_dict(state_dict, strict=True)
encoder4.load_state_dict(state_dict, strict=True)
encoder5.load_state_dict(state_dict, strict=True)
encoder1.eval()  # 设为评估模式
encoder2.eval()  # 设为评估模式
encoder3.eval()  # 设为评估模式
encoder4.eval()  # 设为评估模式
encoder5.eval()  # 设为评估模式
for param in encoder1.parameters():
    param.requires_grad = False
for param in encoder2.parameters():
    param.requires_grad = False
for param in encoder3.parameters():
    param.requires_grad = False
for param in encoder4.parameters():
    param.requires_grad = False
for param in encoder5.parameters():
    param.requires_grad = False
    
# **超参数**
input_dim = 40 * 5  # 输入维度（40 × 5）
output_dim = 5  # 输出 5 个 evaluation 值
hidden_dim1 = 128  # 隐藏层大小
hidden_dim2 = 64  # 隐藏层大小
hidden_dim3 = 20  # 隐藏层大小
num_epochs = 20
batch_size = 32
learning_rate = 0.001

# **实例化 MLP**
mlp_model = MLPRegressor(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)

# 新增全局 batch 计数器和累计损失变量
global_batch_count = 0
accumulated_loss = 0.0

# 在训练开始前清空 training_loss.txt 文件
loss_file_path = data_base + '/models/training_loss.txt'
with open(loss_file_path, 'w') as f:
    f.write('')

# **训练循环**
for epoch in range(num_epochs):
    mlp_model.train()
    total_loss = 0
    
    with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as tbar:
        for batch in tbar:
            close_1, close_10, close_60, close_240, close_1380, \
            volume_1, volume_10, volume_60, volume_240, volume_1380, evaluation = batch

            # 发送到 GPU 或 CPU
            close_1, close_10, close_60, close_240, close_1380 = close_1.to(device), close_10.to(device), close_60.to(device), close_240.to(device), close_1380.to(device)
            volume_1, volume_10, volume_60, volume_240, volume_1380 = volume_1.to(device), volume_10.to(device), volume_60.to(device), volume_240.to(device), volume_1380.to(device)
            
            evaluation = evaluation.to(device)

            # 计算 LSTM 编码器的输出
            encoded1 = encoder1.encoder(torch.stack([close_1, volume_1], dim=-1))
            encoded2 = encoder2.encoder(torch.stack([close_10, volume_10], dim=-1))
            encoded3 = encoder3.encoder(torch.stack([close_60, volume_60], dim=-1))
            encoded4 = encoder4.encoder(torch.stack([close_240, volume_240], dim=-1))
            encoded5 = encoder5.encoder(torch.stack([close_1380, volume_1380], dim=-1))

            # 拼接 5 组 LSTM 输出
            mlp_input = torch.cat([encoded1, encoded2, encoded3, encoded4, encoded5], dim=1).to(device)

            # 计算 MLP 输出
            pred = mlp_model(mlp_input)

            # 计算损失
            loss = criterion(pred, evaluation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            tbar.set_postfix(loss=loss.item())

            # 累计用于每 1000 个 batch 计算平均 loss
            global_batch_count += 1
            accumulated_loss += loss.item()

            # 每 1000 个 batch 保存一次平均训练 loss 到文件中
            if global_batch_count % 1000 == 0:
                avg_loss = accumulated_loss / 1000
                # 以追加模式写入文件，每次写入一行
                with open(data_base + '/models/training_loss.txt', 'a') as f:
                    f.write(f"{avg_loss}\n")
                accumulated_loss = 0.0  # 重置累计 loss
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.6f}')

# **保存 MLP 模型**
mlp_model_path = data_base + '/models/mlp_regressor.pth'
torch.save(mlp_model.state_dict(), mlp_model_path)
print("训练完成，MLP 模型已保存！")

