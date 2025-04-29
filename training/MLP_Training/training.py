import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader_LSTMTS import TimeSeriesLSTMTSDataset
from LSTM1 import LSTMAutoencoder  # 只加载 Encoder
from MLP import MLPRegressor
import pandas as pd
from tqdm import tqdm
import os

# **数据集路径**
data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
#data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
train_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_train.pkl'
validation_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_validation.pkl'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrain_path = data_base + '/models/mlp_regressor_80to5_400+128+40.pth'
loss_file_path = data_base + '/models/training_loss.txt'
mlp_model_path = data_base + '/models/mlp_regressor_80to5_1200+400+80_v1.pth'
lstm_model_path = data_base + '/models/lstm1_encoder/LSTMAutoencoder_trained_stdnorm_120to80_s3_2LSTM.pth'
read_from_pretrained=False
# **统一时间序列长度**
sequence_length = 120
batch_size = 32
# **加载训练数据**
train_df = pd.read_pickle(train_path)
train_dataset = TimeSeriesLSTMTSDataset(train_df, sequence_length, sequence_length, sequence_length, sequence_length, sequence_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_df = pd.read_pickle(validation_path)
val_dataset = TimeSeriesLSTMTSDataset(val_df,sequence_length, sequence_length, sequence_length,sequence_length, sequence_length)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# **加载共享的 LSTMEncoder**
input_size = 2
hidden_size = 120
num_layers = 2
encoded_size = 80
num_heads = 0
transformer_layers = 0

# **创建共享的 LSTMEncoder**
encoder1 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size, num_heads, transformer_layers).to(device)
encoder2 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size, num_heads, transformer_layers).to(device)
encoder3 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size, num_heads, transformer_layers).to(device)
encoder4 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size, num_heads, transformer_layers).to(device)
encoder5 = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size, num_heads, transformer_layers).to(device)

# **加载训练好的模型权重**

state_dict = torch.load(lstm_model_path, map_location=device, weights_only=True)
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
encoders = [encoder1, encoder2, encoder3, encoder4, encoder5]

# **超参数**
input_dim = 80 * 5  # 输入维度（40 × 5）
output_dim = 5  # 输出 5 个 evaluation 值
hidden_dim1 = 1200  # 隐藏层大小
hidden_dim2 = 400  # 隐藏层大小
hidden_dim3 = 80  # 隐藏层大小
num_epochs = 1
batch_size = 64
learning_rate = 0.0001
adam_learning_rate = 0.0001
l2_lambda=1e-2


# 将学习率设为一个可训练的变量
lr_meta = torch.tensor(learning_rate, requires_grad=True, device=device)
# 定义一个用于更新 lr_meta 的 Adam 优化器（外层优化器），超参数更新学习率设为 1e-3
meta_optimizer = optim.Adam([lr_meta], lr=1e-3)

# 实例化 MLP
mlp_model = MLPRegressor(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device)

# 预训练模型文件路径

if read_from_pretrained:
    state = torch.load(pretrain_path, map_location=device)
    mlp_model.load_state_dict(state, strict=True)
    print(f'Loaded pretrained MLP weights from {pretrain_path}')
else:
    print(f'No pretrained file found at {pretrain_path}, training from scratch.')

# 内层优化器：用于更新 mlp_model 的参数，初始学习率由 lr_meta 的值给出
inner_optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate, weight_decay=l2_lambda)


criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(mlp_model.parameters(), lr=adam_learning_rate)

# 新增全局 batch 计数器和累计损失变量
global_batch_count = 0
accumulated_loss = 0.0

def validation_loss(encoders, mlp_model, val_loader, device):
    mlp_model.eval()
    for enc in encoders:
        enc.eval()
    total_val_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            close_1, close_10, close_60, close_240, close_1380, \
            volume_1, volume_10, volume_60, volume_240, volume_1380, evaluation, auxiliary = batch

            # to device
            close_1 = close_1.to(device);   volume_1 = volume_1.to(device)
            close_10 = close_10.to(device); volume_10 = volume_10.to(device)
            close_60 = close_60.to(device); volume_60 = volume_60.to(device)
            close_240 = close_240.to(device); volume_240 = volume_240.to(device)
            close_1380 = close_1380.to(device); volume_1380 = volume_1380.to(device)
            evaluation = evaluation.to(device)

            # 编码
            encoded = []
            for enc, c, v in zip(encoders,
                                 [close_1, close_10, close_60, close_240, close_1380],
                                 [volume_1, volume_10, volume_60, volume_240, volume_1380]):
                x = torch.stack([c, v], dim=-1)
                encoded.append(enc.encoder(x))
            mlp_input = torch.cat(encoded, dim=1)

            # 预测 & loss
            pred = mlp_model(mlp_input)
            loss = criterion(pred, evaluation)
            total_val_loss += loss.item()
            n_batches += 1

    # 切回 train 状态，保证下个 epoch 训练正常
    mlp_model.train()
    for enc in encoders:
        enc.train()  # 如果你不希望 fine-tune encoder，也可以继续保持 eval

    return total_val_loss / n_batches


# 在训练开始前清空 training_loss.txt 文件

with open(loss_file_path, 'w') as f:
    f.write('')

# **训练循环**
for epoch in range(num_epochs):
    mlp_model.train()
    total_loss = 0
    
    with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as tbar:
        for batch in tbar:
            close_1, close_10, close_60, close_240, close_1380, \
            volume_1, volume_10, volume_60, volume_240, volume_1380, evaluation, auxiliary = batch

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

            # 每 5000 个 batch 保存一次平均训练 loss 到文件中
            if global_batch_count % 5000 == 0:
                avg_loss = accumulated_loss / 5000
                # 下面进行超参数更新（更新 lr_meta）
                # 注意：实际实现中要确保整个过程的计算图是连贯的，本示例仅为结构演示
                meta_optimizer.zero_grad()
                # 假设 avg_loss 可以直接反向传播得到关于 lr_meta 的梯度（真实情况需要展开内层更新步骤）
                # 此处为了演示，我们将 avg_loss 转为一个标量张量并调用 backward()
                hyper_loss = torch.tensor(avg_loss, requires_grad=True, device=device)
                hyper_loss.backward()
                meta_optimizer.step()
                
                # 将内层优化器的学习率更新为 lr_meta 新的数值
                new_lr = lr_meta.item()
                for param_group in inner_optimizer.param_groups:
                    param_group['lr'] = new_lr
                # 以追加模式写入文件，每次写入一行
                with open(data_base + '/models/training_loss.txt', 'a') as f:
                    f.write(f"{avg_loss}\n")
                accumulated_loss = 0.0  # 重置累计 loss
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.6f}')

    # **保存 MLP 模型**
    
    torch.save(mlp_model.state_dict(), mlp_model_path)
print("训练完成，MLP 模型已保存！")
val_loss = validation_loss(encoders, mlp_model, val_loader, device)
print(f'val_Loss: {val_loss:.6f}')
