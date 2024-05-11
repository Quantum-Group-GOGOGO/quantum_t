import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from numpy import genfromtxt

# 数据预处理
def create_inout_sequences(input_data, y_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = y_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

# 模型定义
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)  # 注意batch_first参数
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # 重新初始化隐藏状态
        self.hidden_cell = (torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(input_seq.device),
                            torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(input_seq.device))
        
        # LSTM层
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        
        # 只获取序列的最后一个时间点的输出用于预测
        lstm_out = lstm_out[:, -1, :]
        
        # 线性层
        predictions = self.linear(lstm_out)
        return predictions

data=genfromtxt('evaluation.csv', delimiter=',')
data=data[:,:]
# 计算90%数据的索引位置
split_idx = int(len(data) * 0.9)
# 分割数据
train_data = data[:split_idx]
test_data = data[split_idx:]
# 你的时间序列数据和Y值
train_t_data = train_data[:,1]
train_y_data = train_data[:,2]
test_t_data = test_data[:,1]
test_y_data = test_data[:,2]

# 数据标准化
## 标准化训练数据
feature_scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = feature_scaler.fit_transform(train_t_data.reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
## 标准化测试数据（使用训练数据的scaler）
test_data_normalized = feature_scaler.transform(test_t_data.reshape(-1, 1))
test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)
## 标准化训练数据
label_scaler = MinMaxScaler(feature_range=(-1, 1))
train_labels_normalized = label_scaler.fit_transform(train_y_data.reshape(-1, 1))
train_labels_normalized = torch.FloatTensor(train_labels_normalized).view(-1)
## 标准化测试数据（使用训练数据的scaler）
test_labels_normalized = label_scaler.transform(test_y_data.reshape(-1, 1))
test_labels_normalized = torch.FloatTensor(test_labels_normalized).view(-1)

# 创建序列和标签
train_window = 30
train_inout_seq = create_inout_sequences(train_data_normalized, train_labels_normalized, train_window)
test_inout_seq = create_inout_sequences(test_data_normalized, test_labels_normalized, train_window)
# 转换为张量
train_sequence_tensors = torch.stack([torch.FloatTensor(s[0]).unsqueeze(1) for s in train_inout_seq])
train_label_tensors = torch.stack([torch.FloatTensor(s[1]) for s in train_inout_seq])
test_sequence_tensors = torch.stack([torch.FloatTensor(s[0]).unsqueeze(1) for s in test_inout_seq])
test_label_tensors = torch.stack([torch.FloatTensor(s[1]) for s in test_inout_seq])

# 创建TensorDataset
train_dataset = TensorDataset(train_sequence_tensors, train_label_tensors)
test_dataset = TensorDataset(test_sequence_tensors, test_label_tensors)
# 选择一个合适的批处理大小
batch_size = 64
# 创建DataLoader
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# 模型实例化
model = LSTM()
loss_function = nn.MSELoss() # 根据Y的类型选择合适的损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
epochs = 150
for epoch in range(epochs):
    for seq, labels in train_data_loader:
        
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step() 

    if epoch % 10 == 0:
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 在不计算梯度的情况下执行前向传播
            test_losses = []
            for test_seq, test_labels in test_data_loader:
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                     torch.zeros(1, 1, model.hidden_layer_size))
                
                test_preds = model(test_seq)
                test_loss = loss_function(test_preds, test_labels)
                test_losses.append(test_loss.item())

            avg_test_loss = np.mean(test_losses)  # 计算所有测试损失的平均值
            print(f'epoch: {epoch:3} Train loss: {single_loss.item():10.8f} Test loss: {avg_test_loss:10.8f}')