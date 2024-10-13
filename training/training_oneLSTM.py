from dataloader import TimeSeriesDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# 定义 LSTM 模型
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出并通过全连接层
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    # 假设你已经有一个加载好的 DataFrame 'df'
    data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
    T6_data_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1.pkl'

    df = pd.read_pickle(T6_data_path)

    # 划分训练集和验证集
    train_size = int(len(df) * 19 / 20)
    df_train = df[:train_size]
    df_val = df[train_size:]

    # 定义时间序列长度
    sequence_length_1 = 120
    sequence_length_10 = 100
    sequence_length_60 = 60
    sequence_length_240 = 60
    sequence_length_1380 = 60

    # 创建训练集 Dataset 和 DataLoader
    train_dataset = TimeSeriesDataset(df_train, sequence_length_1, sequence_length_10,
                                      sequence_length_60, sequence_length_240, sequence_length_1380)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    # 定义模型参数
    input_size = 10 + 9  # 10 个时间序列 + 9 个协变量
    hidden_size = 50
    num_layers = 2
    output_size = 5  # 5 个需要预测的数值

    # 初始化模型、损失函数和优化器
    model = TimeSeriesLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练模型
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            # 获取输入和目标数据
            close_1, close_10, close_60, close_240, close_1380, volume_1, volume_10, volume_60, volume_240, volume_1380, other_data, evaluation_data = batch
            
            # 将数据移动到 GPU（如果可用）
            close_1 = close_1.to(device)
            close_10 = close_10.to(device)
            close_60 = close_60.to(device)
            close_240 = close_240.to(device)
            close_1380 = close_1380.to(device)
            volume_1 = volume_1.to(device)
            volume_10 = volume_10.to(device)
            volume_60 = volume_60.to(device)
            volume_240 = volume_240.to(device)
            volume_1380 = volume_1380.to(device)
            other_data = other_data.to(device)
            evaluation_data = evaluation_data.to(device)
            
            # 将时间序列和协变量拼接为输入 (batch_size, seq_length, input_size)
            time_series_inputs = torch.cat((close_1, close_10, close_60, close_240, close_1380,
                                            volume_1, volume_10, volume_60, volume_240, volume_1380), dim=2)
            inputs = torch.cat((time_series_inputs, other_data), dim=2)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, evaluation_data)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("训练完成！")