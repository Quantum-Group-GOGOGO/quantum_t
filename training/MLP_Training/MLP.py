import torch
import torch.nn as nn
import torch.optim as optim
# **定义全连接 MLP 模型**
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # 第一层全连接
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # 第二层全连接
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)  # 第二层全连接
        self.fc4 = nn.Linear(hidden_dim3, output_dim)  # 输出层
        #self.sigmoid = nn.Sigmoid()  # 添加 Sigmoid 层
        #LGQ加限制0到1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        #x = self.sigmoid(x)
        return x