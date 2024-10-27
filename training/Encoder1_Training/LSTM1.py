import torch
import torch.nn as nn
import torch.optim as optim

# LSTM Autoencoder 的 Encoder 部分
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, encoded_size):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, encoded_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        encoded = self.fc(h_n[-1])  # 使用最后一个层的隐藏状态进行全连接
        return encoded

# LSTM Autoencoder 的 Decoder 部分
class LSTMDecoder(nn.Module):
    def __init__(self, encoded_size, hidden_size, num_layers, input_size):
        super(LSTMDecoder, self).__init__()
        self.fc = nn.Linear(encoded_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, encoded, sequence_length):
        # 将编码向量转换为 LSTM 的初始输入
        decoded_input = self.fc(encoded).unsqueeze(1).repeat(1, sequence_length, 1)
        decoded_output, _ = self.lstm(decoded_input)
        output = self.output_layer(decoded_output)
        return output

# 编码器-解码器模型
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, encoded_size):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, encoded_size)
        self.decoder = LSTMDecoder(encoded_size, hidden_size, num_layers, input_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, x.size(1))  # 使用输入的序列长度进行解码
        return decoded

