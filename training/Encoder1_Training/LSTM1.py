import torch
import torch.nn as nn
import torch.optim as optim

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, encoded_size, num_heads=4, transformer_layers=1, volume_scaling=0.5):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        
        # Transformer 处理 LSTM 输出
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # 线性层用于降维
        self.fc = nn.Linear(hidden_size, encoded_size)
        self.volume_scaling = volume_scaling  # 控制 volume 在 Transformer 计算后的影响

    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)  # (batch_size, sequence_length, hidden_size)

        # 只让 close 参与 Transformer 计算，而 volume 直接缩放
        transformer_out = self.transformer(lstm_out)  # (batch_size, sequence_length, hidden_size)

        # 对 volume 进行缩放，使得 close 占主导地位
        transformer_out[:, :, 1] = transformer_out[:, :, 1] * self.volume_scaling

        # 取 Transformer 处理后的最后一个时间步
        encoded = self.fc(transformer_out[:, -1, :])  # (batch_size, encoded_size)

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
    def __init__(self, input_size, hidden_size, num_layers, encoded_size, num_heads=4, transformer_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, encoded_size, num_heads, transformer_layers)
        self.decoder = LSTMDecoder(encoded_size, hidden_size, num_layers, input_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, x.size(1))
        return decoded
