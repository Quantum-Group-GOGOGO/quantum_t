import torch
import torch.nn as nn

# LSTM Autoencoder 的 Encoder 部分，新增 Transformer 支持
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, encoded_size,
                 num_heads=1, transformer_layers=0):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 如果需要 transformer，初始化 TransformerEncoder
        if transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
            self.transformer = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=transformer_layers)
        else:
            self.transformer = None
        self.fc = nn.Linear(hidden_size, encoded_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, _) = self.lstm(x)
        # LSTM 输出 out: (batch, seq_len, hidden_size)
        if self.transformer:
            # Transformer 要求 (seq_len, batch, hidden)
            trans_in = out.permute(1, 0, 2)
            trans_out = self.transformer(trans_in)
            # 恢复 (batch, seq_len, hidden)
            out = trans_out.permute(1, 0, 2)
        # 用最后时刻隐状态映射到 encoded_size
        encoded = self.fc(out[:, -1, :])
        return encoded

# LSTM Autoencoder 的 Decoder 部分
class LSTMDecoder(nn.Module):
    def __init__(self, encoded_size, hidden_size, num_layers, input_size):
        super(LSTMDecoder, self).__init__()
        self.fc = nn.Linear(encoded_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, encoded, sequence_length):
        # encoded: (batch, encoded_size)
        decoded_input = self.fc(encoded).unsqueeze(1)
        decoded_input = decoded_input.repeat(1, sequence_length, 1)
        decoded_output, _ = self.lstm(decoded_input)
        output = self.output_layer(decoded_output)
        return output

# 编码器-解码器模型
class LSTMAutoencoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 encoded_size,
                 num_heads=1,
                 transformer_layers=0):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_size,
                                   hidden_size,
                                   num_layers,
                                   encoded_size,
                                   num_heads,
                                   transformer_layers)
        self.decoder = LSTMDecoder(encoded_size,
                                   hidden_size,
                                   num_layers,
                                   input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, x.size(1))
        return decoded
