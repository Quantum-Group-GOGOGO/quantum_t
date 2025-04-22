from LSTM1 import LSTMAutoencoder
from LSTM1_loss import WeightedMSELoss
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader_LSTMTS import TimeSeriesLSTMTSDataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import random



# 将下面的代码放在 main 块中
if __name__ == "__main__":
    # 检查是否可用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 假设你已经有一个加载好的 DataFrame 'df'
    #data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
    data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
    # 读取训练集和测试集
    train_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_train.pkl'
    test_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_test.pkl'

    df = pd.read_pickle(train_path)

    # 定义时间序列长度
    sequence_length_1 = 120
    sequence_length_10 = 100
    sequence_length_60 = 60
    sequence_length_240 = 60
    sequence_length_1380 = 60

    # 创建 Dataset 和 DataLoader
    dataset = TimeSeriesLSTMTSDataset(df, sequence_length_1, sequence_length_10,
                                sequence_length_60, sequence_length_240, sequence_length_1380)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    # 创建测试集的 Dataset 和 DataLoader
    test_df = pd.read_pickle(test_path)
    test_dataset = TimeSeriesLSTMTSDataset(test_df, sequence_length_1, sequence_length_10,
                                        sequence_length_60, sequence_length_240, sequence_length_1380)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=0)
    # 查看一个样本
    #sample_close_1, sample_volume_1 = next(iter(dataloader))
    #print("Close_1 Sequence:", sample_close_1.shape)
    #print("Volume_1 Sequence:", sample_volume_1.shape)
    #first_close_sequence = sample_close_1[0]
    #print("First Close_1 Sequence:", first_close_sequence.shape)
    #print("First Close_1 Sequence Data:", first_close_sequence)


    #————————以上是dataloader的测试部分，上面的没问题了，输入张量就没问题了————————————


    # 超参数
    input_size = 2  # 每个时间步的特征数量（两个变量：Close_1 和 Volume_1）
    hidden_size = 120
    num_layers = 2
    encoded_size = 80
    num_heads = 0
    transformer_layers = 0
    num_epochs = 5
    learning_rate = 0.001

    # 实例化模型
    #model = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size)
    model = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size, num_heads, transformer_layers).to(device)


    # 损失函数和优化器(WeightedMSELoss是自定义损失函数类)
    criterion = WeightedMSELoss(close_weight=1.0, volume_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # 使用 tqdm 包装 dataloader，显示进度条
        with tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as tepoch:
            for batch in tepoch:
                close_1, close_10, close_60, close_240, close_1380, \
                volume_1, volume_10, volume_60, volume_240, volume_1380, evaluation = batch
                sample_close_1 = close_1.to(device)
                sample_volume_1 = volume_1.to(device)
                # 合并输入特征（close 和 volume）
                x_batch = torch.cat((sample_close_1.unsqueeze(-1), sample_volume_1.unsqueeze(-1)), dim=2).float()  # (batch_size, sequence_length, input_size)

                # 前向传播
                optimizer.zero_grad()
                output = model(x_batch)

                # 计算损失
                loss = criterion(output, x_batch)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 累加损失
                total_loss += loss.item()

                # 在进度条中显示当前批次的损失
                tepoch.set_postfix(loss=loss.item())

        # 计算整个数据集上的平均损失
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Train Loss: {avg_loss:.8f}')

        # 评估模型在测试集上的表现
        model.eval()  # 切换到评估模式
        with torch.no_grad():
            # 从测试集中获取一个样本
            sample_close_1, sample_close_10, sample_close_60, sample_close_240, sample_close_1380,sample_volume_1, sample_volume_10, sample_volume_60, sample_volume_240, sample_volume_1380, sample_evaluation_data_current = next(iter(test_dataloader))

            # 把它们都搬到 GPU
            sample_close_1 = sample_close_1.to(device)
            sample_volume_1 = sample_volume_1.to(device)
            # 合并输入特征（close 和 volume）
            x_test = torch.cat((sample_close_1.unsqueeze(-1), sample_volume_1.unsqueeze(-1)), dim=2).float()  # 添加 batch 维度

            # 前向传播
            reconstructed_output = model(x_test)

            # 计算测试损失
            test_loss = criterion(reconstructed_output, x_test)
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss (Random Sample): {test_loss:.8f}')

        # 保存模型
        model_path = data_base + '/models/lstm1_encoder/LSTMAutoencoder_trained_stdnorm_120to80_s3_2LSTM.pth'
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")