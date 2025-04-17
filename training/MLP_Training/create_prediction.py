#!/usr/bin/env python3
# inference.py

import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset

from dataloader_LSTMTS import TimeSeriesLSTMTSDataset
from LSTM1 import LSTMAutoencoder
from MLP import MLPRegressor

def load_encoders(lstm_path, device):
    """
    加载 5 个共享权重的 LSTM 编码器，并冻结参数。
    """
    state_dict = torch.load(lstm_path, map_location=device)
    encoders = []
    for _ in range(5):
        enc = LSTMAutoencoder(input_size=2,
                              hidden_size=60,
                              num_layers=2,
                              encoded_size=40).to(device)
        enc.load_state_dict(state_dict, strict=True)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        encoders.append(enc)
    return encoders

def main():
    parser = argparse.ArgumentParser(description="LSTM+MLP 回归模型推理脚本")
    parser.add_argument('--sequence_length', type=int, default=120,
                        help="训练时使用的序列长度（默认为 120）")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="推理时的 batch size（默认为 1，逐行预测）")
    args = parser.parse_args()
    
    data_base = 'D:\quantum\quantum_t_data\quantum_t_data'
    lstm_model= data_base + '/models/lstm1_encoder/LSTMAutoencoder_trained2.pth'
    mlp_model_dic = data_base + '/models/mlp_regressor.pth'
    
    # 1. 读原始 DataFrame
    data_path = data_base + "/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1.pkl"
    df = pd.read_pickle(data_path)

    fromwhere=int(len(df) * 0.9)

    # 2. 新增 5 个预测列，默认全部填 0
    for j in range(1, 6):
        df[f'prediction{j}'] = 0.0

    # 3. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. 构建完整的 Dataset（用于滑窗）
    dataset = TimeSeriesLSTMTSDataset(
        df,
        args.sequence_length,  # sequence_length_1
        args.sequence_length,  # sequence_length_10
        args.sequence_length,  # sequence_length_60
        args.sequence_length,  # sequence_length_240
        args.sequence_length   # sequence_length_1380
    )

    # 5. 计算 dataset 索引与原 DataFrame 索引的映射偏移
    #    dataset.__getitem__ 中： end_idx = idx + sequence_length_1380 * 1380
    offset = args.sequence_length * 1380

    # 6. 只对行号 ≥ max(N, offset) 的数据做预测
    start_idx = max(fromwhere, offset)
    ds_start   = start_idx - offset
    ds_indices = list(range(ds_start, len(dataset)))

    # 7. 用 Subset + DataLoader 逐行（batch_size=1）加载
    subset = Subset(dataset, ds_indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

    # 8. 加载模型
    encoders  = load_encoders(lstm_model, device)
    mlp_model = MLPRegressor(
        input_dim=40*5,
        hidden_dim1=400,
        hidden_dim2=128,
        hidden_dim3=40,
        output_dim=5
    ).to(device)
    mlp_model.load_state_dict(torch.load(mlp_model_dic, map_location=device))
    mlp_model.eval()

    # 9. 推理
    for batch_idx, batch in enumerate(loader):
        # batch 会是一个 tuple: (close_1,…,close_1380, volume_1,…,volume_1380, evaluation)
        close_1, close_10, close_60, close_240, close_1380, \
        volume_1, volume_10, volume_60, volume_240, volume_1380, _ = batch

        # 逐样本（当 batch_size>1 时也能工作）
        B = close_1.shape[0]
        for b in range(B):
            encs = []
            for ei, enc in enumerate(encoders):
                # 取出对应尺度下的 close & volume
                close_seq = [close_1, close_10, close_60, close_240, close_1380][ei][b].to(device)
                vol_seq   = [volume_1, volume_10, volume_60, volume_240, volume_1380][ei][b].to(device)
                # 拼成 [1, seq_len, 2]
                inp = torch.stack([close_seq, vol_seq], dim=-1).unsqueeze(0)
                encs.append(enc.encoder(inp))
            # 拼接五个编码，再过 MLP
            mlp_in = torch.cat(encs, dim=1)  # [1, 200]
            pred   = mlp_model(mlp_in)       # [1,5]
            pred   = pred.detach().cpu().numpy().reshape(-1)

            # 计算回原 DataFrame 的行号，并写入
            df_idx = ds_indices[batch_idx*args.batch_size + b] + offset
            for j in range(5):
                df.at[df_idx, f'prediction{j+1}'] = float(pred[j])

    # 10. 保存带预测的新 DataFrame
    out_path = data_base + "/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_with_predictions.pkl"
    df.to_pickle(out_path)
    print(f"推理完成，结果保存在：{out_path}")

if __name__ == "__main__":
    main()