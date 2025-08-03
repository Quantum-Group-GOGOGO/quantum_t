#!/usr/bin/env python3
# inference.py
from env import *  
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from auxprojector import AuxAutoencoder
from dataloader_LSTMTS import TimeSeriesLSTMTSDataset
from LSTM1 import LSTMAutoencoder
from MLP import MLPRegressor

def load_encoders(lstm_path, device):
    """
    加载 5 个共享权重的 LSTM 编码器，并冻结参数。
    """
    input_size = 2
    hidden_size = 120
    num_layers = 2
    encoded_size = 80
    num_heads = 0
    transformer_layers = 0
    state_dict = torch.load(lstm_path, map_location=device,weights_only=True)
    encoders = []
    for _ in range(5):
        enc = LSTMAutoencoder(input_size, hidden_size, num_layers, encoded_size, num_heads, transformer_layers).to(device)
        enc.load_state_dict(state_dict, strict=True)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        encoders.append(enc)
    return encoders

def main():
    parser = argparse.ArgumentParser(description="LSTM+MLP 回归模型推理脚本")
    parser.add_argument('--sequence_length', type=int, default=120,
                        help="训练时使用的序列长度（默认为 120)")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="推理时的 batch size (默认为 1, 逐行预测)")
    args = parser.parse_args()
    
    #data_base = 'D:\quantum\quantum_t_data\quantum_t_data'
    lstm_model= data_base + '/models/lstm1_encoder/LSTMAutoencoder_trained_stdnorm_120to80_s3_2LSTM.pth'
    mlp_model_dic = data_base + '/models/mlp_regressor_80to5_1200+400+80_v3.pth'
    aux_encoder_path = data_base + '/models/aux_projector_encoder.pth'

    # 1. 读原始 DataFrame
    #data_path = data_base + "/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1.pkl"
    data_path = live_data_base +'/type6/type6Base.pkl'
    df = pd.read_pickle(data_path)
    df.index.name = 'datetime'
    df = df.reset_index()
    
    print(df.head())
    fromwhere=int(len(df) * 0.8)

    # 2. 新增 3 个预测列，默认全部填 0
    for j in range(1, 4):
        df[f'prediction{j}'] = np.nan

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
    mlp_model = MLPRegressor(80*5+100, 1200, 400, 80, 3).to(device)
    mlp_model.load_state_dict(torch.load(mlp_model_dic, map_location=device,weights_only=True))
    mlp_model.eval()

    proj_dim = 100  # 投影后维度

    aux_autoencoder = AuxAutoencoder(in_dim=9, hidden_dim=64, proj_dim=proj_dim).to(device)
    aux_state = torch.load(aux_encoder_path, map_location=device, weights_only=True)
    aux_autoencoder.load_state_dict(aux_state, strict=True)
    aux_autoencoder.eval()
    for p in aux_autoencoder.parameters():
        p.requires_grad = False
    aux_encoder= aux_autoencoder.encoder

    # 预先计算好 df 要写入的全局偏移列表
    df_indices = [i + offset for i in ds_indices]

    # 9. 推理
    with torch.no_grad():
        for batch_idx, batch in enumerate(
                tqdm(loader, desc="Inference", total=len(loader))):
            # 解包并搬到 device
            close_1, close_10, close_60, close_240, close_1380, \
            vol_1,   vol_10,   vol_60,   vol_240,   vol_1380, _, auxiliary, high_1, low_1, open_1 = batch
            closes = [close_1.to(device), close_10.to(device),
                      close_60.to(device), close_240.to(device),
                      close_1380.to(device)]
            vols   = [vol_1.to(device),   vol_10.to(device),
                      vol_60.to(device),  vol_240.to(device),
                      vol_1380.to(device)]
            auxiliary = auxiliary.to(device)

            # 批量编码：得到 3 个 [B, 80] 的张量
            encoded_list = []
            for enc, c, v in zip(encoders, closes, vols):
                inp = torch.stack([c, v], dim=-1)       # [B, seq_len, 2]
                encoded = enc.encoder(inp)             # [B, 80]
                encoded_list.append(encoded)

            # 拼接 & MLP 推理
            mlp_in = torch.cat(encoded_list, dim=1)     # [B, 400]

            # Aux 投影器
            aux_feat = aux_encoder(auxiliary)      # [B, 100]
            mlp_in = torch.cat([mlp_in, aux_feat], dim=1)  # [B, 500]

            preds  = mlp_model(mlp_in)                  # [B,   3]
            preds_np = preds.cpu().numpy()              # numpy [B,3]

            # 批量写回 DataFrame
            start = batch_idx * args.batch_size
            end   = start + preds_np.shape[0]
            for j in range(3):
                # 一次性给 prediction{j+1} 这一整段赋值
                df.loc[df_indices[start:end],f'prediction{j+1}'] = preds_np[:, j]

    # 保存结果
    out_path = data_base + "/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_with_predictions_120to80_2LSTM_future3.pkl"
    df.to_pickle(out_path)
    print(f"推理完成，结果保存在：{out_path}")

if __name__ == "__main__":
    main()