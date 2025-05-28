import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from dataloader_LSTMTS import TimeSeriesLSTMTSDataset
from auxprojector import AuxAutoencoder

# ----------------------
# 配置
# ----------------------
# 数据路径
DATA_BASE = 'D:/quantum/quantum_t_data/quantum_t_data'
TRAIN_PATH = os.path.join(DATA_BASE, 'type6', 'Nasdaq_qqq_align_labeled_base_evaluated_normST1_train.pkl')
VAL_PATH   = os.path.join(DATA_BASE, 'type6', 'Nasdaq_qqq_align_labeled_base_evaluated_normST1_validation.pkl')

# 序列长度（与原来保持一致）
SEQUENCE_LENGTH = 120
# 预训练输出维度
PROJ_DIM = 100
# 投影网络隐藏层维度
HIDDEN_DIM = 64
# 训练超参数
BATCH_SIZE = 64
LR = 1e-4
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------
# 准备数据
# ----------------------
print("Loading data...")
train_df = pd.read_pickle(TRAIN_PATH)
val_df   = pd.read_pickle(VAL_PATH)

# 使用 TimeSeriesLSTMTSDataset，只提取 auxiliary
train_ds = TimeSeriesLSTMTSDataset(train_df,
                                   SEQUENCE_LENGTH, SEQUENCE_LENGTH,
                                   SEQUENCE_LENGTH, SEQUENCE_LENGTH,
                                   SEQUENCE_LENGTH)
val_ds   = TimeSeriesLSTMTSDataset(val_df,
                                   SEQUENCE_LENGTH, SEQUENCE_LENGTH,
                                   SEQUENCE_LENGTH, SEQUENCE_LENGTH,
                                   SEQUENCE_LENGTH)

def aux_collate(batch):
    # batch: tuple of (close_1,..., evaluation, auxiliary)
    # 把 numpy.ndarray 转成 Tensor
    aux_tensors = [torch.from_numpy(item[-1]).float() for item in batch]
    return torch.stack(aux_tensors, dim=0)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, collate_fn=aux_collate)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=aux_collate)

# ----------------------
# 构建模型、损失、优化器
# ----------------------
model = AuxAutoencoder(in_dim=9, hidden_dim=HIDDEN_DIM, proj_dim=PROJ_DIM).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----------------------
# 训练循环
# ----------------------
best_val_loss = float('inf')
for epoch in range(1, NUM_EPOCHS + 1):
    # 训练
    model.train()
    total_loss = 0.0
    for aux in train_loader:
        aux = aux.to(DEVICE).float()
        optimizer.zero_grad()
        _, aux_rec = model(aux)
        loss = criterion(aux_rec, aux)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * aux.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for aux in val_loader:
            aux = aux.to(DEVICE).float()
            _, aux_rec = model(aux)
            val_loss += criterion(aux_rec, aux).item() * aux.size(0)
    avg_val_loss = val_loss / len(val_loader.dataset)

    print(f"Epoch {epoch}/{NUM_EPOCHS}  Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}")

    # 保存最优 encoder
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_path = os.path.join(DATA_BASE, 'models', 'aux_projector_encoder.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Saved best encoder to {save_path}")

print("Pretraining complete.")
