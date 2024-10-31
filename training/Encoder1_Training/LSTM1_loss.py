import torch
import torch.nn as nn

# 自定义加权损失函数
class WeightedMSELoss(nn.Module):
    def __init__(self, close_weight=1.0, volume_weight=0.5):
        super(WeightedMSELoss, self).__init__()
        self.close_weight = close_weight
        self.volume_weight = volume_weight
        self.mse = nn.MSELoss(reduction='none')  # 保留每个元素的损失

    def forward(self, output, target):
        # 拆分输出和目标
        close_output, volume_output = output[:, :, 0], output[:, :, 1]
        close_target, volume_target = target[:, :, 0], target[:, :, 1]

        # 计算逐元素的 MSE 损失
        close_loss = self.mse(close_output, close_target)
        volume_loss = self.mse(volume_output, volume_target)

        # 给每个时间步的损失加权（假设靠后的时间步越重要）
        seq_length = close_loss.size(1)
        time_weights = torch.linspace(1.0, 2.0, seq_length).to(close_loss.device)  # 线性增加的时间权重，从 1.0 到 2.0

        # 计算加权的损失
        close_loss_weighted = close_loss * self.close_weight * time_weights
        volume_loss_weighted = volume_loss * self.volume_weight * time_weights

        # 计算总损失（对 batch 和序列长度求平均）
        total_loss = (close_loss_weighted + volume_loss_weighted).mean()

        return total_loss