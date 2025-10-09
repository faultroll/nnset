""" 
Encoder：可有可无，通常固定特征提取
Block：核心可训练特征提取
Readout：
如果任务是函数拟合或系数预测 → 线性 / Fourier 基映射
如果任务是分类/识别 → 线性映射到 logits
本质就是 Decoder，但强调从 Block 输出生成最终目标
"""

import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    """
    MLP Block: 提取特征向量，显式作为 Fourier 基系数输入 Readout
    """
    def __init__(self, in_dim=1, hidden_dim=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: [B,1] 坐标输入
        return: [B, out_dim] 作为 Fourier 系数输入
        """
        return self.net(x)

class FourierReadout(nn.Module):
    """
    Fourier Readout: 将 feature 映射为 sin/cos 系数，并计算最终输出
    """
    def __init__(self, feat_dim, n_bases=32, max_freq=10.0):
        super().__init__()
        self.n_bases = n_bases
        self.max_freq = max_freq
        self.coef_proj = nn.Linear(feat_dim, n_bases*2)  # sin + cos coeffs

    def forward(self, feat, x_coord):
        """
        feat: [B, feat_dim]
        x_coord: [B,1] 输入坐标
        """
        coef = self.coef_proj(feat)  # [B, 2*n_bases]
        freqs = torch.linspace(1.0, self.max_freq, steps=self.n_bases, device=feat.device)
        x_proj = x_coord @ freqs.view(1,-1)  # [B, n_bases]
        sin_terms = torch.sin(x_proj)
        cos_terms = torch.cos(x_proj)
        basis = torch.cat([sin_terms, cos_terms], dim=1)  # [B, 2*n_bases]
        y = (coef * basis).sum(dim=1, keepdim=True)       # [B,1]
        return y

class SinFitNetwork(nn.Module):
    """
    组合 MLPBlock + FourierReadout
    """
    def __init__(self, in_dim=1, block_hidden=64, block_out=64, n_bases=32):
        super().__init__()
        self.block = MLPBlock(in_dim, block_hidden, block_out)
        self.readout = FourierReadout(block_out, n_bases=n_bases)

    def forward(self, x):
        feat = self.block(x)
        y = self.readout(feat, x)
        return y

net = SinFitNetwork(in_dim=1, block_hidden=64, block_out=64, n_bases=32)
x = torch.linspace(-3.14, 3.14, 128).unsqueeze(1)  # [128,1]
y_true = torch.sin(x*2)  # target
y_pred = net(x)
loss = nn.MSELoss()(y_pred, y_true)
loss.backward()
