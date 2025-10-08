import torch
import torch.nn as nn

# Residual Block
# -----------------
# Source: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
# 功能: 特征提取（可用于 1D / 2D / ND 信号）
# 结构: x -> Conv -> BN -> ReLU -> Conv -> BN -> SkipAdd -> ReLU
class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bn=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = bn(out_channels)
        self.act1 = act(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = bn(out_channels)
        self.shortcut = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        )
        self.final_act = act(inplace=True)
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.final_act(out)
