import torch
import torch.nn as nn

# Operation层 纯数学操作
class PointwiseConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                         groups=1, bias=bias)
class DepthwiseConv2d(nn.Conv2d):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, groups=in_channels, bias=bias)

# Layer层 添加归一化与激活
class DSC_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 bn=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        padding = kernel_size // 2
        self.dw = DepthwiseConv2d(in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.pw = PointwiseConv2d(in_channels, out_channels, bias=False)
        self.bn = bn(out_channels) if bn is not None else nn.Identity()
        self.act = act(inplace=True) if act is not None else nn.Identity()
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

# Block层 组合多层形成功能单元

