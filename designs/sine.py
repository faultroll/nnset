import torch
import torch.nn as nn
import math

# Sine Block (from SIREN)
# -----------------------
# Source: Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions", NeurIPS 2020
# 功能: 函数拟合 / 隐式表示 (Implicit Representation)
# 结构: y = sin(Wx + b)
class Sine_Block(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        self._init_weights()
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
    def _init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-math.sqrt(6 / self.linear.in_features) / self.omega_0,
                                        math.sqrt(6 / self.linear.in_features) / self.omega_0)
