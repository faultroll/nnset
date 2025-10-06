
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt

from dataset import make_sin_numpy

# python -m nnset.demos.sin_fit.train
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from nnset.designs.mlp import MLP_Block
from nnset.metrics.regression import mse, mae, r2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP_Block(in_features=1, hidden_sizes=[256,256,256], out_features=1, activation='relu')
    def forward(self, x):
        return self.net(x)

class FourierFeature(nn.Module):
    def __init__(self, in_features=1, M=64, scale=1.0, learnable=False):
        super().__init__()
        self.in_features = in_features
        self.M = M
        B = torch.randn(in_features, M) * scale
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    def forward(self, x):
        # x: (batch, in_features)
        x_proj = 2 * math.pi * (x @ self.B)  # shape (batch, M)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class GNetFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping_size = 64
        self.ff = FourierFeature(in_features=1, M=self.mapping_size)
        self.net = MLP_Block(in_features=2*self.mapping_size, hidden_sizes=[64,128,64], out_features=1, activation='relu')
    def forward(self, x):
        return self.net(self.ff(x))


x, y = make_sin_numpy(n_points=400, x_range=(0, 4*np.pi))
x_train = torch.from_numpy(x).to(device)
y_train = torch.from_numpy(y).to(device)
x, y = make_sin_numpy(n_points=400, x_range=(4*np.pi, 8*np.pi))
x_val = torch.from_numpy(x).to(device)
y_val = torch.from_numpy(y).to(device)
G = GNet().to(device)

# --- pretrain ---
def train_g():
    optG = optim.Adam(G.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for epoch in range(500):
        optG.zero_grad()
        y_pred = G(x_train)
        loss = mse(y_pred, y_train)
        loss.backward()
        optG.step()
        if (epoch+1) % 100 == 0:
            print(f"Pretrain G epoch {epoch+1}, mse={loss.item():.6f}")

def evaluate_g():
    with torch.no_grad():
        y_pred = G(x_train).cpu().numpy()
        y_true = y_train.cpu().numpy()
        x_axis = x_train.cpu().numpy()
    plt.figure(figsize=(8,4))
    plt.plot(x_axis, y_true, label="True sin(x)", linestyle="--")
    plt.plot(x_axis, y_pred, label="Predicted", linestyle="--")
    plt.legend()
    plt.show()

    print("MSE:", mse(y_pred, y_true))
    print("MAE:", mae(y_pred, y_true))
    print("R2 :", r2(y_pred, y_true))

    """ with torch.no_grad():
        x_sample = x_train[:1024]  # torch tensor (N,1)
        x_proj = 2*math.pi * (x_sample @ G.ff.B)  # (N, M)
        print("x range:", x_sample.min().item(), x_sample.max().item())
        print("x_proj mean/std:", x_proj.mean().item(), x_proj.std().item())
        print("sin feat std:", torch.sin(x_proj).std().item(), "cos feat std:", torch.cos(x_proj).std().item()) """

def export_g():
    # --- 导出 ONNX ---
    dummy_input = torch.randn(1,1).to(device)
    torch.onnx.export(G, dummy_input, "outputs/mlp_sin.onnx",
                      input_names=["x"], output_names=["y"],
                      dynamic_axes={'x': {0: 'batch_size'}, 'y': {0: 'batch_size'}},
                      opset_version=11)

if __name__ == '__main__':
    train_g()
    evaluate_g()
    export_g()
