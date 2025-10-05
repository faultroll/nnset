
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from dataset import make_sin_numpy

# python -m nnset.demos.sin_fit.train
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from nnset.designs.mlp import MLP_Block
from nnset.metrics.regression import mse, mae, r2

# --- Generator (simple MLP) ---
class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP_Block(in_features=1, hidden_sizes=[16,32], out_features=1, activation='swish')
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x, y = make_sin_numpy(n_points=400, x_range=(0, 4*np.pi))
x_train = torch.from_numpy(x).to(device)
y_train = torch.from_numpy(y).to(device)
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
