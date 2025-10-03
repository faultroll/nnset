# dataset.py
import numpy as np
import torch

def make_sin_dataset(n_points=1000, device="cpu"):
    x = np.linspace(0, 2*np.pi, n_points).reshape(-1,1).astype(np.float32)
    y = np.sin(x).astype(np.float32)

    x_train = torch.from_numpy(x).to(device)
    y_train = torch.from_numpy(y).to(device)

    return x_train, y_train
