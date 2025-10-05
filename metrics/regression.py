
# import torch
import numpy as np

def mse(y_pred, y_true):
    # return torch.nn.functional.mse_loss(y_pred, y_true).item()
    return np.mean((y_true - y_pred) ** 2)

def mae(y_pred, y_true):
    # return torch.mean(torch.abs(y_pred - y_true)).item()
    return np.mean(np.abs(y_true - y_pred))

def r2(y_pred, y_true):
    # ss_res = torch.sum((y_true - y_pred) ** 2)
    # ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot
