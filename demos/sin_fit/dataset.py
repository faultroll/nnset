
import numpy as np

def make_sin_numpy(n_points=200, x_range=(0, 2*np.pi)):
    x = np.linspace(x_range[0], x_range[1], n_points).reshape(-1,1).astype(np.float32)
    y = np.sin(x).astype(np.float32)
    return x, y

""" def make_sin_torch(n_points=200, x_range=(0,2*np.pi), device='cpu'):
    # 在函数内部导入 torch，避免在导入模块时强制要求 torch 安装
    import torch
    x = np.linspace(x_range[0], x_range[1], n_points).reshape(-1,1).astype(np.float32)
    y = np.sin(x).astype(np.float32)
    # x, y = make_sin_numpy(n_points, x_range)
    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)
    return x_t, y_t """
