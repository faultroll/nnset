
import torch
from torch.utils.flop_counter import FlopCounterMode
from typing import Union,Tuple

def torchflops(model, inp:Union[torch.Tensor,Tuple], with_backward=False):
    istrain = model.training
    model.eval()
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)
    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops = flop_counter.get_total_flops()
    if istrain:
        model.train()
    print(total_flops)
    return total_flops


import numpy as np

def compute_psnr(img1, img2, max_pixel=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 10 * np.log10((max_pixel ** 2) / mse)
    print(psnr_value)
    return psnr_value

def avg_pool2d_np(img, window_size, padding):
    rows, cols, channels = img.shape
    padded_img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    pooled_img = np.zeros_like(img)
    for i in range(rows):
        for j in range(cols):
            patch = padded_img[i:i+window_size, j:j+window_size, :]
            pooled_img[i, j, :] = np.mean(patch, axis=(0, 1))
    return pooled_img
def compute_ssim(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    mu_x = avg_pool2d_np(pred, window_size, padding=window_size//2)
    mu_y = avg_pool2d_np(target, window_size, padding=window_size//2)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x2 = avg_pool2d_np(pred * pred, window_size, padding=window_size//2) - mu_x2
    sigma_y2 = avg_pool2d_np(target * target, window_size, padding=window_size//2) - mu_y2
    sigma_xy = avg_pool2d_np(pred * target, window_size, padding=window_size//2) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    ssim_value = ssim_map.mean()
    print(ssim_value)
    return ssim_value
