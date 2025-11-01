
# dataset
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class DehazeDataset(Dataset):
    def __init__(self, txt_file):
        self.lines = open(txt_file).read().strip().split('\n')
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        hazy_path, gt_path = self.lines[idx].split()
        I_pil = Image.open(hazy_path).convert('RGB')
        J_gt_pil = Image.open(gt_path).convert('RGB')
        I_np = np.array(I_pil).astype(np.float32) / 255.0       # [H,W,3]
        J_gt_np = np.array(J_gt_pil).astype(np.float32) / 255.0 # [H,W,3]
        I = torch.from_numpy(I_np).permute(2, 0, 1)             # [3,H,W]
        J_gt = torch.from_numpy(J_gt_np).permute(2, 0, 1)       # [3,H,W]
        return I, J_gt


# train
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import network as N

# python -m nnset.demos.dehaze.train
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from nnset.metrics.complexity import torchflops
from nnset.optims.quant import prepare_qat_model, remove_fake_quant
from nnset.optims.prune import unstructured_prune, structured_prune, remove_prune_reparam, rebuild_structured_model

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = N.LightDehazeNet().to(device)
    criterion = N.LightDehazeLoss().to(device) # nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    train_dataset = DehazeDataset('./test-images/split_txt/train.txt')
    train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=True)
    model.apply(init_weights)
    for epoch in range(1, 31):
        model.train()
        optimizer.zero_grad()
        avg_loss = 0
        for i, (I, J_gt) in enumerate(train_loader):
            I = I.to(device)
            J_gt = J_gt.to(device)
            J_pred = model(I)
            loss = criterion(J_pred, J_gt)
            avg_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = avg_loss / (i + 1)
        print(f"EPOCH {epoch:02d}, LOSS train {avg_loss:.4f}")
    model.eval()
    input_dict = {
        'I': torch.randn(1, 3, 640, 480).to(device),
    }
    torch.onnx.export(model, input_dict, "lightdehazenet.onnx",
                    input_names=['I'],
                    output_names=['J_pred'],
                    dynamic_axes={
                        'I': {0: 'batch_size', 2: 'height', 3: 'width'},
                        'J_pred': {0: 'batch_size', 2: 'height', 3: 'width'}
                    },
                    opset_version=11)
    flops = torchflops(model, input_dict['I'])

# 1. prune
def optimize1():
    # unstructured_prune(model, inplace=True)
    structured_prune(model, inplace=True)
    remove_prune_reparam(model, inplace=True)
    rebuild_structured_model(model, dummy_input=torch.randn(1, 3, 640, 480), device=device)
# 2. quant(qat)
def optimize2():
    prepare_qat_model(model, inplace=True)
    # train, self-distill
# 3. quant(ptq)
# def optimize3():
#     remove_fake_quant(model) # for onnx export
#     onnx_static_quantize(model)

if __name__ == '__main__':
    train()
