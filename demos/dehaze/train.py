
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
from nnset.optims.quant import prepare_qat_model, convert_qat_model, remove_fake_quant
from nnset.optims.prune import unstructured_prune, structured_prune, remove_prune_reparam, rebuild_structured_model
# from nnset.strategies.distill import distill_hook

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

def train1(model, device):
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
        print(f"train1 EPOCH {epoch:02d}, LOSS train {avg_loss:.4f}")

def export(model, device):
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

def train2(model_pruned, model_origin, device):
    criterion = N.LightDehazeLoss().to(device) # nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model_pruned.parameters(), lr=1e-2) # reduced
    train_dataset = DehazeDataset('./test-images/split_txt/val.txt') # validate
    train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for epoch in range(1, 31): # reduced
        model_pruned.train()
        optimizer.zero_grad()
        avg_loss = 0
        for i, (I, J_gt) in enumerate(train_loader):
            I = I.to(device)
            J_gt = J_gt.to(device)
            J_pred = model_pruned(I)
            loss_hard = criterion(J_pred, J_gt)
            # distill_hook(student_outputs, teacher_outputs, kd_type="feature")
            with torch.no_grad():
                J_pred_teacher = model_origin(I)
            loss_soft = criterion(J_pred, J_pred_teacher)
            alpha = 0.5
            loss = alpha * loss_hard + (1 - alpha) * loss_soft
            avg_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = avg_loss / (i + 1)
        print(f"train2 EPOCH {epoch:02d}, LOSS train {avg_loss:.4f}")

# 1. prune
def optimize1(model, device):
    # unstructured_prune(model, inplace=True)
    structured_prune(model, inplace=True)
    remove_prune_reparam(model, inplace=True)
    rebuild_structured_model(model, dummy_input=torch.randn(1, 3, 640, 480).to(device), device=device)
    return model
# 2. quant(qat)
def optimize2(model_pruned, model_origin, device):
    prepare_qat_model(model_pruned, inplace=True)
    # train, 'self'-distill
    train2(model_pruned, model_origin, device)
    convert_qat_model(model_pruned, inplace=True)
    remove_fake_quant(model_pruned) # for onnx export
# 3. quant(ptq)
# def optimize3(model, device):
#     onnx_static_quantize(model)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_origin = N.LightDehazeNet().to(device)
    train1(model_origin, device)
    # model_pruned = optimize1(model_origin, device)
    model_pruned = N.BILDNet().to(device)
    optimize2(model_pruned, model_origin, device)
    export(model_pruned, device)
    # optimize3(model_pruned, device)
    flops = torchflops(model_pruned, torch.randn(1, 3, 640, 480).to(device))
