# wgan_gp_toy.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nnset.designs.mlp import MLP_Block
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 数据 ---
N = 1000
x = np.linspace(0, 2*np.pi, N).reshape(-1,1).astype(np.float32)
y = np.sin(x).astype(np.float32)

x_train = torch.from_numpy(x).to(device)
y_train = torch.from_numpy(y).to(device)

# --- Generator (simple MLP) ---
class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP_Block(in_features=1, hidden_sizes=[16,32], out_features=1, activation='swish')
    def forward(self, x):
        return self.net(x)

# --- Critic / Discriminator (linear output, no sigmoid) ---
class DNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, 1), # wasserstein LINEAR output!
            # nn.Sigmoid(), # bce
        )
    def forward(self, x):
        return self.net(x)

G = GNet().to(device)
D = DNet().to(device)

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


# --- adversarial train ---
n_epochs = 2000
n_critic = 5
# batch_size = x_train.size(0)  # full-batch for toy example

def train_d_bce():
    optG = optim.Adam(G.parameters(), lr=1e-3)
    optD = optim.Adam(D.parameters(), lr=1e-3)

    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    lambda_rec = 10.0
    lambda_adv = 1.0

    for epoch in range(n_epochs):
        # D steps
        for _ in range(n_critic):
            optD.zero_grad()
            y_fake = G(x_train).detach()
            d_real = D(torch.cat([x_train, y_train], dim=1))
            d_fake = D(torch.cat([x_train, y_fake], dim=1))
            # -[y_true*log(y_pred) + (1-y_true)*log(1-y_pred)]
            loss_D = bce_loss(d_real, torch.ones_like(d_real)) + bce_loss(d_fake, torch.zeros_like(d_fake))
            loss_D_total = loss_D
            loss_D_total.backward()
            optD.step()

        # G step
        optG.zero_grad()
        y_gen = G(x_train)
        d_gen = D(torch.cat([x_train, y_gen], dim=1))
        loss_G = bce_loss(d_gen, torch.ones_like(d_gen))
        loss_rec = mse_loss(y_gen, y_train)
        loss_G_total = lambda_rec * loss_rec + lambda_adv * loss_G
        loss_G_total.backward()
        optG.step()
        
        if epoch % 100 == 0:
            print(f"BCE Epoch {epoch}: loss_rec={loss_rec.item():.6f}, loss_D={loss_D_total.item():.4f}, loss_G={loss_G_total.item():.4f}")

def gradient_penalty(D, real, fake):
    # real, fake: [N, dim], interpolate between them
    alpha = torch.rand(real.size(0), 1).to(device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interp)
    grads = torch.autograd.grad(outputs=d_interp,
                                inputs=interp,
                                grad_outputs=torch.ones_like(d_interp),
                                create_graph=True, retain_graph=True)[0]
    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp
def train_d_wasserstein():
    optG = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optD = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))

    mse_loss = nn.MSELoss()
    lambda_gp = 10.0
    lambda_rec = 10.0  # example weight for reconstruction (like SDDN)
    lambda_adv = 1.0

    for epoch in range(n_epochs):
        # D steps
        for _ in range(n_critic):
            optD.zero_grad()
            y_fake = G(x_train).detach()
            d_real = D(torch.cat([x_train, y_train], dim=1))
            d_fake = D(torch.cat([x_train, y_fake], dim=1))
            loss_D = d_fake.mean() - d_real.mean()
            gp = gradient_penalty(D, torch.cat([x_train, y_train], dim=1), torch.cat([x_train, y_fake], dim=1))
            loss_D_total = loss_D + lambda_gp * gp
            loss_D_total.backward()
            optD.step()

        # G step
        optG.zero_grad()
        y_gen = G(x_train)
        d_gen = D(torch.cat([x_train, y_gen], dim=1))
        loss_G = - d_gen.mean()
        loss_rec = mse_loss(y_gen, y_train) # you can combine with reconstruction loss if desired (like SDDN)
        loss_G_total = lambda_rec * loss_rec + lambda_adv * loss_G
        loss_G_total.backward()
        optG.step()

        if epoch % 100 == 0:
            print(f"Wasserstein Epoch {epoch}: loss_rec={loss_rec.item():.6f}, loss_D={loss_D_total.item():.4f}, loss_G={loss_G_total.item():.4f}")

if __name__ == '__main__':
    train_g()
    train_d_bce()
    # train_d_wasserstein()


# Knowledge Distillation Loss (Logit-based)
# Computes soft target loss between teacher and student outputs.

import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    # Args:
    #     temperature (float): softmax temperature (>1 softens distributions)
    #     alpha (float): weight for KD loss vs hard-label loss
    #     hard_loss_fn (callable): optional supervised loss (e.g. nn.CrossEntropyLoss)
    def __init__(self, temperature: float=4.0, alpha: float=0.5, hard_loss_fn=None):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.hard_loss_fn = hard_loss_fn
    # Args:
    #     student_logits (Tensor): [B, C]
    #     teacher_logits (Tensor): [B, C]
    #     targets (Tensor): optional hard labels for supervised loss
    def forward(self, student_logits, teacher_logits, targets=None):
        # --- Soft targets (KL divergence) ---
        p_student = F.log_softmax(student_logits / self.T, dim=1)
        p_teacher = F.softmax(teacher_logits / self.T, dim=1)
        kd_loss = F.kl_div(p_student, p_teacher, reduction='batchmean') * (self.T ** 2)
        # --- Hard targets (optional supervised) ---
        if targets is not None and self.hard_loss_fn is not None:
            hard_loss = self.hard_loss_fn(student_logits, targets)
            return self.alpha * hard_loss + (1 - self.alpha) * kd_loss
        else:
            return kd_loss

# strategies/self_distill.py
from nnset.optims.distill.kd_loss import KDLoss

def self_distill_hook(student, teacher, batch, cfg):
    kd = KDLoss(temperature=cfg.distill.T, alpha=cfg.distill.alpha)
    student_logits = student(batch.x)
    with torch.no_grad():
        teacher_logits = teacher(batch.x)
    loss = kd(student_logits, teacher_logits, batch.y)
    return loss


# nnset/optims/distill/kd_base.py
import torch
import torch.nn.functional as F

class KDParams:
    # 只有logit要软标签
    def __init__(self, temperature=4.0, alpha=0.5):
        self.T = temperature
        self.alpha = alpha
    def soft_targets(self, logits):
        return F.softmax(logits / self.T, dim=1)
    def hard_soft_mix(self, student_logits, teacher_logits, targets):
        p_student = F.log_softmax(student_logits / self.T, dim=1)
        p_teacher = self.soft_targets(teacher_logits)
        soft_loss = F.kl_div(p_student, p_teacher, reduction='batchmean') * (self.T ** 2)
        hard_loss = F.cross_entropy(student_logits, targets)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

    # Feature / Attention / Relational 的辅助函数
    @staticmethod
    def mse_loss(x, y):
        return F.mse_loss(x, y)

    @staticmethod
    def cosine_loss(x, y):
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        return 1 - (x_norm * y_norm).sum(dim=-1).mean()

# strategies/self_distill.py
from nnset.optims.distill.kd_base import KDParams

def self_distill_hook(student_outputs, teacher_outputs, targets=None, kd_params=None, kd_type="logit"):
    """
    student_outputs / teacher_outputs 可以是 logits 或 feature
    """
    if kd_type == "logit":
        return kd_params.hard_soft_mix(student_outputs, teacher_outputs, targets)

    elif kd_type == "feature":
        # student_outputs, teacher_outputs: [B, C, H, W] 或 [B, D]
        return kd_params.mse_loss(student_outputs, teacher_outputs)

    elif kd_type == "attention":
        # 假设 student_outputs, teacher_outputs 是注意力图 [B, N, N]
        return kd_params.cosine_loss(student_outputs, teacher_outputs)

    elif kd_type == "relational":
        # relational: student_features, teacher_features
        # 计算 pairwise cosine 矩阵
        s = F.normalize(student_outputs, dim=-1)
        t = F.normalize(teacher_outputs, dim=-1)
        rel_s = s @ s.transpose(-1, -2)
        rel_t = t @ t.transpose(-1, -2)
        return kd_params.mse_loss(rel_s, rel_t)

    else:
        raise ValueError(f"Unknown KD type {kd_type}")
