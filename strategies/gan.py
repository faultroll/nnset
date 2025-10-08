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


# nnset/gans/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 统一的对抗损失接口 (mode: 'bce'|'hinge'|'wgan-gp')
class AdversarialLoss(nn.Module):
    def __init__(self, mode='bce'):
        super().__init__()
        assert mode in ('bce','hinge','wgan-gp')
        self.mode = mode
        if mode == 'bce':
            self.criterion = nn.BCELoss()
    def d_loss(self, d_real, d_fake):
        if self.mode == 'bce':
            real_labels = torch.ones_like(d_real)
            fake_labels = torch.zeros_like(d_fake)
            return self.criterion(d_real, real_labels) + self.criterion(d_fake, fake_labels)
        if self.mode == 'hinge':
            # hinge loss for D: mean(max(0,1 - D(x))) + mean(max(0,1 + D(G(z))))
            loss_real = F.relu(1.0 - d_real).mean()
            loss_fake = F.relu(1.0 + d_fake).mean()
            return loss_real + loss_fake
        if self.mode == 'wgan-gp':
            # WGAN (D aims to maximize real - fake) -> return -(mean(real)-mean(fake)) as minimization
            return -(d_real.mean() - d_fake.mean())

    def g_loss(self, d_fake):
        if self.mode == 'bce':
            real_labels = torch.ones_like(d_fake)
            return self.criterion(d_fake, real_labels)
        if self.mode == 'hinge':
            return -d_fake.mean()
        if self.mode == 'wgan-gp':
            return -d_fake.mean()

def reconstruction_loss(pred, target, mode='mse'):
    if mode == 'mse':
        return F.mse_loss(pred, target)
    if mode == 'l1':
        return F.l1_loss(pred, target)

# nnset/gans/strategy.py
import torch

class GANTrainingStrategy:
    """
    封装 D-step / G-step 流程
    cfg 包含: n_epochs,n_critic,lambda_rec,lambda_adv,device,adv_mode,recon_mode
    """
    def __init__(self, G, D, optimG, optimD, dataloader, cfg, adv_loss):
        self.G, self.D = G, D
        self.optG, self.optD = optimG, optimD
        self.dataloader = dataloader
        self.cfg = cfg
        self.adv_loss = adv_loss

    def train(self):
        device = self.cfg.device
        G, D = self.G.to(device), self.D.to(device)
        for epoch in range(self.cfg.n_epochs):
            for i, batch in enumerate(self.dataloader):
                x = batch['x'].to(device)
                y = batch['y'].to(device)

                # --- D steps ---
                for _ in range(self.cfg.n_critic):
                    self.optD.zero_grad()
                    with torch.no_grad():
                        y_fake = self.G(x)
                    d_real = self.D(torch.cat([x, y], dim=1))
                    d_fake = self.D(torch.cat([x, y_fake], dim=1))
                    loss_D = self.adv_loss.d_loss(d_real, d_fake)
                    # optional: add gradient penalty for wgan-gp externally
                    loss_D.backward()
                    self.optD.step()

                # --- G step ---
                self.optG.zero_grad()
                y_gen = self.G(x)
                d_gen = self.D(torch.cat([x, y_gen], dim=1))
                adv_term = self.adv_loss.g_loss(d_gen)
                rec_term = reconstruction_loss(y_gen, y, mode=self.cfg.recon_mode)
                loss_G = self.cfg.lambda_rec * rec_term + self.cfg.lambda_adv * adv_term

                # 插入蒸馏 hook（如果存在）
                if hasattr(self, 'distill_hook') and self.distill_hook is not None:
                    loss_G = loss_G + self.distill_hook(self.G, x, y, self.cfg)

                loss_G.backward()
                self.optG.step()

            if epoch % self.cfg.log_interval == 0:
                print(f"Epoch {epoch}: rec={rec_term.item():.6f}, D={loss_D.item():.6f}, G={loss_G.item():.6f}")

    def set_distill_hook(self, hook_fn):
        """hook_fn(G,x,y,cfg) -> extra_loss"""
        self.distill_hook = hook_fn
