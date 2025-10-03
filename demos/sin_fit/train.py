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

    """ with torch.no_grad():
        y_pred = G(x_train).cpu().numpy()
        y_true = y_train.cpu().numpy()
        x_axis = x_train.cpu().numpy()
    plt.figure(figsize=(8,4))
    plt.plot(x_axis, y_true, label="True sin(x)", linestyle="--")
    plt.plot(x_axis, y_pred, label="Predicted", linestyle="--")
    plt.legend()
    plt.show() """


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
    # train_d_bce()
    # train_d_wasserstein()

    """ with torch.no_grad():
        y_pred = model(x_train)
        print("MSE:", mse(y_pred, y_train))
        print("MAE:", mae(y_pred, y_train))
        print("R2 :", r2(y_pred, y_train))

    # 导出 ONNX
    dummy_input = torch.randn(1,1).to(device)
    torch.onnx.export(model, dummy_input, "outputs/mlp_sin.onnx",
                      input_names=["x"], output_names=["y"]) """
