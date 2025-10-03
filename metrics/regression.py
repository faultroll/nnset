

import torch.nn.functional as F

with torch.no_grad():
    y_pred = model(x_train)
    mse = F.mse_loss(y_pred, y_train).item()
print("MSE:", mse)


mae = torch.mean(torch.abs(y_pred - y_train)).item()
print("MAE:", mae)


ss_res = torch.sum((y_train - y_pred) ** 2)
ss_tot = torch.sum((y_train - torch.mean(y_train)) ** 2)
r2 = 1 - ss_res / ss_tot
print("R²:", r2.item())


with torch.no_grad():
    y_pred = model(x_train)
    print("MSE:", mse(y_pred, y_train))
    print("MAE:", mae(y_pred, y_train))
    print("R2 :", r2(y_pred, y_train))
