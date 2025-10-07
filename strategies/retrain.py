# strategy.py
def retrain_hook(model, batch, cfg):
    # 不需要修改
    return model.parameters(), cfg.lr

def finetune_hook(model, batch, cfg):
    # 冻结部分参数
    for name, param in model.named_parameters():
        if "feature" in name:
            param.requires_grad = False
    return filter(lambda p: p.requires_grad, model.parameters()), cfg.lr * 0.1

def self_distill_hook(model, teacher_model, batch, cfg, alpha=0.5):
    student_pred = model(batch.x)
    with torch.no_grad():
        teacher_pred = teacher_model(batch.x)
    loss = alpha * compute_loss(student_pred, batch.y) + (1-alpha) * distill_loss(student_pred, teacher_pred)
    return model.parameters(), cfg.lr, loss

# strategies/self_distill.py
from nnset.optims.distill.kd_loss import KDLoss

def self_distill_hook(student, teacher, batch, cfg):
    kd = KDLoss(cfg.distill.temperature)
    student_pred = student(batch.x)
    with torch.no_grad():
        teacher_pred = teacher(batch.x)
    loss = kd(student_pred, teacher_pred)
    return loss

# train.py
for epoch in range(cfg.epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        if cfg.strategy == "retrain":
            params, lr = retrain_hook(model, batch, cfg)
        elif cfg.strategy == "finetune":
            params, lr = finetune_hook(model, batch, cfg)

        if cfg.strategy != "self-distill":
            y_pred = model(batch.x)
            loss = compute_loss(y_pred, batch.y)
        else:
            params, lr, loss = self_distill_hook(model, teacher_model, batch, cfg)

        loss.backward()
        optimizer.step()
