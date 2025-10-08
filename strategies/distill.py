
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
