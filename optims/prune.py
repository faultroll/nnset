# prune.py
import copy
import torch
import torch.nn.utils.prune as prune

def unstructured_prune(model, amount=0.2, inplace=False):
    """
    非结构化剪枝，稀疏化权重
    amount: 0~1，剪掉的比例
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

def structured_prune(model, amount=0.2, inplace=False):
    """
    结构化剪枝，剪掉整个通道或卷积核
    amount: 0~1，剪掉比例
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
        elif isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=1)
    return model

def remove_prune_reparam(model, inplace=False):
    """
    移除剪枝产生的mask和原参数的reparam，得到真正稀疏的权重
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    for name, module in model.named_modules():
        for pname, _ in module.named_parameters(recurse=False):
            if 'mask' in pname:
                prune.remove(module, pname.replace('_mask',''))
    return model

def rebuild_structured_model(model, dummy_input, device='cpu'):
    """
    针对结构化剪枝后，生成新的模型结构
    dummy_input: 用于推理shape
    """
    # 这里的实现可以结合 nnset.designs.networks 工具，把剩余权重拷贝到新的模型实例
    # 简单demo：
    model.to(device).eval()
    with torch.no_grad():
        _ = model(dummy_input.to(device))
    return model

""" from nnset.optims.prune import unstructured_prune, structured_prune, remove_prune_reparam, rebuild_structured_model

# 非结构化剪枝 + finetune
model = unstructured_prune(model, amount=0.3)
# finetune...

# 结构化剪枝 + rebuild + finetune
model = structured_prune(model, amount=0.3)
model = remove_prune_reparam(model)
model = rebuild_structured_model(model, dummy_input=torch.randn(1,1))  # 假设输入是(1,1)
# finetune... """
