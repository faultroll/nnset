""" import copy, torch
from torch.quantization import quantize_dynamic, get_default_qconfig, prepare, convert, prepare_qat

# 动态量化，适用于任意模型
def dynamic_quantize(model, dtype=torch.qint8, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=dtype)
    return quantized_model

# 静态PTQ量化
# model: nn.Module
# calib_loader: 用于校准的代表性数据加载器
# qconfig: 可选，PyTorch默认get_default_qconfig('fbgemm')
def static_ptq_quantize(model, calib_loader, inplace=False, qconfig=None, device='cpu'):
    
    if not inplace:
        model = copy.deepcopy(model)
    model.to(device).eval()
    if qconfig is None:
        qconfig = get_default_qconfig('fbgemm')
    model.qconfig = qconfig
    prepare(model, inplace=True)
    
    # 校准
    with torch.no_grad():
        for xb, _ in calib_loader:
            xb = xb.to(device)
            model(xb)
    
    convert(model, inplace=True)
    return model """

""" from nnset.optims.quant import dynamic_quantize, static_ptq_quantize

# 动态量化
quant_model = dynamic_quantize(model)

# 静态量化（PTQ）需要校准数据
quant_model = static_ptq_quantize(model, calib_loader=train_loader) """


""" import os
import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static, quantize_dynamic

# CalibrationDataReader for onnxruntime.quantization.quantize_static.
# Expects 'calib_data' to be an iterable of numpy inputs matching the ONNX input shape.
# Example: calib_data = [ {"input": np_float32_array}, ... ]
class NumpyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_name, calib_data):
        self.input_name = input_name
        # calib_data: list of numpy arrays or list of dicts
        # Normalize to list of dicts
        normalized = []
        for d in calib_data:
            if isinstance(d, dict):
                normalized.append(d)
            else:
                normalized.append({self.input_name: d})
        self.data = normalized
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.data):
            return None
        item = self.data[self.idx]
        self.idx += 1
        return item

def onnx_dynamic_quantize(onnx_model_path: str, out_path: str = None, weight_type=QuantType.QInt8):
    if out_path is None:
        out_path = onnx_model_path.replace(".onnx", ".quant.dynamic.onnx")
    quantize_dynamic(model_input=onnx_model_path, model_output=out_path, weight_type=weight_type)
    return out_path

# onnx_model_path: path to exported float ONNX
# calib_inputs: list of numpy arrays (shape matches model input) or list of dicts {input_name: array}
# input_name: if None, infer from model graph (use first input)
# Returns: path to quantized onnx
def onnx_static_quantize(onnx_model_path: str, calib_inputs, input_name=None, out_path: str = None,
                         quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QUInt8,
                         weight_type=QuantType.QInt8):
    if out_path is None:
        out_path = onnx_model_path.replace(".onnx", ".quant.static.onnx")

    model = onnx.load(onnx_model_path)
    # infer input name if not provided
    if input_name is None:
        input_name = model.graph.input[0].name

    dr = NumpyCalibrationDataReader(input_name, calib_inputs)
    quantize_static(model_input=onnx_model_path,
                    model_output=out_path,
                    calibration_data_reader=dr,
                    quant_format=quant_format,
                    per_channel=per_channel,
                    activation_type=activation_type,
                    weight_type=weight_type)
    return out_path """


import copy
import torch
import torch.nn as nn
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert

import warnings
warnings.filterwarnings("ignore", message="Please use quant_min and quant_max")
warnings.filterwarnings("ignore", message="_aminmax is deprecated")


# Returns a model prepared for QAT (with fake-quant nodes).
def prepare_qat_model(model, backend='qnnpack', inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    model.train()
    # 1. default
    qconfig = get_default_qat_qconfig(backend) # 'qnnpack': arm, 'fbgemm': x86
    # 2. user
    # from torch.ao.quantization import QConfig, FakeQuantize
    # from torch.ao.quantization import MovingAverageMinMaxObserver, PerChannelMinMaxObserver
    # from torch.ao.quantization import default_observer, default_per_channel_weight_fake_quant
    # 2.1 
    # custom_fake_quant = FakeQuantize.with_args(
    #     observer=MovingAverageMinMaxObserver,
    #     quant_min=-128,
    #     quant_max=127,
    #     dtype=torch.qint8,
    #     qscheme=torch.per_tensor_symmetric,
    #     reduce_range=False, # True: qint8 范围由 [-128,127] 缩减为 [-64,63] 避免累加溢出
    # )
    # qconfig = QConfig(
    #     activation=custom_fake_quant,
    #     weight=custom_fake_quant,
    # )
    # 2.2 
    # # per-tensor asymmetric
    # activation_observer = MovingAverageMinMaxObserver.with_args(
    #     dtype=torch.quint8,
    #     qscheme=torch.per_tensor_affine,
    # )
    # # per-channel symmetric
    # weight_observer = PerChannelMinMaxObserver.with_args(
    #     dtype=torch.qint8,
    #     qscheme=torch.per_channel_symmetric,   # 
    #     ch_axis=0,                             # 通常为输出通道
    # )
    # qconfig = QConfig(
    #     activation=FakeQuantize.with_args(observer=activation_observer),
    #     weight=FakeQuantize.with_args(observer=weight_observer)
    # )
    # 2.3
    # activation_observer = default_observer
    # weight_observer = default_per_channel_weight_fake_quant
    # qconfig = QConfig(activation=activation_observer, weight=weight_observer)
    model.qconfig = qconfig
    prepare_qat(model, inplace=True)  # inplace wraps modules with FakeQuant
    return model

# 不convert就可以导出float模型，供后续onnx的ptq使用
def _is_fakequant_module(m):
    # 尽量稳妥地检测 FakeQuant 类（覆盖常见类名）
    from torch.ao.quantization import fake_quantize
    cls_names = (
        'FakeQuantize',
        'FusedMovingAvgObsFakeQuantize',
        'DefaultFakeQuantize',  # alternate names
    )
    # isinstance check if class is available
    try:
        if isinstance(m, fake_quantize.FakeQuantize):
            return True
    except Exception:
        pass
    # fallback by name match (handles fused types)
    name = m.__class__.__name__
    return any(n in name for n in cls_names)
def _replace_fakequant_with_identity(model):
    # 遍历 model，若某个子模块是 FakeQuant (或 fused variant)，就用 nn.Identity() 替换之。
    # 修改是 inplace。
    for name, child in list(model.named_children()):
        if _is_fakequant_module(child):
            # print("Replacing FakeQuant:", name, child.__class__.__name__)
            setattr(model, name, nn.Identity())
        else:
            _replace_fakequant_with_identity(child)
def remove_fake_quant(model):
    # for name, module in model.named_children():
    #     if hasattr(module, 'activation_post_process'):
    #         del module.activation_post_process
    #     remove_fake_quant(module)
    # model.apply(torch.ao.quantization.disable_observer)
    # model.apply(torch.ao.quantization.disable_fake_quant)
    # best-effort disable
    try:
        from torch.ao import quantization as tq
        model.apply(tq.disable_observer)
        model.apply(tq.disable_fake_quant)
    except Exception:
        pass
    # Replace FakeQuant modules with Identity
    _replace_fakequant_with_identity(model)
    # verify: no FakeQuant instances left
    from torch.ao.quantization import fake_quantize
    bad = []
    for n, m in model.named_modules():
        # check isinstance and classname fallback
        try:
            if isinstance(m, fake_quantize.FakeQuantize):
                bad.append((n, m.__class__.__name__))
        except Exception:
            if 'FakeQuant' in m.__class__.__name__ or 'FusedMovingAvgObsFakeQuantize' in m.__class__.__name__:
                bad.append((n, m.__class__.__name__))
    if len(bad) > 0:
        print("Warning: still found FakeQuant modules:", bad)
        # as a fallback, try again to replace by name
        for n, m in list(model.named_modules()):
            if 'FakeQuant' in m.__class__.__name__ or 'FusedMovingAvgObs' in m.__class__.__name__:
                # find parent and replace
                parent_path = n.rsplit('.', 1)
                if len(parent_path) == 1:
                    parent = model
                    attr = parent_path[0]
                else:
                    parent = dict(model.named_modules())[parent_path[0]]
                    attr = parent_path[1]
                setattr(parent, attr, nn.Identity())
    # final check (should be clean)
    for n, m in model.named_modules():
        if 'FakeQuant' in m.__class__.__name__:
            raise RuntimeError(f"Failed to remove FakeQuant module: {n} {m.__class__.__name__}")

# Convert a trained QAT (after training) model into a quantized model.
def convert_qat_model(model, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    quantized = convert(model, inplace=False)
    return quantized

""" # training wrapper (simplified)
def train_qat(model, train_loader, val_loader=None, epochs=10, lr=1e-4, device='cpu', optim_ctor=torch.optim.Adam):
    model.to(device)
    opt = optim_ctor(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for ep in range(epochs):
        model.train()
        for xb,yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # optional val eval...
    return model """
