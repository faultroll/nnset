# quant.py
import copy, torch
from torch.quantization import quantize_dynamic, get_default_qconfig, prepare, convert, prepare_qat

def dynamic_quantize(model, dtype=torch.qint8, inplace=False):
    """
    动态量化，适用于任意模型
    """
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=dtype)
    return quantized_model

def static_ptq_quantize(model, calib_loader, inplace=False, qconfig=None, device='cpu'):
    """
    静态PTQ量化
    model: nn.Module
    calib_loader: 用于校准的代表性数据加载器
    qconfig: 可选，PyTorch默认get_default_qconfig('fbgemm')
    """
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
    return model

def prepare_qat_model(model, qconfig=None, inplace=False):
    """
    QAT准备函数，训练时使用
    """
    if not inplace:
        model = copy.deepcopy(model)
    if qconfig is None:
        qconfig = get_default_qconfig('fbgemm')
    model.qconfig = qconfig
    prepare_qat(model, inplace=True)
    return model


""" from nnset.optims.quant import dynamic_quantize, static_ptq_quantize

# 动态量化
quant_model = dynamic_quantize(model)

# 静态量化（PTQ）需要校准数据
quant_model = static_ptq_quantize(model, calib_loader=train_loader) """


# nnset/optims/quant.py (ONNX PTQ part)
import os
import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static, quantize_dynamic

class NumpyCalibrationDataReader(CalibrationDataReader):
    """
    CalibrationDataReader for onnxruntime.quantization.quantize_static.
    Expects 'calib_data' to be an iterable of numpy inputs matching the ONNX input shape.
    Example: calib_data = [ {"input": np_float32_array}, ... ]
    """
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

def onnx_static_quantize(onnx_model_path: str, calib_inputs, input_name=None, out_path: str = None,
                         quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QUInt8,
                         weight_type=QuantType.QInt8):
    """
    onnx_model_path: path to exported float ONNX
    calib_inputs: list of numpy arrays (shape matches model input) or list of dicts {input_name: array}
    input_name: if None, infer from model graph (use first input)
    Returns: path to quantized onnx
    """
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
    return out_path

# nnset/optims/quant.py (QAT part)
import copy
import torch
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert

def prepare_qat_model(model, backend='fbgemm', inplace=False):
    """
    Returns a model prepared for QAT (with fake-quant nodes).
    """
    if not inplace:
        model = copy.deepcopy(model)
    model.train()
    qconfig = get_default_qat_qconfig(backend)
    model.qconfig = qconfig
    prepare_qat(model, inplace=True)  # inplace wraps modules with FakeQuant
    return model

# 不convert就可以导出float模型，供后续onnx的ptq使用
def convert_qat_model(model, inplace=False):
    """
    Convert a trained QAT (after training) model into a quantized model.
    """
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    quantized = convert(model, inplace=False)
    return quantized

# training wrapper (simplified)
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
    return model
