# predict.py
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt

from dataset import make_sin_numpy

# 加载模型
sess = ort.InferenceSession("outputs/mlp_sin.onnx")

# 构造输入
# x = np.linspace(0, 2*np.pi, 100).reshape(-1,1).astype(np.float32)
x, y = make_sin_numpy(n_points=1000, x_range=(0, 6*np.pi))
inputs = {"x": x}

# 推理
y_pred = sess.run(["y"], inputs)[0]

# 可视化
plt.plot(x, np.sin(x), label="True sin(x)")
plt.plot(x, y_pred, label="ONNX Predicted", linestyle="--")
plt.legend()
plt.show()
