
# dataset
import numpy as np
from PIL import Image

class SimpleImageLoader:
    def __init__(self, txt_file):
        with open(txt_file, 'r') as f:
            lines = f.read().strip().split('\n')
        self.image_pairs = [line.strip().split() for line in lines]
    def __len__(self):
        return len(self.image_pairs)
    def __getitem__(self, idx):
        hazy_path, gt_path = self.image_pairs[idx]
        I_pil = Image.open(hazy_path).convert('RGB')
        J_gt_pil = Image.open(gt_path).convert('RGB')
        I_np = np.array(I_pil).astype(np.float32) / 255.0       # [H,W,3]
        J_gt_np = np.array(J_gt_pil).astype(np.float32) / 255.0 # [H,W,3]
        return I_np, J_gt_np

def SimpleImageSaver(tensor, path):
    if tensor.shape[0] == 1:
        img_array = tensor[0]
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    else:
        img_array = np.transpose(tensor, (1, 2, 0))
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    img.save(path)


# predict
import onnx
import onnxruntime as ort
import numpy as np
import time

def predict():
    onnx_model = onnx.load("lightdehazenet.onnx")
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession("lightdehazenet.onnx", providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    loader = SimpleImageLoader('./test-images/split_txt/test.txt')
    for i in range(len(loader)):
        I_np, J_gt_np = loader[i]
        I_input = np.transpose(I_np, (2, 0, 1))
        I_input = np.stack([I_input, I_input])
        start_time = time.perf_counter()
        outputs = session.run(output_names, {input_name: I_input})
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000.0
        print(f"shape {I_np.shape}, time {execution_time:.2f}ms")
        J_pred = outputs[0]
        if i == 0:
            J_gt = np.transpose(J_gt_np, (2, 0, 1))
            SimpleImageSaver(J_gt, './test-images/result/J_gt.png')
            SimpleImageSaver(I_input[0], './test-images/result/I_gt.png')
            SimpleImageSaver(J_pred[0], './test-images/result/J_pred.png')

if __name__ == '__main__':
    predict()
