""" 
## Sequence Modeling Blocks
### 1. RNN / LSTM / GRU
- **来源**: Hochreiter & Schmidhuber, 1997
- **功能**: 捕获时间序列依赖关系
- **优点**: 结构简单、适合中短序列
- **缺点**: 无法并行、长序列梯度消失
- **PyTorch实现**: `nn.RNN`, `nn.LSTM`, `nn.GRU`
### 2. Attention
- **来源**: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- **功能**: 基于相似度权重建模全局依赖
- **优点**: 并行、高效建模长程依赖
- **缺点**: O(n²) 复杂度，高内存消耗
- **PyTorch实现**: `nn.MultiheadAttention`
### 3. Transformer
- **功能**: 编码器-解码器架构，结合多头注意力与前馈层
- **优点**: 可扩展性强、性能优越
- **缺点**: 参数量大，对小数据集易过拟合
- **PyTorch实现**: `nn.Transformer`, `nn.TransformerEncoder`
"""

# nnset/designs/blocks/sequence_blocks.py
import torch
import torch.nn as nn

# 抽象序列建模模块：支持RNN/LSTM/GRU/Transformer
class SequenceBlock(nn.Module):
    def __init__(self, model_type='lstm', input_dim=128, hidden_dim=256, nhead=4, num_layers=2):
        super().__init__()
        self.model_type = model_type.lower()

        if self.model_type == 'rnn':
            self.model = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        elif self.model_type == 'lstm':
            self.model = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        elif self.model_type == 'gru':
            self.model = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        elif self.model_type == 'attention':
            self.model = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        elif self.model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
            self.model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(self, x):
        if self.model_type in ['rnn', 'lstm', 'gru']:
            out, _ = self.model(x)
        elif self.model_type == 'attention':
            out, _ = self.model(x, x, x)
        elif self.model_type == 'transformer':
            out = self.model(x)
        return out
