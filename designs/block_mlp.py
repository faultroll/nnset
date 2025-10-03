# layers/mlp_layer.py
import torch
import torch.nn as nn

class MLP_Layer(nn.Module):
    def __init__(self, in_features, out_features, activation='relu'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'swish':
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()
    def forward(self, x):
        return self.act(self.linear(x))

# blocks/mlp_block.py
# from layers.mlp_layer import MLP_Layer
import torch.nn as nn

class MLP_Block(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features, activation='relu'):
        super().__init__()
        layers = []
        last_features = in_features
        for h in hidden_sizes:
            layers.append(MLP_Layer(last_features, h, activation))
            last_features = h
        layers.append(MLP_Layer(last_features, out_features, 'identity'))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

""" # layer
layer1 = MLP_Layer(in_features=1, out_features=16, activation='relu')
# block
mlp_block = MLP_Block(in_features=1, hidden_sizes=[16,32], out_features=1, activation='swish')
# network (demo里直接用 block)
network = mlp_block """
