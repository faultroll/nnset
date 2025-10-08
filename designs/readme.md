# nnset.designs — Network Design Library Summary


## 分类

| 功能类别           | 典型目标 / 场景                  | 示例 Block                                                  | 特点                         |
| -------------- | -------------------------- | --------------------------------------------------------- | -------------------------- |
| 特征提取       | 提取输入信号的高阶表示，适用 1D/2D/ND 信号 | ConvBlock, ResidualBlock, MLPBlock                        | 保留空间/时间结构，捕捉局部或全局特征        |
| 函数拟合 / 回归  | 对连续信号或特征进行映射，拟合周期或非周期函数    | MLP, Fourier Features, SIREN                              | 可以处理非线性关系，可用于周期函数拟合、激活函数学习 |
| 注意力 / 加权聚合 | 动态选择重要特征或通道                | Self-Attention, Squeeze-and-Excitation, Transformer Block | 增强长程依赖建模能力，可视化权重解释网络决策     |
| 生成 / 对抗    | 学习生成分布或映射，实现 GAN、图像生成等     | GAN Generator / Discriminator Block, UNet Block           | 强调对抗训练、特征重构能力，通常和残差或注意力结合  |
| 下采样 / 上采样  | 调整特征图分辨率                   | Pooling Block, Transposed Conv Block                      | 控制特征尺度，影响感受野和分辨率           |
| 归一化 / 激活   | 稳定训练、增加非线性                 | BatchNorm, LayerNorm, ReLU, GELU                          | 训练稳定性和表达能力的基础组件            |


## 模块层级划分

```
designs/
├── ops/ # Operation层 纯数学操作
├── layers/ # Layer层 添加归一化与激活
└── blocks/ # Block层 组合多层形成功能单元
```
