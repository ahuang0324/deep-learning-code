PyTorch
├── 核心组件
│   ├── torch
│   │   ├── 张量操作
│   │   │   ├── tensor() - 创建张量
│   │   │   ├── ones(), zeros(), eye() - 特殊张量
│   │   │   ├── rand(), randn() - 随机张量
│   │   │   ├── arange(), linspace() - 序列张量
│   │   │   └── from_numpy() - 从NumPy转换
│   │   ├── 数学运算
│   │   │   ├── add(), sub(), mul(), div() - 基本运算
│   │   │   ├── mm(), bmm(), matmul() - 矩阵乘法
│   │   │   ├── pow(), sqrt(), exp(), log() - 幂指数对数
│   │   │   └── mean(), sum(), max(), min() - 统计函数
│   │   ├── 操作符
│   │   │   ├── cat(), stack() - 拼接/堆叠
│   │   │   ├── split(), chunk() - 分割
│   │   │   ├── reshape(), view() - 形状变换
│   │   │   └── permute(), transpose() - 维度转置
│   │   └── 设备管理
│   │       ├── cuda - CUDA支持
│   │       ├── device() - 设备指定
│   │       ├── to() - 设备迁移
│   │       └── cpu() / cuda() - 设备切换
│   │
│   └── Tensor类
│       ├── 属性
│       │   ├── shape, size() - 形状
│       │   ├── dtype - 数据类型
│       │   ├── device - 所在设备
│       │   └── requires_grad - 是否需要梯度
│       ├── 操作方法
│       │   ├── backward() - 反向传播
│       │   ├── detach() - 分离计算图
│       │   ├── item() - 获取标量值
│       │   └── numpy() - 转为NumPy数组
│       └── 索引和切片
│           ├── tensor[idx] - 基本索引
│           ├── tensor.index_select() - 索引选择
│           └── torch.masked_select() - 掩码选择
│
├── 神经网络 (torch.nn)
│   ├── 层 (Layers)
│   │   ├── 基础层
│   │   │   ├── Linear - 全连接层
│   │   │   ├── Conv1d/2d/3d - 卷积层
│   │   │   ├── MaxPool1d/2d/3d - 最大池化
│   │   │   ├── AvgPool1d/2d/3d - 平均池化
│   │   │   ├── BatchNorm1d/2d/3d - 批归一化
│   │   │   ├── Dropout - 随机失活
│   │   │   └── Embedding - 嵌入层
│   │   ├── 循环层
│   │   │   ├── RNN - 简单循环网络
│   │   │   ├── LSTM - 长短期记忆网络
│   │   │   └── GRU - 门控循环单元
│   │   └── Transformer相关
│   │       ├── TransformerEncoder - 编码器
│   │       ├── TransformerDecoder - 解码器
│   │       ├── MultiheadAttention - 多头注意力
│   │       └── TransformerEncoderLayer - 编码器层
│   │
│   ├── 激活函数 (torch.nn.functional)
│   │   ├── relu, leaky_relu - ReLU变体
│   │   ├── sigmoid, tanh - Sigmoid和Tanh
│   │   ├── softmax, log_softmax - Softmax变体
│   │   └── gelu, selu, elu - 其他激活函数
│   │
│   ├── 损失函数 (Loss Functions)
│   │   ├── MSELoss - 均方误差
│   │   ├── CrossEntropyLoss - 交叉熵
│   │   ├── BCELoss - 二元交叉熵
│   │   ├── NLLLoss - 负对数似然
│   │   └── L1Loss - 平均绝对误差
│   │
│   └── Container
│       ├── Module - 基类
│       ├── Sequential - 序列容器
│       ├── ModuleList - 模块列表
│       └── ModuleDict - 模块字典
│
├── 优化器 (torch.optim)
│   ├── SGD - 随机梯度下降
│   ├── Adam, AdamW - Adam优化器变体
│   ├── RMSprop - 均方根传播
│   ├── Adagrad - 自适应梯度
│   └── lr_scheduler - 学习率调度器
│       ├── StepLR - 步进式调度
│       ├── CosineAnnealingLR - 余弦退火
│       ├── ReduceLROnPlateau - 根据指标调整
│       └── OneCycleLR - 单周期策略
│
├── 数据处理 (torch.utils.data)
│   ├── Dataset - 数据集基类
│   ├── DataLoader - 数据加载器
│   ├── TensorDataset - 张量数据集
│   ├── random_split - 随机分割
│   └── Sampler - 采样器
│       ├── RandomSampler - 随机采样
│       ├── SequentialSampler - 顺序采样
│       └── WeightedRandomSampler - 带权重采样
│
├── 自动微分 (torch.autograd)
│   ├── Variable - 变量(现已合并入Tensor)
│   ├── grad - 梯度
│   ├── backward() - 反向传播
│   ├── no_grad() - 禁用梯度上下文
│   └── Function - 自定义自动微分函数
│
└── 工具与扩展
    ├── JIT与TorchScript
    │   ├── trace - 跟踪
    │   ├── script - 脚本化
    │   └── save/load - 保存加载
    ├── 分布式 (torch.distributed)
    │   ├── init_process_group - 初始化
    │   ├── all_reduce - 归约
    │   └── DistributedDataParallel - 分布式包装器
    ├── torch.hub - 模型仓库
    ├── torch.onnx - ONNX导出
    └── torchvision - 视觉库
        ├── models - 预训练模型
        ├── transforms - 图像变换
        └── datasets - 视觉数据集