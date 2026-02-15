# 仓库内容与 Baseline 架构总结

## 一、仓库概述

本仓库 **Neuroscience-baseline** 实现了一套用于**无监督特征学习**的基准评测流水线（Pipeline），核心目标是使用脉冲神经网络（SNN）和生物可信的学习规则（STDP、Hebbian 等），在聚类和局部敏感哈希（LSH）两大任务上建立基线性能。

### 两大核心任务

| 任务 | 描述 | 数据集 | 评估指标 |
|------|------|--------|----------|
| **Task A** — 无监督特征学习与聚类 | 提取无监督特征后进行聚类 | MNIST, Fashion-MNIST | ACC（匈牙利匹配聚类精度）、NMI（归一化互信息） |
| **Task B** — 局部敏感哈希 / 近似近邻检索 | 将样本编码为二值码后做 ANN 检索 | SIFT1M, GloVe | mAP（平均精度均值）、Recall@K（K=10,50,100） |

### 仓库结构

```
Neuroscience-baseline/
├── run.py                     # 统一入口
├── baselines/                 # 8 种 Baseline 实现
│   ├── base_encoder.py        # 抽象基类 BaseEncoder
│   ├── flyhash/               # FlyHash
│   ├── diehl_cook/            # Diehl & Cook STDP-WTA
│   ├── softhebb/              # SoftHebb
│   ├── krotov/                # Krotov-Hopfield
│   ├── biohash/               # BioHash
│   ├── wta_hash/              # WTA Hash
│   ├── som/                   # Self-Organizing Map
│   └── lsh/                   # LSH / SimHash
├── pipeline/                  # 评测核心模块
│   ├── datasets.py            # 数据加载（MNIST / SIFT1M / GloVe）
│   ├── binarization.py        # 二值化（WTA / top-k / 阈值）
│   ├── clustering.py          # 聚类算法（KMeans / KMedoids / Spectral）
│   ├── retrieval.py           # 检索评测（FAISS）
│   ├── metrics.py             # NMI / ARI / ACC / mAP / Recall@K
│   ├── supervised_eval.py     # 有监督评估（SVM）
│   └── utils.py               # 工具函数
├── configs/                   # 每个 Baseline 的 YAML 配置
├── scripts/                   # 运行 / 测试脚本
├── docs/                      # 文档
└── tests/                     # 测试
```

### 统一接口设计

所有 Encoder 均继承自 `baselines/base_encoder.py` 中的 `BaseEncoder`，提供统一接口：

```python
class BaseEncoder(ABC):
    def fit(self, train_data, train_labels=None)   # 训练
    def encode(self, data) -> dict                  # 编码，返回 {'pre_code': 连续特征, 'code': 二值码}
    def save(self, path) / load(self, path)         # 模型持久化
```

---

## 二、各 Baseline 来源与架构

### 1. FlyHash — 果蝇嗅觉回路启发的哈希

| 项目 | 内容 |
|------|------|
| **来源论文** | Dasgupta et al., *"A neural algorithm for a fundamental computing problem"*, **Science**, 2017 |
| **生物启发** | 模拟果蝇嗅觉系统的 Kenyon 细胞层：先做稀疏随机投影扩展维度，再用 WTA 竞争生成稀疏码 |
| **架构** | 输入 → 稀疏随机投影矩阵（维度扩展约 2.5 倍，每个输出神经元只连接一部分输入）→ Winner-Take-All（保留 top 5% 激活） |
| **关键参数** | `projection_dim=2000`, `hash_length=100`, `sampling_ratio=0.1` |
| **是否需要训练** | **不需要**，权重为随机初始化后固定 |
| **核心特点** | 非参数化、即时编码、实现简单，可作为最基本的生物可信哈希基线 |

---

### 2. Diehl & Cook STDP-WTA — 脉冲时序依赖可塑性网络

| 项目 | 内容 |
|------|------|
| **来源论文** | Diehl & Cook, *"Unsupervised learning of digit recognition using spike-timing-dependent plasticity"*, **Frontiers in Computational Neuroscience**, 2015 |
| **依赖框架** | BindsNET（PyTorch 上的 SNN 模拟框架） |
| **架构** | 三层脉冲网络：① **输入层**：Poisson 编码的速率神经元 → ② **兴奋层**：400 个 LIF 神经元，接收 STDP 突触 → ③ **抑制层**：侧向抑制（全连接除对角线），实现 WTA 竞争 |
| **连接方式** | Input→Exc（STDP 学习）、Exc→Inh（一对一）、Inh→Exc（侧向抑制） |
| **学习规则** | **STDP (PostPre)**：突触前后脉冲时序差控制突触权重更新 |
| **关键机制** | 自适应阈值 θ+（鼓励不同神经元特化学习不同模式）、脉冲监控器记录兴奋层发放次数 |
| **特征提取** | 兴奋层脉冲计数 → top-k 二值化 |
| **关键参数** | `n_neurons=400`, `simulation_time=350ms`, `nu=[1e-4, 1e-2]` |
| **训练时间** | 约 6 小时（60K 样本），推荐使用 GPU |

---

### 3. SoftHebb — 软 WTA Hebbian 网络

| 项目 | 内容 |
|------|------|
| **来源论文** | Kozachkov et al., *"SoftHebb: Bayesian inference in unsupervised Hebbian soft winner-take-all networks"*, **Neural Computation and Engineering (2022)** / **ICLR (2023)** |
| **架构** | 多层前馈网络（PyTorch 实现）：`784 → 1000 → 500 → 400`，每层依次包含线性变换 → ReLU → Soft-WTA 选择 |
| **学习规则** | **Hebbian 学习**：ΔW = η · y · xᵀ（突触前后同时激活则增强连接），附加权重归一化防止无界增长 |
| **核心机制** | Soft Winner-Take-All：在每层选取 top-k 个最强激活的神经元，其余抑制。不同于硬 WTA，保留了梯度信号的连续性 |
| **关键参数** | `hidden_dims=[1000, 500]`, `output_dim=400`, `k_values=[50, 20]`, `eta=0.01`, `n_epochs=10` |
| **训练时间** | 约 2 分钟 |

---

### 4. Krotov-Hopfield — 竞争隐藏单元的无监督学习

| 项目 | 内容 |
|------|------|
| **来源论文** | Krotov & Hopfield, *"Unsupervised Learning by Competing Hidden Units"*, **PNAS**, 2019 |
| **架构** | 单层网络：输入 → n_neurons 个隐藏单元，使用幂律权重激活 |
| **激活函数** | Q = sign(W) · \|W\|^(p-1) · X（幂律加权，p=2 为默认） |
| **学习规则** | **Hebbian + Anti-Hebbian 竞争**：① k-WTA 选择：排名第 1 的 winner 标记为 +1，排名前 k 的标记为 -δ ② 权重更新 ΔW 同时包含 Hebbian 项（winner 强化）和 Anti-Hebbian 项（竞争抑制） |
| **关键参数** | `n_neurons=400`, `delta=0.4`（Anti-Hebbian 强度）, `p=2.0`（Lebesgue 范数）, `k=2`, `lr=0.02`, `n_epochs=200` |
| **训练时间** | 约 1 分钟 |
| **核心特点** | 竞争机制确保不同隐藏单元学习到不同特征，Anti-Hebbian 项防止所有神经元收敛到同一模式 |

---

### 5. BioHash — 生物启发的可学习哈希

| 项目 | 内容 |
|------|------|
| **来源** | 受 FlyHash（果蝇回路）启发的扩展方法，加入学习能力 |
| **架构** | 稀疏随机初始化连接矩阵（仅 10% 的连接被激活）→ Hebbian 学习更新权重 → WTA 二值化 |
| **与 FlyHash 的区别** | FlyHash 权重完全随机固定；BioHash 在随机初始化基础上通过 Hebbian 学习进行数据自适应调整 |
| **学习规则** | ΔW = η · xᵀ · y，并做列归一化（L2）。连接掩码在训练期间保持不变（只更新已存在的连接） |
| **关键参数** | `hash_dim=256`, `sparse_ratio=0.1`, `k_winners=13`, `n_epochs=5`, `lr=0.01` |
| **训练时间** | 约 2 分钟 |

---

### 6. WTA Hash — 基于局部竞争的哈希

| 项目 | 内容 |
|------|------|
| **来源** | 灵感来自 Yagnik et al., *"The Power of Comparative Reasoning"*, **ICCV**, 2011 |
| **生物启发** | 模拟局部侧向抑制机制 |
| **架构** | 将输入特征随机分成 n_hashes 组 "窗口"（每窗口 window_size 个特征）→ 在每个窗口内做局部 WTA（仅最大值保留，生成 one-hot） |
| **输出结构** | 每个窗口输出一个 one-hot 向量，总体稀疏度为 1/window_size |
| **关键参数** | `n_hashes=64`, `window_size=8`, `output_dim=512` |
| **是否需要训练** | **不需要**，仅随机划分窗口 |

---

### 7. SOM (Self-Organizing Map) — 自组织映射

| 项目 | 内容 |
|------|------|
| **来源论文** | Kohonen, *"Self-organized formation of topologically correct feature maps"*, **Biological Cybernetics**, 1982 |
| **架构** | 2D 网格（默认 20×20 = 400 个神经元），每个神经元有与输入同维度的权重向量 |
| **学习算法** | ① 对每个输入找到 Best Matching Unit (BMU)，即权重最接近的神经元 ② 更新 BMU 及其邻域神经元（高斯邻域函数）③ 学习率和邻域半径随 epoch 指数衰减 |
| **编码方式** | 输入与所有神经元的距离向量 → top-k 二值化 |
| **关键参数** | `map_height=20`, `map_width=20`, `n_epochs=10`, `lr_init=0.5`, `sigma_init=auto` |
| **训练时间** | 约 5 分钟 |
| **核心特点** | 保留输入数据的拓扑结构，相似输入映射到网格上相邻位置 |

---

### 8. LSH / SimHash — 随机超平面哈希

| 项目 | 内容 |
|------|------|
| **来源论文** | Charikar, *"Similarity estimation techniques from rounding algorithms"*, **STOC**, 2002 |
| **架构** | hash(x) = sign(R · x)，其中 R 为高斯随机矩阵 (N(0,1))，每行 L2 归一化 |
| **理论保证** | Pr[h(x) = h(y)] = 1 - angle(x, y) / π，即哈希碰撞概率正比于余弦相似度 |
| **输出** | {0, 1}^hash_dim 的二值码 |
| **关键参数** | `hash_dim=128` |
| **是否需要训练** | **不需要** |
| **角色** | 经典非学习哈希基线，用于与生物可信方法进行性能对比 |

---

## 三、Baseline 对比总览

| Baseline | 年份 | 生物可信度 | 学习规则 | 是否需训练 | 参数量级 | 核心创新 |
|----------|------|-----------|---------|-----------|---------|---------|
| FlyHash | 2017 | ★★★★ 果蝇嗅觉回路 | 无（随机投影） | 否 | 极少 | 稀疏随机扩展 + WTA |
| Diehl & Cook | 2015 | ★★★★★ 脉冲网络 | STDP | 是（慢） | 中等 | 脉冲时序 + 侧向抑制 |
| SoftHebb | 2022 | ★★★★ Hebbian | Hebbian + Soft-WTA | 是 | 较多 | 概率化软竞争 |
| Krotov | 2019 | ★★★★ Hebbian | Hebbian + Anti-Hebbian | 是 | 中等 | 竞争隐藏单元 |
| BioHash | 2020 | ★★★ 果蝇启发 | Hebbian | 是 | 中等 | 稀疏连接 + 学习 |
| WTA Hash | 2011 | ★★ 局部抑制 | 无 | 否 | 极少 | 窗口内局部 WTA |
| SOM | 1982 | ★★★ 竞争学习 | 竞争 + 邻域 | 是 | 中等 | 拓扑保持 |
| LSH | 2002 | ★ 无 | 无 | 否 | 极少 | 随机超平面（对照基线） |

---

## 四、评测流水线

```
输入数据 (MNIST/SIFT1M/GloVe)
    │
    ▼
Encoder.fit()          ← 训练阶段（部分方法无需训练）
    │
    ▼
Encoder.encode()       ← 编码：输出 pre_code（连续）+ code（二值）
    │
    ├──→ 聚类评测       ← KMeans / KMedoids / Spectral → ACC, NMI
    │
    └──→ 检索评测       ← FAISS ANN → mAP, Recall@K
```

---

## 五、关键参考文献

1. Dasgupta, S., Stevens, C. F., & Bhatt, S. (2017). A neural algorithm for a fundamental computing problem. *Science*, 358(6364), 793-796.
2. Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition using STDP. *Frontiers in Computational Neuroscience*, 9, 99.
3. Kozachkov, L., et al. (2022). SoftHebb: Bayesian inference in unsupervised Hebbian soft winner-take-all networks. *Neural Computation and Engineering*.
4. Krotov, D., & Hopfield, J. J. (2019). Unsupervised learning by competing hidden units. *PNAS*, 116(16), 7723-7731.
5. Charikar, M. (2002). Similarity estimation techniques from rounding algorithms. *STOC*.
6. Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. *Biological Cybernetics*, 43(1), 59-69.
7. Yagnik, J., et al. (2011). The power of comparative reasoning. *ICCV*.
8. Hazan, H., et al. (2018). BindsNET: A machine learning-oriented spiking neural networks library in Python. *Frontiers in Neuroinformatics*.
