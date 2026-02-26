# Neuroscience-baseline 仓库总结

## 项目定位

这是一个**生物启发式无监督特征学习基准评估平台**，统一实现和比较了多种基于脉冲神经网络 (SNN)、Hebbian 学习和哈希方法的编码器，用于聚类和检索任务。

## 8 个 Baseline 编码器

| Baseline | 方法 | 训练时间 |
|---|---|---|
| **FlyHash** | 随机投影 + Winner-Take-All | 即时（无需训练） |
| **Diehl & Cook** | STDP + 侧抑制（基于 BindsNET） | ~6 小时（60K 样本, GPU） |
| **SoftHebb** | Hebbian + Soft-WTA | ~2 分钟 |
| **Krotov** | Hebbian + WTA | ~1 分钟 |
| **BioHash** | Hebbian + 稀疏投影 | ~2 分钟 |
| **WTA Hash** | 随机窗口 + 局部 WTA | 即时 |
| **SOM** | 竞争学习（自组织映射） | ~5 分钟 |
| **LSH / SimHash** | 随机超平面 | 即时 |

所有编码器都继承自统一的 `BaseEncoder` 抽象基类 (`baselines/base_encoder.py`)，保证一致的 `fit()` / `encode()` API。

## 支持的数据集

- **MNIST** (784 维) — 聚类 + 检索
- **Fashion-MNIST** (784 维) — 聚类 + 检索
- **SIFT1M** (128 维) — 仅检索

## 评估指标

- **聚类**: ACC（准确率）、NMI（归一化互信息）、ARI（调整兰德指数）
- **检索**: mAP（平均精度）、Recall@K

## 项目结构

```
.
├── baselines/              # 8 个编码器实现（每个一个子目录）
│   ├── base_encoder.py     # 统一抽象基类
│   ├── flyhash/            # FlyHash 编码器
│   ├── diehl_cook/         # Diehl & Cook SNN（含 train.py）
│   ├── softhebb/           # SoftHebb 编码器
│   ├── krotov/             # Krotov 编码器
│   ├── biohash/            # BioHash 编码器
│   ├── wta_hash/           # WTA Hash 编码器
│   ├── som/                # SOM 编码器
│   └── lsh/                # LSH/SimHash 编码器
├── configs/                # 11 个 YAML 配置文件
├── pipeline/               # 评估流水线（数据集、聚类、检索、指标）
├── scripts/                # ~30 个运行/测试/诊断脚本
├── tests/                  # 单元测试
├── docs/                   # ~50 个文档
├── outputs/                # 结果输出
├── run.py                  # 主 CLI 入口
├── Makefile                # 便捷命令
└── requirements.txt        # 依赖
```

## 核心工作流

1. **`run.py`** — 统一 CLI 入口，支持 `--baseline`、`--config`、`--dataset`、`--seed` 等参数
2. **`scripts/run_benchmark.py`** — 批量跑所有 baseline × 所有数据集，输出汇总表
3. **`pipeline/`** — 完整评估流水线：加载数据 → 编码 → 二值化 → 聚类/检索评估 → 输出指标

## 技术栈

- **PyTorch / TorchVision** — 张量计算与数据集
- **BindsNET** — SNN 仿真（Diehl & Cook baseline）
- **scikit-learn** — 聚类、SVM 评估
- **NumPy / SciPy** — 数值计算
