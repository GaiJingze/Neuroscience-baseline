# Neuroscience-baseline 仓库详细总结

## 项目概述

这是一个**生物启发式无监督特征学习基线的统一评估平台**。它提供了标准化框架，用于比较多种脉冲神经网络 (SNN) 和 Hebbian 学习方法在聚类和检索任务上的表现。

## 核心架构

**12 种基线编码器**，全部继承自统一的 `BaseEncoder` 抽象基类：

| 基线 | 类型 | 训练耗时 |
|------|------|----------|
| **FlyHash** | 随机投影 + WTA | 瞬时 |
| **Diehl & Cook** | STDP + 侧向抑制 (BindsNET) | ~6h (GPU) |
| **Deep STDP** | 多层 STDP + K-means | ~8h (GPU) |
| **LC-SNN** | 局部竞争 SNN | ~4h (GPU) |
| **LM-SNN** | 侧向调制 SNN | ~6h (GPU) |
| **CSDP** | 对比信号依赖可塑性 | ~3min |
| **SoftHebb** | Hebbian + Soft-WTA (PyTorch) | ~2min |
| **Krotov** | k-WTA + 反 Hebbian 学习 | ~1min |
| **BioHash** | Hebbian + 稀疏投影 | ~2min |
| **WTA Hash** | 随机窗口 + 局部 WTA | 瞬时 |
| **SOM** | 自组织映射 | ~5min |
| **LSH/SimHash** | 随机超平面 | 瞬时 |

每个编码器遵循统一接口：
- `fit(data)` — 无监督训练
- `encode(data)` — 返回 `pre_code`（连续值）和 `code`（二值 {0,1}）
- `save()/load()` — 模型持久化

## 目录结构

```
├── baselines/          # 12 种编码器实现
├── pipeline/           # 评估框架（数据集、指标、聚类、检索）
├── configs/            # 每个基线的 YAML 配置文件（15个）
├── scripts/            # 入口脚本和工具脚本（40+个）
├── tests/              # 测试基础设施
├── docs/               # 详细文档（50+ markdown 文件）
├── outputs/            # 结果存储（git-ignored）
├── run.py              # 主 CLI 入口
├── Makefile            # 构建/测试自动化（30+目标）
└── requirements.txt    # Python 依赖
```

## 评估管道 (`pipeline/`)

- **datasets.py** — 数据加载器：MNIST、Fashion-MNIST（784维）、SIFT1M（128维，100万向量）
- **metrics.py** — 聚类指标（NMI, ARI, ACC, Purity, Silhouette）和检索指标（mAP, Recall@K）
- **clustering.py** — K-means、K-medoids、谱聚类、层次聚类
- **retrieval.py** — 相似性搜索（余弦、欧氏、汉明距离）
- **binarization.py** — 二值化方法（Top-K、阈值、WTA、百分比）
- **supervised_eval.py** — 可选的 SVM 分类评估

## 数据集

| 数据集 | 训练集 | 测试集 | 维度 | 用途 |
|--------|--------|--------|------|------|
| MNIST | 60K | 10K | 784 | 聚类 |
| Fashion-MNIST | 60K | 10K | 784 | 聚类 |
| SIFT1M | 1M | 100K | 128 | 检索 |

## 主要依赖

- **核心**: numpy, scipy, scikit-learn, PyYAML, matplotlib
- **深度学习**: PyTorch ≥1.13, TorchVision ≥0.14
- **可选**: BindsNET（SNN 基线）、FAISS（GPU 加速检索）

## 使用方式

```bash
# 运行单个基线
python run.py --baseline flyhash --dataset mnist --seed 42

# 运行全部基准测试
python scripts/run_benchmark.py --seeds 42 123 456

# 快速测试
make quick-test
```

## 设计特点

- **统一接口**：所有编码器输入 float32 [0,1]，输出标准化的 pre_code + binary code
- **可复现性**：全局种子控制、确定性算法、编码缓存
- **优雅降级**：缺少 BindsNET 时相关编码器标记为不可用，不影响其他基线
- **配置驱动**：YAML 配置 + 命令行参数覆盖
- **丰富文档**：50+ markdown 文件覆盖安装、故障排查、基线细节、指标说明等

总体而言，这是一个面向研究人员的**完整、可复现的生物启发式无监督学习基准测试平台**。
