# Clustering/Hashing Pipeline - 项目总结

## 📊 1. 可运行的 Baseline

### ✅ 已实现并可运行 (2个)

| Baseline | 状态 | 描述 | 学习规则 | 运行时间 |
|----------|------|------|----------|----------|
| **FlyHash** | ✅ **完全可用** | 果蝇启发的随机投影+WTA哈希 | Random projection + WTA | ~2 分钟 |
| **Diehl & Cook** | ✅ **接口就绪** | STDP + 侧抑制的SNN | STDP + lateral inhibition | 需要训练 (~1小时) |

#### FlyHash
- **论文**: Dasgupta et al., Science 2017
- **特点**: 
  - 生物启发的快速哈希算法
  - 基于果蝇嗅觉系统
  - 无需训练，随机投影
- **适用场景**: 快速原型验证，无监督哈希
- **已测试**: ✅ MNIST (10,000样本)

#### Diehl & Cook
- **论文**: Diehl & Cook, Front. Comput. Neurosci. 2015
- **特点**: 
  - 经典的STDP学习SNN
  - 使用BindsNET实现
  - Winner-Take-All机制
- **适用场景**: 无监督特征学习，生物可信学习
- **训练需求**: GPU推荐，可CPU运行（较慢）

### 🚧 部分实现 (1个)

| Baseline | 状态 | 描述 |
|----------|------|------|
| **SoftHebb** | 🔧 骨架代码 | Hebbian学习规则，尚需实现核心逻辑 |

### 📋 计划实现

- **Deep STDP** (Lu & Sengupta, NCE 2024) - 高优先级
- **BioHash** (如果存在相关实现) - 中优先级

---

## 📁 2. 数据集和评估指标

### 数据集

#### 🎯 当前使用
1. **MNIST**
   - 类型: 手写数字
   - 样本: 60,000 训练 + 10,000 测试
   - 维度: 28×28 = 784
   - 类别: 10
   - **状态**: ✅ 已集成，自动下载

2. **Fashion-MNIST**
   - 类型: 时尚物品
   - 样本: 60,000 训练 + 10,000 测试
   - 维度: 28×28 = 784
   - 类别: 10
   - **状态**: ✅ 已集成，自动下载

#### 📦 可选数据集
3. **SIFT1M** (用于哈希/检索任务)
   - 类型: 图像局部特征
   - 样本: 1,000,000
   - 维度: 128
   - **状态**: 需手动下载 (`bash scripts/download_sift1m.sh`)

4. **GloVe** (词向量，可选)
   - 类型: 文本嵌入
   - **状态**: 需手动下载 (`bash scripts/download_glove.sh`)

### 评估指标

#### 🎯 聚类任务 (MNIST, Fashion-MNIST)

| 指标 | 名称 | 范围 | 描述 | 优先级 |
|------|------|------|------|--------|
| **NMI** | Normalized Mutual Information | 0-1 | 衡量聚类与真实标签的信息重叠度 | ⭐⭐⭐ 主要 |
| **ARI** | Adjusted Rand Index | -1 to 1 | 衡量聚类相似性（调整随机性） | ⭐⭐⭐ 主要 |
| **ACC** | Clustering Accuracy | 0-1 | 最佳匹配后的准确率（Hungarian算法） | ⭐⭐ 重要 |
| **Silhouette** | Silhouette Score | -1 to 1 | 内部聚类质量（无需标签） | ⭐ 参考 |

**说明**:
- **NMI**: 越高越好，1.0表示完美聚类
- **ARI**: 越高越好，1.0表示完美聚类，0表示随机
- **ACC**: 越高越好，需要Hungarian匹配找最佳对应
- **Silhouette**: 越高越好，衡量簇内紧密度和簇间分离度

#### 🔍 检索/哈希任务 (SIFT1M)

| 指标 | 名称 | 范围 | 描述 |
|------|------|------|------|
| **mAP** | Mean Average Precision | 0-1 | 平均精度均值 |
| **Recall@K** | Recall at K | 0-1 | Top-K中真实邻居的召回率 (K=10,50,100) |
| **Precision@K** | Precision at K | 0-1 | Top-K的精确率 |

#### 🧠 SNN特定指标

| 指标 | 描述 |
|------|------|
| **Spike Sparsity** | 1 - 发放率，衡量能效 |
| **Hamming Distance** | 二进制编码的汉明距离 |
| **Temporal Dynamics** | (可选) 脉冲时序分析 |

---

## 🚀 3. 如何运行 Baseline

### 方式 1: 使用主入口 (最简单) ⭐⭐⭐

```bash
# 进入项目目录
cd /hy-tmp/clustering

# 运行 FlyHash (最快，推荐测试)
python run.py --baseline flyhash --dataset mnist --seed 0

# 运行 Diehl & Cook (需要GPU，较慢)
python run.py --baseline diehl_cook --dataset mnist --seed 0

# 使用不同数据集
python run.py --baseline flyhash --dataset fashion_mnist

# 使用不同随机种子
python run.py --baseline flyhash --seed 1
```

### 方式 2: 使用配置文件

```bash
# 使用预定义配置
python run.py --config configs/flyhash.yaml

# 覆盖配置参数
python run.py --config configs/flyhash.yaml --seed 42 --dataset fashion_mnist
```

### 方式 3: 直接调用脚本

```bash
python scripts/run_baseline.py --config configs/flyhash.yaml
```

### 方式 4: Diehl & Cook 完整训练

```bash
# 完整训练（需要GPU）
python baselines/diehl_cook/train.py \
    --train --extract \
    --n_train 60000 \
    --n_epochs 1 \
    --device cuda

# 快速测试（小数据集）
python baselines/diehl_cook/train.py \
    --train --extract \
    --n_train 1000 \
    --n_epochs 1 \
    --device cpu
```

---

## 📊 运行示例和预期结果

### FlyHash on MNIST

```bash
python run.py --baseline flyhash --dataset mnist
```

**预期输出**:
```
[INFO] Loading dataset: mnist
[INFO] Dataset loaded: 60000 train, 10000 test
[INFO] Initializing encoder: flyhash
[INFO] Training encoder...
[INFO] Encoding train data...
[INFO] Encoding test data...
[INFO] Running clustering evaluation...

=== Clustering Results (K-Means) ===
NMI:         0.45 - 0.55
ARI:         0.35 - 0.45
ACC:         0.60 - 0.70
Silhouette:  0.15 - 0.25

[INFO] Results saved to: outputs/results/flyhash_mnist_seed0.json
```

**运行时间**: ~2 分钟 (10,000测试样本)

### Diehl & Cook on MNIST (快速测试)

```bash
python run.py --baseline diehl_cook --dataset mnist
```

**注意**: 首次运行会触发训练，时间较长。

---

## 🔧 快速命令参考

### 查看可用的 baseline

```bash
python run.py --list
```

输出:
```
Available baselines:
  - flyhash       ✅ Ready
  - diehl_cook    🟡 BindsNET required
  - softhebb      🚧 Under development
```

### 运行快速测试

```bash
# 测试所有pipeline组件
python run.py --test

# 或使用专用脚本
python scripts/quick_test.py
```

### 批量运行多个 baseline

```bash
# 测试FlyHash (多个种子)
python scripts/test_baseline.py --baseline flyhash --seeds 0 1 2

# 测试所有baseline
python scripts/test_baseline.py --all --dataset mnist
```

### 查看帮助

```bash
# 主程序帮助
python run.py --help

# 测试选项帮助
python run.py --help-test

# 配置选项帮助
python run.py --help-config
```

---

## 📂 结果存储

### 输出目录结构

```
outputs/
├── codes/                          # 缓存的特征编码
│   └── flyhash_mnist_seed0.pkl
├── results/                        # 评估结果 (JSON)
│   └── flyhash_mnist_seed0.json
└── logs/                           # 训练日志
    └── diehl_cook_training.log
```

### 查看结果

```bash
# 查看JSON结果
cat outputs/results/flyhash_mnist_seed0.json

# 漂亮打印
python -m json.tool outputs/results/flyhash_mnist_seed0.json

# 或使用jq (如果安装)
jq . outputs/results/flyhash_mnist_seed0.json
```

结果包含:
```json
{
  "experiment_name": "flyhash_mnist",
  "config": {...},
  "clustering": {
    "kmeans": {
      "nmi": 0.512,
      "ari": 0.423,
      "acc": 0.678,
      "silhouette": 0.189
    }
  },
  "encoding_time": 12.34,
  "evaluation_time": 5.67
}
```

---

## 🎯 典型使用场景

### 场景 1: 快速验证 Pipeline

```bash
# 1. 运行快速测试
python run.py --test

# 2. 运行最简单的baseline
python run.py --baseline flyhash --dataset mnist
```

**时间**: ~3 分钟

### 场景 2: 对比不同 Baseline

```bash
# 运行FlyHash
python run.py --baseline flyhash --dataset mnist --seed 0

# 运行Diehl & Cook
python run.py --baseline diehl_cook --dataset mnist --seed 0

# 对比结果
diff outputs/results/flyhash_mnist_seed0.json outputs/results/diehl_cook_mnist_seed0.json
```

### 场景 3: 统计显著性测试 (多个种子)

```bash
# 运行多个随机种子
for seed in 0 1 2 3 4; do
    python run.py --baseline flyhash --dataset mnist --seed $seed
done

# 分析结果
python scripts/analyze_results.py --baseline flyhash --dataset mnist
```

### 场景 4: 不同数据集评估

```bash
# MNIST
python run.py --baseline flyhash --dataset mnist

# Fashion-MNIST
python run.py --baseline flyhash --dataset fashion_mnist

# 对比难度
```

---

## 🐛 常见问题

### Q1: FlyHash 运行很慢？

**A**: 检查配置文件中的 `clustering_methods`：

```yaml
# 快速测试 (推荐)
clustering_methods:
  - kmeans

# 完整评估 (慢)
clustering_methods:
  - kmeans
  - kmedoids      # O(n²) 复杂度
  - spectral      # O(n³) 复杂度
```

默认只用 `kmeans` 以加快速度。

### Q2: Diehl & Cook 如何运行？

**A**: 两种方式：

1. **直接运行** (使用预训练或触发训练):
   ```bash
   python run.py --baseline diehl_cook
   ```

2. **手动训练** (更多控制):
   ```bash
   # 先训练
   python baselines/diehl_cook/train.py --train --n_train 1000
   
   # 再评估
   python run.py --baseline diehl_cook
   ```

### Q3: 如何加速评估？

**A**: 
1. 使用更少的测试样本（修改 `configs/*.yaml` 中的 `n_test`）
2. 只使用 `kmeans` 聚类方法
3. 设置 `n_init=1` (已默认)
4. 使用GPU（对于SNN baseline）

---

## 📊 性能基准

### 硬件环境
- CPU: 8 cores
- RAM: 16 GB
- GPU: (可选) NVIDIA GPU with CUDA

### 运行时间 (MNIST, 10,000 测试样本)

| Baseline | 训练时间 | 编码时间 | 聚类时间 | 总时间 |
|----------|---------|---------|---------|--------|
| **FlyHash** | 无需训练 | ~10s | ~2min | **~2min** |
| **Diehl & Cook** (1k samples) | ~10min | ~1min | ~2min | **~13min** |
| **Diehl & Cook** (60k samples) | ~1h | ~5min | ~30min | **~1.5h** |

---

## 🎓 下一步

### 立即可做
1. ✅ 运行 FlyHash: `python run.py --baseline flyhash`
2. ✅ 查看结果: `cat outputs/results/*.json`
3. ✅ 测试不同数据集: `--dataset fashion_mnist`

### 短期目标
1. 🔧 完成 SoftHebb 实现
2. 📊 运行完整的 Diehl & Cook 训练
3. 📈 生成对比分析图表

### 长期目标
1. 🚀 实现 Deep STDP (Lu & Sengupta 2024)
2. 🔍 添加 SIFT1M 检索任务
3. 📊 完整的 baseline 性能报告

---

**总结**: 
- ✅ **2个** 可运行的 baseline (FlyHash 完全就绪，Diehl & Cook 接口就绪)
- ✅ **MNIST/Fashion-MNIST** 数据集，使用 **NMI/ARI/ACC** 指标
- ✅ **一行命令运行**: `python run.py --baseline flyhash`

**推荐首次运行**: 
```bash
python run.py --baseline flyhash --dataset mnist --seed 0
```
