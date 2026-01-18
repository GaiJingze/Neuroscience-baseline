# Diehl & Cook 训练和评估完整指南

## 📁 训练结果保存位置

### 完整训练命令

```bash
cd /hy-tmp/clustering/baselines/diehl_cook
python train.py --train --extract \
    --n_train 60000 \
    --n_epochs 1 \
    --device cuda
```

### 保存的文件

```
baselines/diehl_cook/
├── saved_models/                    # 模型权重
│   ├── diehl_cook_network.pt       # ⭐ 训练好的网络
│   └── diehl_cook_weights.npy      # 权重（NumPy格式）
│
└── outputs/                         # 特征和结果
    ├── train_features.npy          # 训练集特征
    ├── test_features.npy           # ⭐ 测试集特征
    ├── train_labels.npy            # 训练集标签
    └── test_labels.npy             # 测试集标签
```

**或者（根据 `--output_dir` 参数）**:

```
../../outputs/diehl_cook/            # 如果指定了输出目录
├── diehl_cook_network.pt
├── train_features.npy
└── test_features.npy
```

---

## 🎯 如何评估训练结果

训练完成后，有**三种评估方法**：

---

## 方法 1: 使用主 Pipeline（推荐）⭐

### 步骤 1: 将特征复制到标准位置

```bash
# 从训练脚本的输出目录复制特征
cd /hy-tmp/clustering

# 查找保存的特征文件
find baselines/diehl_cook -name "test_features.npy"
find baselines/diehl_cook -name "train_features.npy"

# 创建标准的 codes 文件
python -c "
import pickle
import numpy as np
from pathlib import Path

# 加载训练脚本保存的特征
train_features = np.load('baselines/diehl_cook/outputs/train_features.npy')
test_features = np.load('baselines/diehl_cook/outputs/test_features.npy')

# 保存为 pipeline 期望的格式
output_dir = Path('outputs/codes')
output_dir.mkdir(parents=True, exist_ok=True)

codes_data = {
    'train_codes': train_features,
    'test_codes': test_features
}

with open(output_dir / 'diehl_cook_mnist_seed0.pkl', 'wb') as f:
    pickle.dump(codes_data, f)

print('✅ Features converted and saved to outputs/codes/diehl_cook_mnist_seed0.pkl')
"
```

### 步骤 2: 运行聚类评估

```bash
# 使用主 pipeline 运行聚类评估
python run.py --baseline diehl_cook --dataset mnist --force

# --force: 强制使用已有的特征文件
```

### 步骤 3: 查看结果

```bash
cat outputs/results/diehl_cook_mnist_seed0.json
```

---

## 方法 2: 手动评估（更灵活）

### Python 脚本评估

创建评估脚本 `eval_diehl_cook.py`:

```python
#!/usr/bin/env python
"""评估训练好的 Diehl & Cook 特征"""

import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/hy-tmp/clustering')

from pipeline.datasets import load_dataset
from pipeline.clustering import run_clustering_evaluation
from pipeline.supervised_eval import run_supervised_evaluation

# 1. 加载训练脚本保存的特征
print("Loading features...")
train_features = np.load('baselines/diehl_cook/outputs/train_features.npy')
test_features = np.load('baselines/diehl_cook/outputs/test_features.npy')

print(f"Train features: {train_features.shape}")
print(f"Test features: {test_features.shape}")

# 2. 加载标签
dataset = load_dataset('mnist', root='./data')
train_labels = dataset['train_labels']
test_labels = dataset['test_labels']

# 3. 无监督聚类评估
print("\n" + "="*80)
print("UNSUPERVISED CLUSTERING EVALUATION")
print("="*80)

clustering_results = run_clustering_evaluation(
    codes=test_features,
    labels=test_labels,
    n_clusters=10,
    methods=['kmeans']  # 使用 K-Means
)

for method, metrics in clustering_results.items():
    print(f"\n{method}:")
    print(f"  NMI: {metrics['nmi']:.4f}")
    print(f"  ARI: {metrics['ari']:.4f}")
    print(f"  ACC: {metrics['acc']:.4f}")

# 4. 有监督 SVM 评估
print("\n" + "="*80)
print("SUPERVISED CLASSIFICATION EVALUATION")
print("="*80)

supervised_results = run_supervised_evaluation(
    train_features, train_labels,
    test_features, test_labels,
    methods=['linear_svm']
)

for method, metrics in supervised_results.items():
    print(f"\n{method}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1: {metrics['f1']:.4f}")

# 5. 对比原文
print("\n" + "="*80)
print("COMPARISON WITH DIEHL & COOK (2015)")
print("="*80)
paper_accuracy = 0.95
our_accuracy = supervised_results['linear_svm']['accuracy']
print(f"Paper (full STDP): ~{paper_accuracy:.2%}")
print(f"Ours:              {our_accuracy:.2%}")
if our_accuracy >= paper_accuracy * 0.9:
    print("✅ Excellent! Very close to paper results.")
elif our_accuracy >= paper_accuracy * 0.8:
    print("✅ Good! Within reasonable range.")
else:
    print("⚠️  Lower than expected. May need more training.")

print("="*80)
```

### 运行评估

```bash
cd /hy-tmp/clustering
python eval_diehl_cook.py
```

---

## 方法 3: 直接使用 SVM 评估脚本

### 准备特征文件

```bash
cd /hy-tmp/clustering

# 转换为标准格式
python -c "
import pickle
import numpy as np
from pathlib import Path

train_features = np.load('baselines/diehl_cook/outputs/train_features.npy')
test_features = np.load('baselines/diehl_cook/outputs/test_features.npy')

output_dir = Path('outputs/codes')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'diehl_cook_mnist_seed0.pkl', 'wb') as f:
    pickle.dump({
        'train_codes': train_features,
        'test_codes': test_features
    }, f)

print('Done!')
"
```

### 运行 SVM 评估

```bash
python scripts/run_supervised_eval.py \
    --baseline diehl_cook \
    --dataset mnist \
    --seed 0
```

---

## 📊 完整评估流程（推荐）

### 一键脚本

创建 `eval_trained_diehl_cook.sh`:

```bash
#!/bin/bash
# 完整评估流程

cd /hy-tmp/clustering

echo "=================================================="
echo "  Evaluating Trained Diehl & Cook Network"
echo "=================================================="

# 1. 检查特征文件是否存在
if [ ! -f "baselines/diehl_cook/outputs/test_features.npy" ]; then
    echo "❌ Error: test_features.npy not found!"
    echo "Please run training first:"
    echo "  cd baselines/diehl_cook"
    echo "  python train.py --train --extract --n_train 60000 --device cuda"
    exit 1
fi

echo "✅ Feature files found"
echo ""

# 2. 转换特征格式
echo "Converting features to pipeline format..."
python3 << 'EOF'
import pickle
import numpy as np
from pathlib import Path

train_features = np.load('baselines/diehl_cook/outputs/train_features.npy')
test_features = np.load('baselines/diehl_cook/outputs/test_features.npy')

output_dir = Path('outputs/codes')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'diehl_cook_mnist_seed0.pkl', 'wb') as f:
    pickle.dump({
        'train_codes': train_features,
        'test_codes': test_features
    }, f)

print(f"Train: {train_features.shape}")
print(f"Test: {test_features.shape}")
print("✅ Converted!")
EOF

echo ""

# 3. 聚类评估
echo "=================================================="
echo "  Running Clustering Evaluation"
echo "=================================================="
python scripts/run_baseline.py --config configs/diehl_cook.yaml --force

echo ""

# 4. SVM 评估
echo "=================================================="
echo "  Running SVM Evaluation"
echo "=================================================="
python scripts/run_supervised_eval.py \
    --baseline diehl_cook \
    --dataset mnist \
    --seed 0

echo ""
echo "=================================================="
echo "  Evaluation Complete!"
echo "=================================================="
echo ""
echo "Results saved in:"
echo "  - outputs/results/diehl_cook_mnist_seed0.json (clustering)"
echo "  - outputs/results/diehl_cook_mnist_seed0_supervised.json (SVM)"
echo ""
```

### 使用方法

```bash
cd /hy-tmp/clustering
chmod +x eval_trained_diehl_cook.sh
./eval_trained_diehl_cook.sh
```

---

## 📈 预期结果

### 完整 STDP 训练后（60000样本）

```
无监督聚类 (K-Means):
├─ NMI: 0.60-0.70
├─ ARI: 0.50-0.60
└─ ACC: 0.65-0.75

有监督 SVM:
├─ Accuracy: 0.85-0.95
└─ F1: 0.83-0.93

原文报告:
└─ SVM Accuracy: ~95%
```

如果你的结果接近这些数值，说明训练成功！

---

## 🔍 检查训练文件

### 查看保存的文件

```bash
# 查找所有保存的文件
find baselines/diehl_cook -type f -name "*.npy" -o -name "*.pt" -o -name "*.pkl"

# 查看文件大小和时间
ls -lh baselines/diehl_cook/outputs/
ls -lh baselines/diehl_cook/saved_models/

# 检查特征形状
python -c "
import numpy as np
test_features = np.load('baselines/diehl_cook/outputs/test_features.npy')
train_features = np.load('baselines/diehl_cook/outputs/train_features.npy')
print(f'Train features: {train_features.shape}')
print(f'Test features: {test_features.shape}')
print(f'Expected: (60000, n_neurons) and (10000, n_neurons)')
"
```

---

## 🐛 常见问题

### Q1: 找不到特征文件

```bash
# 检查默认输出位置
ls -la baselines/diehl_cook/outputs/
ls -la baselines/diehl_cook/saved_models/

# 如果指定了 --output_dir，检查那个目录
ls -la outputs/diehl_cook/
```

### Q2: 特征形状不对

```python
# 应该是:
train_features: (60000, n_neurons)  # 如 (60000, 400)
test_features: (10000, n_neurons)   # 如 (10000, 400)

# 如果形状不对，检查训练脚本的保存逻辑
```

### Q3: 如何重新训练

```bash
# 删除旧的输出
rm -rf baselines/diehl_cook/outputs/
rm -rf baselines/diehl_cook/saved_models/

# 重新训练
cd baselines/diehl_cook
python train.py --train --extract \
    --n_train 60000 \
    --n_epochs 1 \
    --device cuda
```

---

## 📝 快速参考

### 训练

```bash
cd /hy-tmp/clustering/baselines/diehl_cook
python train.py --train --extract --n_train 60000 --device cuda
```

### 评估（方法选一）

```bash
# 方法 1: 使用主 pipeline
cd /hy-tmp/clustering
# (先转换特征格式，见上文)
python run.py --baseline diehl_cook --force

# 方法 2: 手动评估
python eval_diehl_cook.py

# 方法 3: SVM 评估
python scripts/run_supervised_eval.py --baseline diehl_cook --dataset mnist
```

### 查看结果

```bash
# 聚类结果
cat outputs/results/diehl_cook_mnist_seed0.json

# SVM 结果
cat outputs/results/diehl_cook_mnist_seed0_supervised.json
```

---

## 🎯 总结

### 训练后的文件位置

```
baselines/diehl_cook/
├── outputs/
│   ├── train_features.npy  ⭐ 训练集特征
│   └── test_features.npy   ⭐ 测试集特征
└── saved_models/
    └── diehl_cook_network.pt  ⭐ 网络权重
```

### 评估流程

1. ✅ **转换特征格式** → `outputs/codes/diehl_cook_mnist_seed0.pkl`
2. ✅ **聚类评估** → `python run.py --baseline diehl_cook --force`
3. ✅ **SVM 评估** → `python scripts/run_supervised_eval.py ...`
4. ✅ **查看结果** → `cat outputs/results/*.json`

---

**推荐使用"方法 1"或一键脚本进行评估！** 🚀
