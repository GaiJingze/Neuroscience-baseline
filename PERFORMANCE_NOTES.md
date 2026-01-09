# Performance Notes

## Clustering Speed Issues

### Problem
运行 baseline evaluation 时，在 clustering 步骤会非常慢，主要原因：

1. **数据规模大**: MNIST test set 有 10,000 samples × 2,000 features
2. **多个聚类算法**: 默认运行 kmeans, kmedoids, spectral 三种算法
3. **算法复杂度**:
   - K-Means: O(n×k×d×i) - 快，但需要多次初始化
   - K-Medoids: O(n²×k×i) - 慢，尤其是大数据集
   - Spectral: O(n³) - 非常慢，需要计算相似度矩阵

### 性能测试结果

在 10,000 samples × 2,000 features 的数据上：

| 算法 | 时间 | 备注 |
|------|------|------|
| KMeans (n_init=10) | ~3-5 分钟 | 10次初始化，每次20-30迭代 |
| KMeans (n_init=3) | ~1-2 分钟 | 优化后 |
| KMedoids | >5 分钟 | O(n²) 复杂度 |
| Spectral | >10 分钟 | 需要构建相似度矩阵 |

### 解决方案

#### 方案 1: 只使用 KMeans（推荐）

```yaml
# configs/flyhash.yaml
clustering_methods:
  - kmeans
```

**优点**: 快速，1-2分钟完成  
**缺点**: 只有一个算法的结果

#### 方案 2: 优化算法参数

已在 `pipeline/clustering.py` 中实现：

```python
# KMeans: 减少初始化次数
defaults = {
    'n_init': 3,  # 从 10 降到 3
    'max_iter': 100,
    'random_state': 0,
}

# Spectral: 减少初始化次数
defaults = {
    'n_init': 3,
    'random_state': 0,
}

# KMedoids: 限制迭代次数
kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, 
                    random_state=0, max_iter=100)
```

#### 方案 3: 使用数据子集

对于快速测试，可以只用部分数据：

```python
# 在 scripts/run_baseline.py 中
test_data = test_data[:1000]  # 只用 1000 samples
test_labels = test_labels[:1000]
```

#### 方案 4: 并行化（未实现）

可以使用 `joblib` 并行运行多个聚类算法：

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(run_single_clustering)(method, codes, labels) 
    for method in methods
)
```

### 当前配置

**默认配置** (`configs/flyhash.yaml`):
```yaml
clustering_methods:
  - kmeans  # 只用 KMeans，快速
```

**完整评估** (如需要):
```yaml
clustering_methods:
  - kmeans
  - kmedoids
  - spectral
```
⚠️ 警告：完整评估可能需要 15-30 分钟

### 推荐工作流程

#### 快速测试（1-2分钟）
```bash
# 使用默认配置（只有 kmeans）
python run.py --baseline flyhash
```

#### 完整评估（15-30分钟）
```bash
# 修改 configs/flyhash.yaml，启用所有算法
# 然后运行
python run.py --baseline flyhash
```

#### 超快速测试（<30秒）
```bash
# 使用小数据集
python scripts/quick_test.py
```

### 优化效果

| 配置 | 时间 | 算法数 |
|------|------|--------|
| 原始（3个算法，n_init=10） | ~30分钟 | 3 |
| 优化（3个算法，n_init=3） | ~10分钟 | 3 |
| 快速（只kmeans，n_init=3） | ~2分钟 | 1 |

### 未来改进

1. **增量聚类**: 对大数据集使用 Mini-Batch K-Means
2. **近似算法**: 使用 FAISS 等库加速
3. **GPU加速**: 使用 cuML (RAPIDS) 进行 GPU 聚类
4. **采样**: 先在子集上评估，再在全集上验证

### 相关文件

- `pipeline/clustering.py` - 聚类算法实现
- `configs/flyhash.yaml` - FlyHash 配置
- `configs/diehl_cook.yaml` - Diehl & Cook 配置
- `scripts/run_baseline.py` - 主运行脚本

---

**更新日期**: 2026-01-09  
**问题**: Clustering evaluation 太慢  
**解决**: 只使用 KMeans，优化参数
