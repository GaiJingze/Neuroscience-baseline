# Diehl & Cook：原文 vs 我们的实现

## ❓ 问题

**这三种聚类方法（kmeans, spectral, kmedoids）是 Diehl & Cook 原生支持的吗？**

## ✅ 简短答案

**不是！** 这些聚类方法不是 Diehl & Cook 原生的，而是**我们的 pipeline 提供的通用评估工具**。

---

## 🔍 详细解释

### Diehl & Cook 原文做了什么？

**论文**: "Unsupervised learning of digit recognition using spike-timing-dependent plasticity" (Diehl & Cook, 2015)

#### 1. 任务类型

- ❌ **不是聚类任务**
- ✅ **是分类任务**（有监督）

#### 2. 方法流程

```
原文的完整流程:
──────────────────────────────────────────────────────

1. 无监督特征学习（STDP）
   ├─ 输入: MNIST 图像
   ├─ 网络: SNN (LIF neurons + STDP)
   └─ 输出: Spike counts (特征向量)

2. 标签分配（Label Assignment）
   ├─ 输入: Spike counts + 训练集标签
   ├─ 方法: 为每个神经元分配最常激活的类别
   └─ 输出: 神经元→类别的映射

3. 分类（Classification）
   ├─ 输入: 测试集的 spike counts
   ├─ 方法: 线性 SVM 或简单投票
   └─ 输出: 分类准确率

最终评估指标:
├─ Classification Accuracy: ~95%
└─ 使用监督学习方法（SVM）评估
```

#### 3. 原文使用的评估方法

Diehl & Cook 原文使用：

- ✅ **线性 SVM**（支持向量机）- 有监督分类
- ✅ **简单投票机制** - 每个神经元对应一个类别
- ❌ **没有使用聚类算法**

---

### 我们的实现做了什么？

**任务**: 无监督聚类评估（不是分类）

#### 1. 任务类型

- ✅ **聚类任务**（无监督）
- ❌ **不是分类任务**

#### 2. 方法流程

```
我们的流程:
──────────────────────────────────────────────────────

1. 特征提取（使用 Diehl & Cook）
   ├─ 输入: MNIST 图像
   ├─ 网络: SNN (LIF neurons + STDP)
   └─ 输出: Spike counts (特征向量)

2. 聚类（我们的 pipeline）⭐ 这是新增的
   ├─ 输入: Spike counts（不使用标签）
   ├─ 方法: K-Means / Spectral / KMedoids
   └─ 输出: 聚类标签

3. 评估（我们的 pipeline）
   ├─ 输入: 聚类标签 + 真实标签
   ├─ 方法: 计算 NMI, ARI, ACC
   └─ 输出: 聚类质量指标

关键区别:
├─ 原文: STDP + SVM（有监督）
└─ 我们: STDP + 聚类（无监督）⭐
```

---

## 🎯 关键区别总结

| 方面 | Diehl & Cook 原文 | 我们的实现 |
|------|------------------|-----------|
| **任务类型** | 分类（Classification） | 聚类（Clustering） |
| **是否监督** | 有监督评估 | 无监督评估 |
| **特征提取** | STDP (SNN) ✅ | STDP (SNN) ✅ |
| **评估方法** | 线性 SVM | K-Means / Spectral / KMedoids |
| **使用标签** | 训练+测试都用标签 | 只在评估时用标签对比 |
| **评估指标** | Accuracy (~95%) | NMI, ARI, ACC |

---

## 📊 聚类方法来源

### 这些聚类算法从哪来？

```python
# 在 pipeline/clustering.py 中定义
from sklearn.cluster import KMeans, SpectralClustering
from sklearn_extra.cluster import KMedoids

def run_clustering_evaluation(codes, labels, n_clusters=10, methods=['kmeans']):
    """
    运行聚类评估
    
    这是我们 pipeline 的通用聚类评估工具
    不是 Diehl & Cook 原生的！
    """
    results = {}
    
    if 'kmeans' in methods:
        results['kmeans'] = kmeans_clustering(codes, labels, n_clusters)
    
    if 'spectral' in methods:
        results['spectral'] = spectral_clustering(codes, labels, n_clusters)
    
    if 'kmedoids' in methods:
        results['kmedoids'] = kmedoids_clustering(codes, labels, n_clusters)
    
    return results
```

### 聚类算法的来源

| 算法 | 来源 | 用途 |
|------|------|------|
| **K-Means** | scikit-learn | 通用聚类，最常用 |
| **Spectral** | scikit-learn | 非线性聚类，基于图论 |
| **KMedoids** | scikit-learn-extra | 对离群点更鲁棒 |

这些都是**标准的机器学习聚类算法**，不是 Diehl & Cook 特有的。

---

## 🔬 为什么我们用聚类而不是分类？

### 项目目标

我们的项目是：**"Clustering/Hashing feature track"**

```
项目任务:
├─ Task A: 无监督特征学习 & 聚类 ⭐
└─ Task B: 局部敏感哈希（LSH）
```

### 评估策略

我们要评估的是：

1. ✅ **特征质量**：STDP 学到的特征有多好？
2. ✅ **聚类性能**：这些特征能否将相似样本聚在一起？
3. ✅ **无监督能力**：不使用标签能做到什么程度？

### 为什么不用原文的 SVM？

原文使用 SVM 是因为：
- 目标：分类准确率（有监督）
- 任务：手写数字识别

我们使用聚类是因为：
- 目标：无监督特征学习
- 任务：评估特征的内在结构

---

## 💡 统一的评估 Pipeline

### 所有 baseline 都用相同的聚类评估

```python
所有方法的评估流程:
──────────────────────────────────────

FlyHash:
├─ 特征提取: Random projection + WTA
└─ 聚类评估: K-Means / Spectral / KMedoids ⭐

Diehl & Cook:
├─ 特征提取: STDP (SNN)
└─ 聚类评估: K-Means / Spectral / KMedoids ⭐

SoftHebb:
├─ 特征提取: Hebbian learning
└─ 聚类评估: K-Means / Spectral / KMedoids ⭐

相同的评估方法 → 公平对比！
```

### 优势

1. ✅ **公平对比**：所有方法用相同的评估标准
2. ✅ **标准化**：使用广泛认可的聚类算法
3. ✅ **灵活性**：可以尝试不同的聚类方法
4. ✅ **无监督**：符合项目目标

---

## 🎓 深入理解：分类 vs 聚类

### 分类（Classification）- 有监督

```python
# Diehl & Cook 原文的做法

# 1. 训练阶段（使用标签）
for image, label in train_data:
    spike_counts = snn.encode(image)
    # 为激活最多的神经元分配标签
    neuron_labels[argmax(spike_counts)] = label

# 2. 测试阶段（使用学到的映射）
for image in test_data:
    spike_counts = snn.encode(image)
    predicted_label = neuron_labels[argmax(spike_counts)]
    # 或使用 SVM
    predicted_label = svm.predict(spike_counts)

# 3. 评估
accuracy = (predicted == true_labels).mean()
print(f"Accuracy: {accuracy}")  # ~95%
```

### 聚类（Clustering）- 无监督

```python
# 我们的做法

# 1. 特征提取（不使用标签）
spike_counts = []
for image in data:
    spike_counts.append(snn.encode(image))

# 2. 聚类（不使用标签）
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
cluster_labels = kmeans.fit_predict(spike_counts)

# 3. 评估（只在这一步用真实标签对比）
nmi = normalized_mutual_info_score(true_labels, cluster_labels)
ari = adjusted_rand_score(true_labels, cluster_labels)
acc = clustering_accuracy(true_labels, cluster_labels)

print(f"NMI: {nmi}, ARI: {ari}, ACC: {acc}")
```

---

## 📊 性能对比：分类 vs 聚类

### Diehl & Cook 原文（分类）

```
任务: 有监督分类
方法: STDP + SVM
结果: Accuracy ~95%
```

### 我们的实现（聚类）

```
任务: 无监督聚类
方法: STDP + K-Means
结果: 
  - 骨架版本: NMI ~0.40, ACC ~0.49
  - 完整版本: NMI ~0.65, ACC ~0.70 (预期)
```

### 为什么聚类性能更低？

这是**正常且预期的**：

| 因素 | 分类 | 聚类 |
|------|------|------|
| **标签使用** | 训练时用 | 完全不用 |
| **监督信号** | 强 | 无 |
| **任务难度** | 较易 | 较难 |
| **典型性能** | 90-95% | 50-70% |

聚类本身就比分类难，因为没有监督信号！

---

## 🎯 回答你的问题

### Q: 这三种聚类方法是 Diehl & Cook 原生支持的吗？

**A: 不是！**

1. **Diehl & Cook 只负责特征提取**
   - 使用 STDP 学习
   - 输出 spike counts

2. **聚类算法是我们 pipeline 提供的**
   - K-Means (scikit-learn)
   - Spectral Clustering (scikit-learn)
   - KMedoids (scikit-learn-extra)

3. **原文使用的是分类方法**
   - 线性 SVM
   - 简单投票机制
   - 不是聚类

4. **所有 baseline 都用相同的聚类评估**
   - FlyHash → K-Means / Spectral / KMedoids
   - Diehl & Cook → K-Means / Spectral / KMedoids
   - SoftHebb → K-Means / Spectral / KMedoids
   - 确保公平对比

---

## 📝 总结

### 关键要点

```
Diehl & Cook 的角色:
├─ ✅ 特征提取器（STDP-based SNN）
├─ ✅ 生成 spike counts 作为特征
└─ ❌ 不包含聚类算法

我们 Pipeline 的角色:
├─ ✅ 调用 Diehl & Cook 提取特征
├─ ✅ 提供统一的聚类评估（K-Means等）
└─ ✅ 计算聚类指标（NMI, ARI, ACC）

聚类算法来源:
├─ K-Means: scikit-learn（标准算法）
├─ Spectral: scikit-learn（标准算法）
└─ KMedoids: scikit-learn-extra（标准算法）
```

### 流程图

```
┌─────────────────────────────────────────────┐
│         Diehl & Cook (特征提取)             │
│  MNIST → STDP SNN → Spike Counts            │
│  [原文负责的部分]                           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│      我们的 Pipeline (聚类评估)             │
│  Spike Counts → K-Means/Spectral/KMedoids  │
│              → NMI, ARI, ACC                │
│  [我们添加的部分]                           │
└─────────────────────────────────────────────┘
```

---

**简单说**：

- ✅ Diehl & Cook = **特征提取器**（生成 spike counts）
- ✅ K-Means/Spectral/KMedoids = **我们 pipeline 的通用聚类工具**
- ✅ 原文用 SVM 做分类，我们用聚类做无监督评估
- ✅ 这样可以公平对比所有 baseline 的特征质量

你的结果（NMI=0.40, ARI=0.29, ACC=0.49）反映的是：
- Diehl & Cook（骨架版本）提取的特征质量
- 在无监督聚类任务上的表现
- 不是原文报告的有监督分类性能（~95%）
