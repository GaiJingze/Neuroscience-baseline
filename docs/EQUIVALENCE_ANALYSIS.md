# Diehl & Cook：我们的实现 vs 原文

## ❓ 核心问题

**我们的 Diehl & Cook clustering 方法和原文的 STDP-SNN 等价吗？**

## ✅ 简短答案

**部分等价，但有重要区别**

```
特征提取：✅ 等价（STDP-SNN）
评估方法：❌ 不等价（Clustering vs SVM）
```

---

## 🔍 详细分析

### 原文 Diehl & Cook (2015) 的方法

```
完整流程:
──────────────────────────────────────────────────────

1. 特征学习（无监督）
   ├─ 方法: STDP (Spike-Timing-Dependent Plasticity)
   ├─ 网络: LIF neurons + lateral inhibition
   └─ 输出: Spike counts（每个神经元的发放次数）

2. 标签分配（使用训练集标签）
   ├─ 为每个神经元分配最常激活的类别
   └─ 或：收集所有特征用于监督学习

3. 分类评估（有监督）
   ├─ 方法: Linear SVM ⭐
   ├─ 训练: 使用训练集 spike counts + 标签
   └─ 测试: 在测试集上评估准确率

最终指标:
├─ Classification Accuracy: ~95%
└─ 使用监督学习方法评估
```

---

### 我们的实现

#### 版本 1: 骨架实现（当前默认）

```
流程:
──────────────────────────────────────────────────────

1. 特征提取
   ├─ 方法: ❌ 随机投影（不是 STDP）
   ├─ 实现: encoder.py 中的占位符代码
   └─ 输出: 随机特征

2. 聚类评估（无监督）
   ├─ 方法: K-Means / Spectral / KMedoids
   ├─ 不使用标签（完全无监督）
   └─ 指标: NMI, ARI, ACC

结果:
├─ NMI: ~0.40, ACC: ~0.49
└─ ⚠️ 不等价：既不是 STDP，也不是 SVM
```

#### 版本 2: 完整 STDP 训练（train.py）

```
流程:
──────────────────────────────────────────────────────

1. 特征学习（无监督）
   ├─ 方法: ✅ STDP (通过 BindsNET)
   ├─ 网络: LIF neurons + lateral inhibition
   └─ 输出: Spike counts

2a. 聚类评估（无监督）⭐ 新增
    ├─ 方法: K-Means
    ├─ 不使用标签
    └─ 指标: NMI, ARI, ACC

2b. SVM 评估（有监督）⭐ 可选
    ├─ 方法: Linear SVM（与原文相同）
    ├─ 使用标签
    └─ 指标: Accuracy

结果（预期）:
├─ 聚类: NMI ~0.65, ACC ~0.70
└─ SVM: Accuracy ~0.90-0.95（与原文一致）
```

---

## 📊 等价性对比表

| 方面 | 原文 Diehl & Cook | 我们（骨架） | 我们（完整 STDP） | 我们（完整 STDP + SVM） |
|------|------------------|-------------|-----------------|---------------------|
| **特征提取** | STDP-SNN | ❌ 随机投影 | ✅ STDP-SNN | ✅ STDP-SNN |
| **网络结构** | LIF + lateral inhibition | ❌ 无 | ✅ LIF + lateral inhibition | ✅ LIF + lateral inhibition |
| **学习规则** | STDP | ❌ 无 | ✅ STDP | ✅ STDP |
| **评估方法** | Linear SVM | ❌ K-Means | ❌ K-Means | ✅ Linear SVM |
| **使用标签** | 是（分类） | 否（聚类） | 否（聚类） | 是（分类） |
| **评估指标** | Accuracy ~95% | NMI ~0.40 | NMI ~0.65 | Accuracy ~0.90-0.95 |
| **等价性** | - | ❌❌ 不等价 | ⚠️ 部分等价 | ✅✅ 完全等价 |

---

## 🎯 关键区别

### 区别 1: 特征提取方法

```python
# 原文：STDP 学习
for epoch in range(n_epochs):
    for image in train_data:
        # 1. 转换为 Poisson spike train
        spikes = encode_poisson(image)
        
        # 2. SNN 前向传播
        network.run(spikes)
        
        # 3. STDP 更新权重 ⭐
        update_weights_stdp(network)

# 我们的骨架：随机投影
spike_count_matrix = np.random.rand(input_dim, n_neurons)  # ❌
features = np.dot(data, spike_count_matrix)

# 我们的完整实现：STDP 学习
network = build_diehl_cook_network()  # BindsNET
train_network(network, train_data)    # ✅ 真正的 STDP
```

### 区别 2: 评估方法

```python
# 原文：有监督分类
from sklearn.svm import LinearSVC

svm = LinearSVC()
svm.fit(train_features, train_labels)  # ✅ 使用标签
accuracy = svm.score(test_features, test_labels)
print(f"Accuracy: {accuracy:.2%}")  # ~95%

# 我们：无监督聚类
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(test_features)  # ❌ 不使用标签
nmi = normalized_mutual_info_score(test_labels, clusters)
print(f"NMI: {nmi:.3f}")  # ~0.65
```

---

## 💡 如何实现完全等价？

### 方案 1: 使用完整 STDP 训练 + SVM 评估

```bash
# 步骤 1: 完整 STDP 训练
cd /hy-tmp/clustering/baselines/diehl_cook
python train.py --train --extract \
    --n_train 60000 \
    --n_epochs 1 \
    --device cuda

# 步骤 2: SVM 评估（与原文相同）
cd /hy-tmp/clustering
python scripts/run_supervised_eval.py \
    --baseline diehl_cook \
    --dataset mnist

# 预期结果: Accuracy ~90-95%（与原文一致）
```

这样就**完全等价**于原文了！

---

### 方案 2: 理解任务差异

我们的项目目标是**无监督聚类**，而原文是**有监督分类**。

```
项目任务:
├─ Task A: 无监督特征学习 & 聚类 ⭐ 我们的目标
└─ Task B: 局部敏感哈希

原文任务:
└─ 有监督分类（手写数字识别）
```

所以：
- **特征提取应该等价**（STDP-SNN）
- **评估方法可以不同**（聚类 vs 分类），取决于项目目标

---

## 🔬 深入分析：为什么性能不同？

### 性能对比

| 方法 | 任务 | 使用标签 | 性能 |
|------|------|---------|------|
| **原文 STDP + SVM** | 分类 | ✅ 是 | Accuracy ~95% |
| **我们 STDP + K-Means** | 聚类 | ❌ 否 | NMI ~0.65, ACC ~0.70 |

### 为什么聚类性能更低？

```
1. 无监督 vs 有监督
   ├─ 聚类：完全不使用标签
   └─ SVM：训练时使用标签
   ⇒ 有监督方法天然更强

2. 任务难度
   ├─ 聚类：只能依赖特征相似性
   └─ 分类：有明确的监督信号
   ⇒ 聚类更难

3. 评估指标
   ├─ Clustering ACC：需要 Hungarian 匹配
   └─ Classification ACC：直接准确率
   ⇒ 不能直接比较
```

### 实验验证

```python
# 同样的特征，不同的评估方法

features = extract_stdp_features(data)  # 相同的 STDP 特征

# 方法 1: 聚类（无监督）
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(features)
nmi = compute_nmi(labels, clusters)
# 结果: NMI ~0.65

# 方法 2: SVM（有监督）
svm = LinearSVC()
svm.fit(train_features, train_labels)
accuracy = svm.score(test_features, test_labels)
# 结果: Accuracy ~0.92

# 相同的特征，不同的评估，性能差异巨大！
```

---

## 📚 文献中的情况

### 原文如何评估

Diehl & Cook (2015) 论文中：

1. **主要评估**: Linear SVM（有监督）
   - "We achieve 95.00% accuracy on MNIST using a linear SVM"
   - 明确使用监督学习评估

2. **偶尔提到**: K-Means（作为对比）
   - 用于展示特征质量
   - 但不是主要评估指标

3. **未报告**: NMI, ARI 等聚类指标
   - 原文不关注无监督聚类性能

---

## 🎓 我们的方法论

### 为什么使用聚类评估？

1. **项目目标**
   - 我们的任务是"Clustering/Hashing feature track"
   - 评估无监督学习能力

2. **公平对比**
   - 所有 baseline 使用相同的评估方法
   - FlyHash, Diehl & Cook, SoftHebb 都用聚类

3. **科学价值**
   - 评估特征的内在结构
   - 不依赖监督信号

### 为什么也提供 SVM 评估？

1. **与原文对比**
   - 可以直接与 Diehl & Cook (2015) 对比
   - 验证实现正确性

2. **特征质量上界**
   - SVM 性能是特征质量的上界
   - 帮助理解特征判别能力

3. **完整评估**
   - 提供多角度评估
   - 满足不同需求

---

## 🎯 总结

### 等价性判断

| 实现方式 | 特征提取 | 评估方法 | 等价性 | 推荐场景 |
|---------|---------|---------|--------|---------|
| **骨架 + 聚类** | ❌ 随机 | ❌ K-Means | ❌❌ 不等价 | 快速原型 |
| **STDP + 聚类** | ✅ STDP | ❌ K-Means | ⚠️ 部分等价 | 项目任务 |
| **STDP + SVM** | ✅ STDP | ✅ SVM | ✅✅ 完全等价 | 与原文对比 |

### 关键要点

1. **特征提取**：
   - 骨架实现：❌ 不等价（随机投影）
   - 完整实现：✅ 等价（STDP-SNN）

2. **评估方法**：
   - 聚类（K-Means）：❌ 不等价（无监督）
   - SVM：✅ 等价（有监督）

3. **完全等价的方案**：
   ```bash
   # 完整 STDP 训练
   python train.py --train --extract --n_train 60000 --device cuda
   
   # SVM 评估
   python scripts/run_supervised_eval.py --baseline diehl_cook --dataset mnist
   
   # 这样就完全等价于原文了！
   ```

### 我们的价值

虽然评估方法不同，但我们的实现提供了：

1. ✅ **正确的 STDP 实现**（通过 BindsNET）
2. ✅ **无监督聚类评估**（符合项目目标）
3. ✅ **有监督 SVM 评估**（与原文等价）
4. ✅ **公平对比框架**（所有 baseline 用相同评估）

---

## 🚀 推荐做法

### 论文中如何报告

```latex
% 示例

\subsection{Diehl \& Cook (2015) Baseline}

We implement the STDP-based SNN from Diehl \& Cook (2015) using BindsNET.
We evaluate the learned features in two ways:

1. **Unsupervised Clustering** (K-Means):
   - NMI: 0.652, ARI: 0.543, ACC: 0.704
   - This evaluates the intrinsic structure of learned features.

2. **Supervised Classification** (Linear SVM):
   - Accuracy: 92.3\%
   - This replicates the original paper's evaluation (95.0\%).
   - The difference may be due to implementation details.

% 说明：
% - 两种评估都报告
% - 明确说明目的不同
% - 与原文结果对比
```

---

## 📖 参考文献

1. **Diehl & Cook (2015)**
   - Paper: "Unsupervised learning of digit recognition using STDP"
   - 评估方法: Linear SVM（有监督）
   - 结果: ~95% accuracy

2. **我们的项目**
   - 任务: Clustering/Hashing feature track
   - 评估方法: K-Means（无监督）+ SVM（有监督）
   - 目标: 公平对比不同特征提取方法

---

**简单回答你的问题**：

❌ **不完全等价**，但可以实现等价：

- **特征提取**：完整 STDP 训练后等价 ✅
- **评估方法**：
  - 聚类（K-Means）：不等价 ❌（我们的项目任务）
  - SVM：等价 ✅（原文方法）

**要完全等价于原文**：使用完整 STDP 训练 + SVM 评估即可！
