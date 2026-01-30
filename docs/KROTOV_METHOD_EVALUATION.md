# Krotov 方法整合评估报告

## 📚 方法概述

**论文**: [Unsupervised Learning by Competing Hidden Units](https://doi.org/10.1073/pnas.1820458116)  
**作者**: Dmitry Krotov & John Hopfield (2019, PNAS)  
**仓库**: https://github.com/DimaKrotov/Biological_Learning

### 核心思想

**竞争隐藏单元学习** (Competing Hidden Units):
- 类似 Winner-Take-All，但使用 **k-WTA**（不只是1个获胜者）
- **反 Hebbian 学习** (Anti-Hebbian)：获胜者加强，第k名削弱
- **权重归一化**：使用 Lebesgue p-范数

### 数学公式

权重更新规则（简化版）:
```
Δw_i = η * [y_i * x - y_i * Q_i * w_i]
```

其中:
- `y_i`: 神经元激活（winner: +1, k-th: -δ, 其他: 0）
- `Q_i = w_i · x`: 输入重叠
- `δ`: 反 Hebbian 强度（~0.4）

## 🔍 代码分析

### 实现结构

**文件**: `Unsupervised_learning_algorithm_MNIST.ipynb`
- **语言**: Python (NumPy)
- **依赖**: scipy, numpy, matplotlib
- **代码量**: ~50 行核心代码
- **训练方式**: 小批量梯度下降（minibatch）

### 关键参数

```python
eps0 = 2e-2      # 初始学习率
hid = 100        # 隐藏单元数量 (10×10)
Nep = 200        # 训练轮数
Num = 100        # Minibatch 大小
delta = 0.4      # 反 Hebbian 强度
p = 2.0          # Lebesgue 范数
k = 2            # 排名参数 (k-WTA)
```

### 算法流程

```python
for epoch in range(Nep):
    for minibatch in data:
        # 1. 计算激活: Q = sign(W) * |W|^(p-1) @ X
        tot_input = dot(sign(W) * abs(W)**(p-1), X)
        
        # 2. k-WTA 选择
        y = argsort(tot_input, axis=0)  # 排序
        yl[winner] = 1.0                 # 最强激活
        yl[k-th] = -delta                # 第k名削弱
        
        # 3. 权重更新（反 Hebbian）
        ds = yl @ X.T - (yl * tot_input).sum(1) * W
        W += eps * ds / max(abs(ds))
```

## 📊 与当前 Baselines 对比

### 算法比较

| 特性 | FlyHash | SoftHebb | Krotov | Diehl&Cook |
|------|---------|----------|--------|------------|
| **学习方式** | 无训练 | Soft WTA + Hebbian | k-WTA + Anti-Hebbian | STDP (SNN) |
| **生物合理性** | ❌ 低 | ✅ 中 | ✅✅ 高 | ✅✅✅ 最高 |
| **训练速度** | - | ⚡ 快 (~10 epochs) | ⚡⚡ 很快 (~200 epochs, minibatch) | 🐌 极慢 (~hours) |
| **MNIST NMI** | 0.5494 | 0.1806 | ❓ 未知 | 0.0000 (bug) |
| **实现复杂度** | ⭐ 简单 | ⭐⭐ 中等 | ⭐⭐ 中等 | ⭐⭐⭐⭐ 复杂 |

### 关键差异

**与 SoftHebb 的区别**:
1. **k-WTA vs Soft-WTA**: Krotov 使用硬选择（排序），SoftHebb 使用 softmax
2. **Anti-Hebbian**: Krotov 有负反馈（-δ），SoftHebb 只有正向
3. **权重归一化**: Krotov 使用 p-范数，SoftHebb 使用 L2
4. **数学支撑**: Krotov 有能量函数证明，SoftHebb 较经验

## 🎯 整合难度评估

### ✅ 优势

1. **代码简洁** - 核心算法只有 ~50 行
2. **依赖简单** - 只需 NumPy（无需 PyTorch）
3. **训练快速** - Minibatch 更新，比 Diehl&Cook 快得多
4. **理论支撑强** - PNAS 论文，有数学证明
5. **已在 MNIST 验证** - 仓库就是 MNIST 示例

### ⚠️ 挑战

1. **性能未知** - 原论文主要展示特征学习，未报告聚类性能
2. **参数敏感** - 需要调整 7 个超参数（delta, p, k, eps, 等）
3. **只有单层** - 原实现是单层，多层需要自己实现
4. **没有编码接口** - 只有训练代码，需要添加 encode 方法
5. **可能性能不高** - 类似 SoftHebb 的方法，可能 NMI < 0.3

### 📈 整合工作量估算

**最小实现** (2-4 小时):
- ✅ 简单：将 notebook 转换为 Python 类
- ✅ 实现 `fit()` 和 `encode()` 方法
- ✅ 添加配置文件
- ✅ 跑一个 MNIST 实验验证

**完整实现** (1-2 天):
- ⚠️ 多层扩展（如果需要）
- ⚠️ 参数调优（grid search）
- ⚠️ 在 Fashion-MNIST 上验证
- ⚠️ 与其他 baselines 全面对比

## 💡 整合建议

### 方案 A: 快速原型 ✅ **推荐**

**适用场景**: 想快速验证这个方法的性能

**步骤**:
1. 创建 `baselines/krotov/encoder.py`（2小时）
2. 复制核心算法代码（已经很简洁）
3. 实现 `encode()`：使用学习好的权重编码
4. 跑 MNIST seed=0 验证（1小时训练）
5. 如果 NMI > 0.3，继续完整实验

**优点**: 
- 快速验证（半天内完成）
- 代码简单，易于 debug
- 如果性能好，可以作为额外 baseline

**缺点**:
- 可能性能不如 FlyHash
- 需要额外调参

### 方案 B: 暂时搁置 ⏸️ 

**适用场景**: 当前结果已经足够发表

**理由**:
1. FlyHash (NMI 0.55) 已经很强
2. Diehl&Cook 修复后可能有提升
3. 时间有限，优先完成主要实验

**后续**:
- 在论文 Future Work 中提及
- 作为后续研究方向

## 🔬 快速验证方案

如果你想快速测试，我可以帮你：

### 步骤 1: 创建 Krotov Encoder (30分钟)

```python
# baselines/krotov/encoder.py
class KrotovEncoder(BaseEncoder):
    def __init__(self, config):
        self.input_dim = 784
        self.n_neurons = config.get('n_neurons', 100)
        self.n_epochs = config.get('n_epochs', 200)
        self.batch_size = config.get('batch_size', 100)
        self.lr = config.get('lr', 0.02)
        self.delta = config.get('delta', 0.4)
        self.p = config.get('p', 2.0)
        self.k = config.get('k', 2)
        self.W = None
        
    def fit(self, train_data, train_labels=None):
        # 实现 Krotov 算法（直接从 notebook 复制）
        pass
    
    def encode(self, data):
        # Q = sign(W) * |W|^(p-1) @ X
        # Top-k binarization
        pass
```

### 步骤 2: 快速测试 (1-2小时训练)

```bash
cd /hy-tmp/clustering
python scripts/run_baseline.py --config configs/krotov.yaml --dataset mnist --seed 0
```

### 步骤 3: 评估结果

- 如果 NMI > 0.4: ✅ 值得完整整合
- 如果 NMI 0.2-0.4: ⚠️ 与 SoftHebb 相当
- 如果 NMI < 0.2: ❌ 性能不足

## 📝 总结

### 整合难度: ⭐⭐ (中等)

**核心评估**:
- ✅ 代码简单，理论扎实
- ⚠️ 性能未知，可能不如 FlyHash
- ⏱️ 快速原型只需半天
- 🎲 风险：可能投入产出比不高

### 我的建议

**如果你有 4-6 小时空余时间**:
→ 值得快速实现并验证

**如果时间紧张**:
→ 先完成 FlyHash/SoftHebb 的论文撰写
→ Krotov 作为 Future Work 提及

---

**下一步行动**:

想要我现在帮你：
1. ✅ **立即实现** Krotov encoder？（我可以在 30 分钟内完成代码）
2. ⏸️ **暂时搁置**，专注现有结果？
3. 📖 **先看论文**，再决定是否整合？

请告诉我你的选择！
