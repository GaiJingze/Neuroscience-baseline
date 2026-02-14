# Diehl & Cook Baseline 深度审查报告

**审查日期**: 2026-02-14
**审查范围**: encoder 实现正确性、与原文一致性、pipeline 端到端可运行性

---

## 一、结论概要

| 维度 | 状态 | 说明 |
|------|------|------|
| **与原文一致性** | ❌ 有严重偏差 | 侧向抑制电路权重符号错误、Exc→Inh 权重过弱、兴奋层 reset 电位不符 |
| **实现 Bug** | ❌ 有致命 Bug | 抑制回路完全失效，WTA 竞争机制不起作用 |
| **Pipeline 可运行性** | ⚠️ MNIST 可跑通但结果无效 | 已有失败报告：NMI=0, ARI=0, 所有样本生成相同编码 |
| **非 MNIST 数据集** | ❌ 无法运行 | input_dim=784 硬编码，SIFT1M/GloVe 维度不匹配会崩溃 |

---

## 二、致命 Bug：侧向抑制电路完全失效

这是导致 "所有神经元响应完全一致" 的**根本原因**。

### Bug #1（严重）：Inh→Exc 权重符号错误 — 侧向"抑制"实为侧向激励

**文件**: `baselines/diehl_cook/encoder.py:139-148`

```python
# 当前代码
w = 10.4 * (torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons)))
inh_exc_conn = Connection(
    source=inh_layer, target=exc_layer,
    w=w,                # ← 权重为 +10.4（正值！）
    wmin=-120.0,        # ← 声明范围为 [-120, 0]
    wmax=0.0,           # ← 但实际权重 +10.4 不在此范围内
)
```

**问题分析**：
- 权重矩阵 `w` 的非对角元素为 **+10.4**（正值）
- BindsNET 在显式提供 `w` 时**不会**自动裁剪到 `[wmin, wmax]`（仅在学习更新时裁剪）
- 此连接无学习规则，权重永远不会被更新/裁剪，始终保持 +10.4
- 正权重意味着：当抑制层神经元发放时，其余兴奋层神经元的膜电位**上升**（去极化）
- 这是**侧向激励**，不是侧向抑制！完全违背 WTA 竞争机制

**原文要求**: Inh→Exc 权重应为**负值**（如 -17.0），使抑制层发放时压低其他兴奋层神经元

**修复方案**:
```python
# 正确写法：权重应为负值
w = -17.0 * (torch.ones(self.n_neurons, self.n_neurons)
             - torch.diag(torch.ones(self.n_neurons)))
```

### Bug #2（严重）：Exc→Inh 权重过弱 — 抑制层神经元根本不会发放

**文件**: `baselines/diehl_cook/encoder.py:128-137`

```python
# 当前代码
w = torch.eye(self.n_neurons)  # ← 权重仅为 1.0
exc_inh_conn = Connection(
    source=exc_layer, target=inh_layer,
    w=w,              # ← one-to-one，权重=1.0
    wmin=0,
    wmax=22.5,        # ← 允许最大 22.5，但实际只有 1.0
)
```

**问题分析**（膜电位计算）：

抑制层参数：`rest=-60, thresh=-40, decay=0.1`

BindsNET LIF 更新公式：
```
v_new = decay × (v - rest) + rest + x
      = 0.1 × (v - (-60)) + (-60) + x
```

从静息电位出发，持续接收 x=1.0 的输入：
```
v_ss = rest + x / (1 - decay) = -60 + 1.0 / 0.9 = -58.89 mV
```

阈值为 -40 mV，稳态电位 -58.89 mV **远低于阈值**。即使兴奋层持续发放，抑制层也**永远无法达到阈值**。

**原文参数**: Exc→Inh 权重在原文实现中通常为 10.4 或更高（而非 1.0），以确保抑制层可靠发放。

**修复方案**:
```python
# 正确写法：权重需足够大以驱动抑制层发放
w = 22.5 * torch.eye(self.n_neurons)
```

### 综合影响

两个 Bug 叠加的结果：
1. 抑制层神经元**从不发放**（Exc→Inh 权重 1.0 太弱）
2. 即使抑制层发放，也会**激励**而非抑制其他兴奋层神经元（权重为正）
3. **WTA 竞争机制完全失效** → 所有兴奋层神经元响应相同 → STDP 无法分化 → 所有权重趋同
4. 这与已有失败报告 (`docs/DIEHL_COOK_FAILURE_REPORT.md`) 中 "所有神经元产生相同脉冲计数" 的现象完全吻合

---

## 三、与 Diehl & Cook (2015) 原文的逐项对比

### 网络架构

| 参数 | 原文 (2015) | 当前实现 | 一致？ |
|------|-------------|---------|--------|
| 输入层 | 784 Poisson 神经元 | 784 Input(traces=True) | ✅ |
| 兴奋层 | 400 LIF (n_e=400) | 400 LIFNodes | ✅ |
| 抑制层 | 400 LIF (n_i=400) | 400 LIFNodes | ✅ |
| Input→Exc | STDP, all-to-all | PostPre, (784,400) | ✅ |
| Exc→Inh | one-to-one | eye(400) | ✅ 结构对，权重不对 |
| Inh→Exc | all-to-all minus diagonal | ones-diag | ✅ 结构对，符号不对 |

### 神经元参数

| 参数 | 原文 | 当前实现 | 一致？ |
|------|------|---------|--------|
| v_rest (Exc) | -65.0 mV | -65.0 mV | ✅ |
| **v_reset (Exc)** | **-65.0 mV** | **-60.0 mV** | ❌ 偏差 5mV |
| v_thresh (Exc) | -52.0 mV | -52.0 mV | ✅ |
| refrac (Exc) | 5 ms | 5 ms | ✅ |
| v_rest (Inh) | -60.0 mV | -60.0 mV | ✅ |
| v_reset (Inh) | -45.0 mV | -45.0 mV | ✅ |
| v_thresh (Inh) | -40.0 mV | -40.0 mV | ✅ |
| refrac (Inh) | 2 ms | 2 ms | ✅ |
| theta_plus | 0.05 mV | 0.05 | ✅ |
| tc_theta_decay | 1e7 ms | 1e7 | ✅ |

**v_reset 偏差的影响**: 原文 reset=-65（等于 rest），代码 reset=-60（比 rest 高 5mV），导致神经元在脉冲后恢复更快、更容易再次发放。影响中等。

### 连接权重

| 参数 | 原文 | 当前实现 | 一致？ |
|------|------|---------|--------|
| Input→Exc 初始化 | Uniform [0, 0.3] | `0.3*rand()` → [0, 0.3] | ✅ |
| Input→Exc 范围 | [0, 1] | wmin=0, wmax=1 | ✅ |
| Input→Exc norm | 78.4 per neuron | norm=78.4 | ✅ |
| STDP nu_pre | 1e-4 | 1e-4 | ✅ |
| STDP nu_post | 1e-2 | 1e-2 | ✅ |
| **Exc→Inh 权重** | **10.4** | **1.0 (eye)** | ❌ 过弱 |
| **Inh→Exc 权重** | **-17.0** | **+10.4** | ❌ 符号反了 |

### 训练设置

| 参数 | 原文 | 当前实现 | 一致？ |
|------|------|---------|--------|
| Poisson 编码时间 | 350 ms | 350 ms | ✅ |
| 训练数据 | 60,000 MNIST | 支持（默认全部） | ✅ |
| **训练轮数** | **多轮（原始代码 3 轮）** | **仅 1 轮** | ⚠️ |
| **特征提取** | **脉冲计数 → 标签分配 → SVM** | **脉冲计数 → top-k 二值化** | ⚠️ 不同 |

---

## 四、Pipeline Bug 分析

### Bug #3（中等）：双重二值化 — Encoder 内部 + Pipeline 外部各做一次

**文件**: `baselines/diehl_cook/encoder.py:339-341` 和 `scripts/run_baseline.py:248-256`

```python
# encoder.py 内部已做 top-5% 二值化
k = max(int(self.n_neurons * 0.05), 1)
code = self._top_k_binarization(pre_code, k)

# run_baseline.py 又从 config 读取参数，对 pre_code 再做一次
if config['binarization_method'] == 'top_k_percent':
    percent = config['binarization_params']['percent']
    code = top_k_percent_binarization(pre_code, percent)
```

**影响**: 最终结果由 run_baseline.py 的二值化覆盖（因为用的是 pre_code），所以功能上不算错，但 encoder 内部的二值化是冗余的，且 encoder 的 `code` 返回值在 pipeline 中被丢弃了，造成语义混乱。

### Bug #4（中等）：检索评测中 query 和 database 二值化不一致

**文件**: `scripts/run_baseline.py:308-323`

```python
# 有标签的检索模式 (MNIST)
if has_labels:
    train_encoded = encoder.encode(dataset['train_data'])
    database_code = train_encoded['code']      # ← 使用 encoder 内部二值化（5%）
    query_code = code                           # ← 使用 config 二值化后的结果（也是5%，但来源不同）
```

当 config 的 binarization 参数与 encoder 内部不一致时，query 和 database 会使用不同的二值化策略。此外，对 train_data 的编码没有经过 config 中的 binarization pipeline。

### Bug #5（高）：非 MNIST 数据集无法运行

**文件**: `configs/diehl_cook.yaml:15`

```yaml
encoder_config:
  input_dim: 784  # 硬编码为 MNIST
```

SIFT1M 为 128 维，GloVe 为 50/100/200/300 维。使用 diehl_cook 跑这些数据集时，网络输入层为 784 维，但实际数据维度不匹配，会在 `network.run()` 时因矩阵乘法维度错误而崩溃：
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x1 and 784x400)
```

### Bug #6（低）：环境缺失 — BindsNET 未安装

BindsNET 未安装在当前环境中，且存在与 PyTorch 2.x 的兼容性问题（`torch._six` 被移除）。需要：
```
torch==1.13.1
numpy<2.0.0
bindsnet==0.3.1
```

---

## 五、已知问题与失败报告对应

仓库中已有 `docs/DIEHL_COOK_FAILURE_REPORT.md`，记录了以下现象：

```
Unique codes: 1 / 10,000 samples
NMI = 0.0, ARI = 0.0, ACC = 0.1135 (随机水平)
所有 400 个神经元对每个样本产生完全相同的脉冲计数
```

**该报告归因于 "权重归一化过强"，但未发现真正的根本原因。**

根据本次审查，**真正的根本原因**是：

1. **Inh→Exc 权重符号错误**（+10.4 应为 ~-17.0）→ 侧向"抑制"实际是侧向激励
2. **Exc→Inh 权重过弱**（1.0 太小，抑制层永远不发放）
3. 以上两点导致 WTA 机制完全失效 → 所有神经元行为一致 → STDP 无法分化权重

权重归一化 (norm=78.4) 本身是原文参数，不是主要问题。

---

## 六、完整修复方案

### 优先级 1：修复侧向抑制电路（致命）

```python
# encoder.py: _build_network()

# 修复 Exc→Inh 权重（从 1.0 增加到 22.5）
w = 22.5 * torch.eye(self.n_neurons)
exc_inh_conn = Connection(
    source=exc_layer, target=inh_layer,
    w=w, wmin=0, wmax=22.5,
)

# 修复 Inh→Exc 权重（从 +10.4 改为 -17.0）
w = -17.0 * (torch.ones(self.n_neurons, self.n_neurons)
             - torch.diag(torch.ones(self.n_neurons)))
inh_exc_conn = Connection(
    source=inh_layer, target=exc_layer,
    w=w, wmin=-120.0, wmax=0.0,
)
```

### 优先级 2：修复 v_reset（中等）

```python
exc_layer = LIFNodes(
    ...
    reset=-65.0,  # 从 -60.0 改为 -65.0，与原文一致
    ...
)
```

### 优先级 3：Pipeline 二值化一致性

```python
# 方案 A：从 encoder 中移除内部二值化，统一由 pipeline 处理
# encoder.py encode() 方法中删除 _top_k_binarization 调用
# 让 code = pre_code，由 run_baseline.py 统一二值化

# 方案 B：在 run_baseline.py 中对 database_code 也应用 config 二值化
train_encoded = encoder.encode(dataset['train_data'])
database_code = top_k_percent_binarization(train_encoded['pre_code'], percent)
```

### 优先级 4：支持多 epoch 训练

```python
# encoder.py fit() 中增加 epoch 循环
n_epochs = self.config.get('n_epochs', 1)
for epoch in range(n_epochs):
    for i, image in enumerate(train_data_normalized):
        # ... 现有训练逻辑
```

### 优先级 5：数据集维度自适应

```python
# run_baseline.py 中根据数据集自动设置 input_dim
actual_dim = dataset['train_data'].shape[1]
config['encoder_config']['input_dim'] = actual_dim
```

---

## 七、总结

Diehl & Cook baseline 存在**两个致命的实现错误**（Inh→Exc 权重符号、Exc→Inh 权重量级），直接导致 WTA 竞争机制完全失效。这不是参数调优能解决的问题，而是需要修正代码中的权重定义。修复后应能看到神经元分化和有意义的聚类性能。

Pipeline 方面存在双重二值化、检索评测不一致等中等问题，以及非 MNIST 数据集维度不匹配的高优先级问题，但这些都不影响 MNIST 上 encoder 本身的正确性判断。
