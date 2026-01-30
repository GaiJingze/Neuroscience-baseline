# SoftHebb 详细结果

## 📊 统计结果（Mean ± Std）

### MNIST 数据集（3 seeds: 0, 1, 2）

| Metric | Mean ± Std |
|--------|-----------|
| **NMI** | **0.1806 ± 0.0009** |
| **ARI** | **0.0878 ± 0.0023** |
| **ACC** | **0.2094 ± 0.0008** |

**详细数据：**
- Seed 0: NMI=0.1818, ARI=0.0903, ACC=0.2106
- Seed 1: NMI=0.1802, ARI=0.0883, ACC=0.2089
- Seed 2: NMI=0.1797, ARI=0.0848, ACC=0.2088

**方差分析：**
- ✅ 标准差很小（< 0.003），结果稳定
- ⚠️ 但整体性能较低（NMI < 0.2）

---

### Fashion-MNIST 数据集（2 seeds: 0, 1）

| Metric | Mean ± Std |
|--------|-----------|
| **NMI** | **0.4113 ± 0.0392** |
| **ARI** | **0.2061 ± 0.0503** |
| **ACC** | **0.2448 ± 0.0449** |

**详细数据：**
- Seed 0: NMI=0.3722, ARI=0.1558, ACC=0.1999
- Seed 1: NMI=0.4505, ARI=0.2565, ACC=0.2897

**方差分析：**
- ⚠️ 标准差较大（0.04-0.05），结果变化明显
- ⚠️ 建议补充 seed=2 以提高可靠性

---

## 📋 填表格式

### 简洁版（仅 Mean）

| Dataset | NMI | ARI | ACC |
|---------|-----|-----|-----|
| MNIST | 0.1806 | 0.0878 | 0.2094 |
| Fashion-MNIST | 0.4113 | 0.2061 | 0.2448 |

### 完整版（Mean ± Std）

| Dataset | NMI | ARI | ACC |
|---------|-----|-----|-----|
| MNIST | 0.1806 ± 0.0009 | 0.0878 ± 0.0023 | 0.2094 ± 0.0008 |
| Fashion-MNIST | 0.4113 ± 0.0392 | 0.2061 ± 0.0503 | 0.2448 ± 0.0449 |

---

## 📊 Excel/Sheets 复制格式

```
Dataset	NMI_Mean	NMI_Std	ARI_Mean	ARI_Std	ACC_Mean	ACC_Std	N_Seeds
MNIST	0.1806	0.0009	0.0878	0.0023	0.2094	0.0008	3
Fashion-MNIST	0.4113	0.0392	0.2061	0.0503	0.2448	0.0449	2
```

---

## 🎯 关键观察

1. **MNIST 表现差但稳定**
   - NMI 仅 0.18（接近随机）
   - 但方差极小（± 0.001），说明实现稳定
   - 可能是参数设置不适合 MNIST

2. **Fashion-MNIST 表现中等但不稳定**
   - NMI 达到 0.41（明显好于 MNIST）
   - 但方差较大（± 0.04），seed 间差异显著
   - Seed 1 (0.45) 比 Seed 0 (0.37) 好很多

3. **建议**
   - ✅ 补充 Fashion-MNIST seed=2 实验
   - ✅ 尝试增加训练轮数（10→50 epochs）
   - ✅ 尝试增加学习率（eta: 0.01→0.1）

---

## 🔧 补充 Fashion-MNIST seed=2

如需补充实验：

```bash
cd /hy-tmp/clustering
python scripts/run_baseline.py --config configs/softhebb.yaml --dataset fashion_mnist --seed 2
```

预计时间：5-15 分钟
