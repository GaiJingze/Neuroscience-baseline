# 聚类实验结果总结

## 📊 主要结果表格

### MNIST 数据集

| Baseline | NMI | ARI | ACC | Seeds |
|----------|-----|-----|-----|-------|
| **FlyHash** | **0.5494 ± 0.0345** | **0.4089 ± 0.0187** | **0.5748 ± 0.0129** | 3 |
| SoftHebb | 0.1806 ± 0.0009 | 0.0878 ± 0.0023 | 0.2094 ± 0.0008 | 3 |
| Diehl & Cook | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1135 ± 0.0000 | 2 ⚠️ |

### Fashion-MNIST 数据集

| Baseline | NMI | ARI | ACC | Seeds |
|----------|-----|-----|-----|-------|
| **FlyHash** | **0.5936 ± 0.0019** | **0.4128 ± 0.0133** | **0.5424 ± 0.0217** | 3 |
| SoftHebb | 0.4113 ± 0.0392 | 0.2061 ± 0.0503 | 0.2448 ± 0.0449 | 2 |
| Diehl & Cook | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 ± 0.0000 | 3 ⚠️ |

## 📈 性能排名

### 综合表现（按 NMI）

1. 🥇 **FlyHash** - NMI: 0.55-0.59 ✅ 最佳
2. 🥈 **SoftHebb** - NMI: 0.18-0.41 ⚠️ 中等（需要参数调整）
3. 🥉 **Diehl & Cook** - NMI: 0.00 ❌ 失败（有bug，需要重新训练）

## ⚠️ 重要说明

### Diehl & Cook (STDP-SNN)

**状态：** ❌ 当前结果无效

**问题：**
- 发现 Poisson 编码bug（输入范围错误）
- 导致神经元完全不发放脉冲
- 所有编码都是零向量

**已修复：**
- ✅ 修改了 `baselines/diehl_cook/encoder.py`
- ✅ 保持输入数据在 [0, 255] 范围

**需要重新运行：**
```bash
cd /hy-tmp/clustering
python scripts/clear_cache.py --baseline diehl_cook --yes
rm -rf outputs/codes/diehl_cook outputs/results/diehl_cook_*.json
python scripts/run_diehl_cook_full.py
```

**预计时间：** ~4-6 小时

## 📋 Excel 复制格式

```
Baseline	Dataset	NMI	ARI	ACC
FlyHash	mnist	0.5494	0.4089	0.5748
FlyHash	fashion_mnist	0.5936	0.4128	0.5424
SoftHebb	mnist	0.1806	0.0878	0.2094
SoftHebb	fashion_mnist	0.4113	0.2061	0.2448
```

## 🎯 结论

1. **FlyHash 表现最好**
   - 无需训练，快速可靠
   - 在两个数据集上都有稳定的高性能
   - NMI ~0.55，可以作为强基线

2. **SoftHebb 表现中等**
   - 需要训练（10 epochs）
   - MNIST 上表现较差（NMI ~0.18）
   - Fashion-MNIST 上表现尚可（NMI ~0.41）
   - 可能通过调参改善（增加 epochs、学习率）

3. **Diehl & Cook 需要重新实验**
   - 当前结果因 bug 而无效
   - 修复后需要重新训练（耗时较长）
   - 预期性能未知（原论文主要用于分类而非聚类）

## 📁 结果文件位置

- **详细结果：** `outputs/results/`
- **编码特征：** `outputs/codes/`
- **训练日志：** `outputs/logs/`
- **本总结：** `RESULTS_SUMMARY.md`
