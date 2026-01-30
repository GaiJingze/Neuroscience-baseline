# Diehl & Cook修复总结

## 🐛 发现的问题

1. **根本原因**: Poisson编码需要 [0, 255] 范围的输入，但我们归一化到了 [0, 1]
   - 结果：输入神经元完全不发放脉冲
   - 后果：所有编码都是零向量，聚类失败

2. **次要问题**: `network.train(False)` 可能影响BindsNET的行为
   - 已移除，改为依赖BindsNET的learning参数

## ✅ 已修复

1. 修改了 `baselines/diehl_cook/encoder.py`：
   - 训练阶段：保持数据在 [0, 255] 范围
   - 编码阶段：保持数据在 [0, 255] 范围
   - 移除了 `network.train(False)` 调用

## 📊 修复后的状态

- ✅ 神经元现在能够发放脉冲
- ✅ 所有100个神经元都是活跃的
- ⚠️  编码多样性仍然较低（需要更多训练/参数调整）

## 🚀 下一步

### 选项 1: 重新运行所有实验（推荐）

```bash
cd /hy-tmp/clustering

# 清除旧结果
python scripts/clear_cache.py --baseline diehl_cook --yes
rm -rf outputs/codes/diehl_cook outputs/results/diehl_cook_*.json outputs/logs/diehl_cook_*.log

# 运行完整实验
python scripts/run_diehl_cook_full.py
```

### 选项 2: 快速验证修复

```bash
cd /hy-tmp/clustering

# 只运行MNIST, seed=0来验证
python scripts/run_baseline.py --config configs/diehl_cook.yaml --dataset mnist --seed 0

# 检查结果
python scripts/diagnose_diehl_cook.py
python scripts/collect_diehl_cook_results.py
```

### 选项 3: 调整参数后再跑

编辑 `configs/diehl_cook.yaml` 来调整参数：
- `simulation_time`: 增加到500或更高（更多时间让神经元发放）
- `n_neurons`: 增加到800（更多神经元学习不同特征）
- `n_train_samples`: 保持为null（使用所有60000训练样本）

## ⏱️ 预计时间

- 单个实验（MNIST, 1 seed）: ~30-60分钟
- 完整实验（2数据集 × 3 seeds）: ~4-6小时

## 📝 注意事项

- 之前的实验结果无效（全是0），需要重跑
- 修复后应该能看到非零的聚类指标
- 如果结果仍然不好，可能需要调整超参数
