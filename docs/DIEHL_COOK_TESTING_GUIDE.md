# Diehl & Cook Baseline 测试指南

## 📋 概述

**Diehl & Cook** baseline 是基于 BindsNET 实现的 STDP (Spike-Timing-Dependent Plasticity) 学习的脉冲神经网络。

**论文**: Diehl & Cook, "Unsupervised learning of digit recognition using spike-timing-dependent plasticity", Frontiers in Computational Neuroscience, 2015

**当前状态**: 🟡 **接口就绪，但需要完整训练实现**

---

## 🚀 测试方法

### 方法 1: 使用主入口（最简单）

```bash
cd /hy-tmp/clustering

# 运行 Diehl & Cook baseline
python run.py --baseline diehl_cook --dataset mnist --seed 0
```

**注意**: 
- ⚠️ 首次运行会使用**骨架实现**（随机权重），不是真正的 STDP 学习
- ⏱️ 运行时间: ~5-10 分钟（骨架版本）
- 💻 需要 GPU（推荐）或 CPU（较慢）

---

### 方法 2: 使用配置文件

```bash
# 使用预定义配置
python run.py --config configs/diehl_cook.yaml

# 覆盖配置参数
python run.py --config configs/diehl_cook.yaml --seed 42
```

---

### 方法 3: 完整 BindsNET 训练（推荐用于正式实验）

```bash
# 进入 baseline 目录
cd baselines/diehl_cook

# 快速测试 (1000 样本，1 epoch)
python train.py \
    --train --extract \
    --n_train 1000 \
    --n_epochs 1 \
    --device cpu \
    --output_dir ../../outputs/diehl_cook

# 完整训练 (60000 样本，1 epoch)
python train.py \
    --train --extract \
    --n_train 60000 \
    --n_epochs 1 \
    --device cuda \
    --output_dir ../../outputs/diehl_cook

# 查看帮助
python train.py --help
```

**参数说明**:
- `--train`: 训练网络
- `--extract`: 训练后提取特征
- `--n_train`: 训练样本数（默认 60000）
- `--n_epochs`: 训练轮数（通常 1 轮足够）
- `--device`: `cpu` 或 `cuda`
- `--output_dir`: 输出目录

---

## 📊 预期结果

### 骨架实现（当前版本）

```
实现状态: 占位符（随机投影）
NMI:       ~0.30-0.40 (低于 FlyHash)
ARI:       ~0.20-0.30
ACC:       ~0.40-0.50
运行时间:   ~5-10 分钟
```

**说明**: 骨架版本使用随机权重，不是真正的 STDP 学习，所以性能较差。

---

### 完整 BindsNET 实现（预期）

```
实现状态: 完整 STDP 训练
NMI:       0.60-0.70 (预期)
ARI:       0.50-0.60 (预期)
ACC:       0.65-0.75 (预期)
训练时间:   ~1-2 小时 (GPU)
           ~5-10 小时 (CPU)
编码时间:   ~10-20 分钟
```

**说明**: 完整实现需要通过 BindsNET 进行 STDP 训练，性能应该显著优于随机方法。

---

## 🔧 当前实现状态

### ✅ 已实现

1. **接口完整**
   - `DiehlCookEncoder` 类符合 `BaseEncoder` 接口
   - 可以正常调用 `fit()` 和 `encode()`
   - 集成到主 pipeline

2. **骨架实现**
   - 使用随机投影作为占位符
   - 可以完成端到端测试
   - 输出格式正确

3. **配置文件**
   - `configs/diehl_cook.yaml` 完整
   - 参数设置合理

4. **训练脚本**
   - `baselines/diehl_cook/train.py` 包含完整框架
   - BindsNET 网络构建代码

### 🚧 待完成

1. **完整 STDP 训练**
   - ❌ BindsNET 网络训练循环（部分实现）
   - ❌ Poisson 编码图像输入
   - ❌ 权重保存和加载
   - ❌ 特征提取逻辑

2. **性能优化**
   - ❌ GPU 加速
   - ❌ 批处理
   - ❌ 检查点保存

---

## 🎯 如何测试

### 场景 1: 快速验证接口（推荐开始）

**目的**: 验证 pipeline 集成和接口正确性

```bash
# 1. 测试骨架实现
python run.py --baseline diehl_cook --dataset mnist

# 2. 查看结果
cat outputs/results/diehl_cook_mnist_seed0.json
```

**预期**:
- ✅ 程序正常运行，无报错
- ✅ 生成结果文件
- ⚠️ 性能较低（NMI ~0.3-0.4），这是正常的
- ⚠️ 会看到警告："This is a skeleton implementation!"

**时间**: ~5-10 分钟

---

### 场景 2: BindsNET 依赖测试

**目的**: 验证 BindsNET 是否正确安装

```bash
# 测试 BindsNET 导入
python -c "import bindsnet; print(f'BindsNET {bindsnet.__version__} OK')"

# 测试 PyTorch + CUDA（如果有 GPU）
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 测试训练脚本的帮助
cd baselines/diehl_cook
python train.py --help
```

**预期输出**:
```
BindsNET 0.2.7 OK
PyTorch 1.10.0
CUDA: True
```

---

### 场景 3: 小规模 STDP 训练测试（如果时间允许）

**目的**: 测试完整的 STDP 训练流程

```bash
# 快速训练测试 (1000 样本，~10-15分钟)
cd baselines/diehl_cook

python train.py \
    --train \
    --extract \
    --n_train 1000 \
    --n_epochs 1 \
    --n_neurons 100 \
    --device cpu \
    --output_dir ../../outputs/diehl_cook_test

# 查看输出
ls -lh ../../outputs/diehl_cook_test/
```

**预期**:
- ✅ 训练过程有进度显示
- ✅ 生成权重文件（`.pt` 或 `.pkl`）
- ✅ 生成编码特征文件
- ⚠️ 性能可能还是不高（训练样本太少）

**时间**: ~10-15 分钟（CPU）

---

### 场景 4: 完整训练（正式实验用）

**目的**: 获得最佳性能的 Diehl & Cook baseline

```bash
# 完整训练 (需要 GPU，~1-2 小时)
cd baselines/diehl_cook

python train.py \
    --train \
    --extract \
    --n_train 60000 \
    --n_epochs 1 \
    --n_neurons 400 \
    --device cuda \
    --output_dir ../../outputs/diehl_cook_full \
    --save_interval 10000

# 使用训练好的模型评估
cd ../..
python run.py \
    --baseline diehl_cook \
    --dataset mnist \
    --seed 0
```

**时间**: 
- 训练: ~1-2 小时（GPU）或 ~5-10 小时（CPU）
- 编码: ~10-20 分钟
- 聚类: ~5-10 分钟

---

## 🐛 常见问题

### Q1: 提示 "BindsNET not installed"

**A**: 安装 BindsNET:

```bash
# 方法 1: 从 GitHub（推荐）
pip install git+https://github.com/BindsNET/bindsnet.git

# 方法 2: 从 PyPI
pip install bindsnet>=0.2.7

# 验证安装
python -c "import bindsnet; print(bindsnet.__version__)"
```

如果遇到依赖问题，查看:
- `docs/INSTALL.md` - 详细安装指南
- `docs/VERSION_STATUS.md` - 版本兼容性
- `docs/INSTALLATION_QUICK_FIXES.md` - 常见问题修复

---

### Q2: 提示 "torch._six" 错误

**A**: NumPy/PyTorch 版本不兼容:

```bash
# 卸载并重装兼容版本
pip uninstall numpy torch torchvision -y

# 安装兼容版本
pip install numpy==1.26.4
pip install torch==1.10.0
pip install git+https://github.com/BindsNET/bindsnet.git
```

---

### Q3: "This is a skeleton implementation!" 警告

**A**: 这是正常的！当前 `encoder.py` 使用骨架实现。

**解决方案**:
1. **接受骨架版本**（用于快速测试和接口验证）
2. **使用完整训练** (`train.py`，需要更多时间）

---

### Q4: 训练非常慢

**A**: 这是 SNN 训练的正常现象。优化方法：

1. **减少训练样本**:
   ```bash
   --n_train 1000  # 从 60000 减少到 1000
   ```

2. **减少神经元数量**:
   ```bash
   --n_neurons 100  # 从 400 减少到 100
   ```

3. **使用 GPU**:
   ```bash
   --device cuda
   ```

4. **减少仿真时间**:
   ```bash
   --sim_time 250  # 从 350ms 减少到 250ms
   ```

---

### Q5: CUDA out of memory

**A**: GPU 内存不足:

```bash
# 使用 CPU（较慢但不受内存限制）
python train.py --device cpu --n_train 10000

# 或减少批处理大小
python train.py --batch_size 1 --device cuda
```

---

## 📊 性能对比

| Baseline | 训练时间 | NMI (预期) | ARI (预期) | ACC (预期) | 状态 |
|----------|---------|-----------|-----------|-----------|------|
| **FlyHash** | 无需训练 | 0.545 | 0.408 | 0.579 | ✅ 完成 |
| **Diehl & Cook (骨架)** | 无需训练 | 0.30-0.40 | 0.20-0.30 | 0.40-0.50 | ✅ 可测试 |
| **Diehl & Cook (完整)** | ~1-2小时 | 0.60-0.70 | 0.50-0.60 | 0.65-0.75 | 🚧 待实现 |

---

## 🎯 推荐测试流程

### 第 1 步: 快速验证（必做）✅

```bash
# 测试骨架实现
python run.py --baseline diehl_cook --dataset mnist
```

**目的**: 确认接口正确，pipeline 集成成功

**时间**: ~5 分钟

---

### 第 2 步: 依赖检查（推荐）

```bash
# 检查 BindsNET
python -c "import bindsnet; print('BindsNET OK')"

# 检查 PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**目的**: 确保环境配置正确

**时间**: ~1 分钟

---

### 第 3 步: 小规模训练（可选）

```bash
# 如果有时间，测试真实的 STDP 训练
cd baselines/diehl_cook
python train.py --train --extract --n_train 1000 --device cpu
```

**目的**: 验证完整训练流程

**时间**: ~10-15 分钟

---

### 第 4 步: 完整实验（正式使用）

```bash
# 在有 GPU 的机器上运行
python train.py --train --extract --n_train 60000 --device cuda
```

**目的**: 获得最佳性能结果

**时间**: ~1-2 小时

---

## 📝 检查清单

测试前检查:

- [ ] BindsNET 已安装
- [ ] PyTorch 已安装（GPU 可选）
- [ ] NumPy 版本兼容（< 2.0）
- [ ] 配置文件存在：`configs/diehl_cook.yaml`
- [ ] MNIST 数据集已下载

测试项目:

- [ ] 骨架版本可运行：`python run.py --baseline diehl_cook`
- [ ] 生成结果文件：`outputs/results/diehl_cook_*.json`
- [ ] 训练脚本可调用：`python baselines/diehl_cook/train.py --help`
- [ ] （可选）完整训练成功

---

## 🚀 下一步

### 如果骨架版本测试成功：

1. ✅ **继续测试其他 baseline**（如 SoftHebb）
2. ✅ **对比 FlyHash vs Diehl & Cook（骨架）性能**
3. ⏸️ **完整 STDP 实现**（如果需要更好性能）

### 如果需要完整实现：

1. 📝 完善 `train.py` 中的训练循环
2. 🔧 实现权重保存和加载
3. 📊 运行完整训练并评估
4. 📈 对比完整版本性能

---

## 📚 相关文档

- `README.md` - 项目主文档
- `docs/BINDSNET_INTEGRATION.md` - BindsNET 集成详情
- `docs/VERSION_STATUS.md` - 依赖版本状态
- `baselines/diehl_cook/README.md` - Diehl & Cook 详细说明
- `PROJECT_SUMMARY.md` - 项目总结

---

**总结**: 

✅ **可以立即测试** Diehl & Cook 的骨架实现  
🔧 **完整 STDP 训练**需要更多开发时间  
⚡ **推荐先测试骨架版本**，验证接口和集成

**快速开始**:
```bash
python run.py --baseline diehl_cook --dataset mnist
```
