# CUDA 设备错误修复指南

## ❌ 错误信息

```
RuntimeError: expected self and mask to be on the same device, 
but got mask on cpu and self on cuda:0
```

## 🔍 问题原因

这是一个典型的 PyTorch CUDA 设备不匹配错误：

- **网络模型**在 GPU (cuda:0) 上
- **输入数据**还在 CPU 上
- PyTorch 不允许 CPU 和 GPU 的张量直接运算

## ✅ 解决方案

### 已修复

在 `baselines/diehl_cook/train.py` 中，我们添加了：

```python
# 在训练循环中 (line ~158)
image = torch.from_numpy(image).float()
if image.max() > 1.0:
    image = image / 255.0

# ✅ 添加这一行：将数据移动到与网络相同的设备
image = image.to(device)

# 然后再进行编码
encoded = poisson(datum=image, time=int(time), dt=network.dt)
```

### 修复位置

修复了两个地方（每个地方修复两处）：

1. **训练函数** (`train_network`，line ~158-170)
   ```python
   image = image.to(device)  # 训练时移动到GPU
   encoded = poisson(datum=image, time=int(time), dt=network.dt)
   encoded = encoded.to(device)  # ✅ 确保编码数据也在GPU
   ```

2. **特征提取函数** (`extract_spike_counts`，line ~207-219)
   ```python
   image = image.to(device)  # 提取特征时移动到GPU
   encoded = poisson(datum=image, time=int(time), dt=network.dt)
   encoded = encoded.to(device)  # ✅ 确保编码数据也在GPU
   ```

**关键修复**: `poisson` 编码函数返回的张量也需要移到正确的设备！

---

## 🎯 现在可以重新运行

### 完整训练命令

```bash
cd /hy-tmp/clustering/baselines/diehl_cook

# 快速测试 (1000样本, CPU)
python train.py --train --extract \
    --n_train 1000 \
    --n_epochs 1 \
    --device cpu

# 完整训练 (60000样本, GPU)
python train.py --train --extract \
    --n_train 60000 \
    --n_epochs 1 \
    --device cuda
```

---

## 💡 理解设备管理

### PyTorch 设备原则

在 PyTorch 中，**所有参与计算的张量必须在同一设备上**：

```python
# ❌ 错误：张量在不同设备
x = torch.randn(10).to('cpu')
y = torch.randn(10).to('cuda')
z = x + y  # RuntimeError!

# ✅ 正确：都在同一设备
x = torch.randn(10).to('cuda')
y = torch.randn(10).to('cuda')
z = x + y  # OK!
```

### 常见场景

#### 1. 模型在 GPU，数据在 CPU

```python
model = Model().to('cuda')  # 模型在GPU

for data in dataloader:
    data = data.to('cuda')  # ✅ 数据也移到GPU
    output = model(data)
```

#### 2. 检查当前设备

```python
# 检查张量所在设备
print(x.device)  # cuda:0 或 cpu

# 检查模型所在设备
print(next(model.parameters()).device)
```

#### 3. 统一设备管理

```python
# 推荐做法：统一使用一个变量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model().to(device)
data = data.to(device)
```

---

## 🐛 其他常见 CUDA 错误

### 1. CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**解决方案**：

```bash
# 方案1: 减少批处理大小
python train.py --batch_size 1

# 方案2: 减少样本数
python train.py --n_train 1000

# 方案3: 减少神经元数量
python train.py --n_neurons 100

# 方案4: 使用CPU
python train.py --device cpu
```

### 2. 找不到 CUDA

```
RuntimeError: CUDA not available
```

**检查**：

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
```

**解决方案**：
- 确保安装了支持 CUDA 的 PyTorch
- 检查 NVIDIA 驱动是否正确安装

### 3. 设备 ID 错误

```
RuntimeError: Invalid device ordinal
```

**解决方案**：

```python
# 检查可用的 GPU 数量
n_gpus = torch.cuda.device_count()
print(f"Available GPUs: {n_gpus}")

# 使用正确的设备 ID (0 到 n_gpus-1)
device = torch.device('cuda:0')  # 使用第一个GPU
```

---

## 📋 调试检查清单

### 训练前检查

```python
import torch

# 1. 检查 CUDA 是否可用
assert torch.cuda.is_available(), "CUDA not available!"

# 2. 检查 PyTorch 版本
print(f"PyTorch: {torch.__version__}")

# 3. 检查 CUDA 版本
print(f"CUDA: {torch.version.cuda}")

# 4. 检查可用内存
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 运行时检查

```python
# 在训练循环中添加断言
def train_step(model, data, device):
    data = data.to(device)
    
    # 检查设备一致性
    assert data.device.type == device.type, f"Data on {data.device}, expected {device}"
    assert next(model.parameters()).device.type == device.type, "Model not on correct device"
    
    output = model(data)
    return output
```

---

## 🚀 性能优化建议

### 1. 固定内存（Pinned Memory）

```python
# 在 DataLoader 中使用
dataloader = DataLoader(
    dataset, 
    batch_size=32,
    pin_memory=True  # 加速 CPU -> GPU 传输
)
```

### 2. 异步传输

```python
# 使用 non_blocking 加速
data = data.to(device, non_blocking=True)
```

### 3. 避免频繁的 CPU <-> GPU 传输

```python
# ❌ 不好：每次都传输
for i in range(1000):
    x = data[i].to('cuda')
    output = model(x)
    result = output.cpu()  # 传回CPU

# ✅ 好：批量传输
data = data.to('cuda')  # 一次性传输所有数据
for i in range(1000):
    output = model(data[i])
results = outputs.cpu()  # 一次性传回
```

---

## 📝 代码模板

### 标准训练循环

```python
import torch
import torch.nn as nn

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 模型
model = MyModel().to(device)

# 训练循环
for epoch in range(n_epochs):
    for batch_data, batch_labels in dataloader:
        # 数据移到设备
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        # 前向传播
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 如需在CPU上处理，显式移回
        loss_value = loss.item()  # 自动移到CPU
        print(f"Loss: {loss_value:.4f}")
```

---

## ✅ 验证修复

运行以下命令验证问题已解决：

```bash
# 快速测试（5-10分钟）
cd /hy-tmp/clustering/baselines/diehl_cook
python train.py --train --n_train 100 --device cuda

# 如果成功，应该看到：
# Using device: cuda
# Building Diehl & Cook network...
# Training for 1 epoch(s)...
# Epoch 1/1
#   Sample 100/100 (label=...)
# Epoch 1 complete!
```

如果还有问题，尝试：

```bash
# 使用 CPU（慢但稳定）
python train.py --train --n_train 100 --device cpu
```

---

## 🎯 总结

### 问题
- ❌ 网络在 GPU，数据在 CPU，导致设备不匹配

### 修复
- ✅ 在数据编码前添加 `image = image.to(device)`

### 结果
- ✅ 现在可以在 GPU 上训练 Diehl & Cook baseline 了！

---

**快速开始**：

```bash
cd /hy-tmp/clustering/baselines/diehl_cook
python train.py --train --extract --n_train 1000 --device cuda
```

预计时间：~10-15 分钟 🚀
