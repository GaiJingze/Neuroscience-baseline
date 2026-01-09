# Installation Quick Fixes

快速解决安装问题的指南。

## 🚨 Quick Problem Solver

### 问题：BindsNET 版本找不到

**错误信息:**
```
ERROR: No matching distribution found for bindsnet>=0.3.1
```

**原因:** PyPI 上最新版本只有 0.2.7

**快速修复:**
```bash
# 修改 requirements.txt 中的版本
# bindsnet>=0.3.1  改为  bindsnet>=0.2.7

# 然后重新安装
pip install -r requirements.txt
```

**状态:** ✅ 已修复（requirements.txt 已更新）

---

### 问题：torch._six 模块找不到

**错误信息:**
```
ModuleNotFoundError: No module named 'torch._six'
```

**原因:** BindsNET 0.2.7 太老，不兼容 PyTorch 2.x

**快速修复 (3个选项):**

#### 选项 1: 使用 FlyHash (推荐用于快速测试)
```bash
# FlyHash 不需要 BindsNET，直接可用
python run.py --baseline flyhash
```

#### 选项 2: 从 GitHub 安装 BindsNET (推荐用于生产)
```bash
pip uninstall bindsnet
pip install git+https://github.com/BindsNET/bindsnet.git
```

#### 选项 3: 降级 PyTorch (最稳定)
```bash
pip install torch==1.13.1 torchvision==0.14.1
pip install bindsnet==0.2.7
```

**状态:** ⚠️ 需要手动选择一个方案

---

### 问题：NumPy 版本冲突

**错误信息:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**原因:** 系统有 NumPy 2.x，但 BindsNET 需要 1.x

**快速修复:**
```bash
pip install 'numpy>=1.21.0,<2.0.0'
```

**状态:** ✅ 已修复（requirements.txt 已指定版本）

---

## 🎯 推荐的安装流程

### 方案 A: 快速测试（无 BindsNET）

```bash
# 1. 安装基础依赖
pip install numpy scipy scikit-learn matplotlib torch torchvision

# 2. 测试 FlyHash
python run.py --test
python run.py --baseline flyhash

# ✅ 5分钟内完成
```

**适用于:**
- 快速验证 pipeline
- 测试 FlyHash baseline
- 不需要 SNN 模拟

---

### 方案 B: 完整安装（GitHub BindsNET）

```bash
# 1. 安装基础依赖
pip install -r requirements.txt

# 2. 从 GitHub 安装 BindsNET
pip uninstall bindsnet -y
pip install git+https://github.com/BindsNET/bindsnet.git

# 3. 验证安装
python -c "import bindsnet; print('BindsNET OK')"

# 4. 测试
python run.py --test
python run.py --baseline diehl_cook

# ⏱️ 10-15分钟
```

**适用于:**
- 需要 Diehl & Cook baseline
- 想要最新的 BindsNET 代码
- 可以接受开发版本

---

### 方案 C: 稳定安装（降级 PyTorch）

```bash
# 1. 降级 PyTorch
pip install torch==1.13.1 torchvision==0.14.1

# 2. 安装其他依赖
pip install 'numpy>=1.21.0,<2.0.0'
pip install scipy scikit-learn matplotlib bindsnet==0.2.7

# 3. 验证
python -c "import bindsnet; print('OK')"

# 4. 测试
python run.py --baseline diehl_cook

# ⏱️ 10分钟
```

**适用于:**
- 需要稳定的环境
- 不需要最新 PyTorch 特性
- 生产环境部署

---

## 🔧 故障排除命令

### 检查当前版本
```bash
python -c "
import sys
try:
    import numpy; print(f'NumPy: {numpy.__version__}')
except: print('NumPy: NOT INSTALLED')

try:
    import torch; print(f'PyTorch: {torch.__version__}')
except: print('PyTorch: NOT INSTALLED')

try:
    import bindsnet; print(f'BindsNET: {bindsnet.__version__}')
except Exception as e: print(f'BindsNET: ERROR - {e}')
"
```

### 完全重装
```bash
# 卸载所有相关包
pip uninstall numpy torch torchvision bindsnet -y

# 重新安装（选择一个方案）
pip install -r requirements.txt  # 然后按上面的方案修复
```

### 使用虚拟环境（推荐）
```bash
# 创建干净的环境
python -m venv venv_clustering
source venv_clustering/bin/activate  # Linux/Mac
# 或 venv_clustering\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

---

## 📊 当前环境状态

运行此命令查看状态：
```bash
make check-env
```

或手动检查：
```bash
cat docs/VERSION_STATUS.md
```

---

## 🆘 仍然有问题？

### 1. 查看详细文档
```bash
cat docs/TROUBLESHOOTING.md      # 完整故障排除指南
cat docs/VERSION_STATUS.md       # 版本兼容性矩阵
cat docs/BINDSNET_INTEGRATION.md # BindsNET 集成详情
```

### 2. 运行诊断
```bash
python run.py --test  # 快速诊断
make check-env        # 环境检查
```

### 3. 使用无依赖的 baseline
```bash
# FlyHash 完全不依赖 BindsNET
python run.py --baseline flyhash
```

---

## ✅ 成功标志

安装成功后，应该能看到：

```bash
$ python run.py --list

Available Baselines
===================

✅ flyhash
   Status: Ready
   ...

✅ diehl_cook  # 如果 BindsNET 正常工作
   Status: Ready
   ...
```

```bash
$ python run.py --test

✅ All tests passed
```

---

**更新日期**: 2026-01-09  
**适用版本**: clustering pipeline v2.0  
**维护者**: Jingze Gai
