# Testing Summary - Quick Reference

## 🎯 测试单个Baseline

```bash
# 方法1: Python脚本
python scripts/test_baseline.py flyhash

# 方法2: Makefile
make test-baseline BASELINE=flyhash

# 带选项
python scripts/test_baseline.py flyhash --dataset mnist --seeds 0 1 2
```

## 🎯 测试多个Baseline

```bash
# 测试指定的几个
python scripts/test_baseline.py flyhash diehl_cook

# 测试所有可用的
python scripts/test_baseline.py --all

# 快速测试所有（1个seed）
make test-baselines-quick

# 完整测试所有（3个seed）
make test-baselines-full
```

## 🎯 批量测试

```bash
# 快速批量测试（1 seed, 1 dataset）
bash scripts/batch_test.sh --quick
# 或
make batch-test-quick

# 完整批量测试（3 seeds, 2 datasets）
bash scripts/batch_test.sh --full
# 或
make batch-test-full

# 自定义批量测试
bash scripts/batch_test.sh \
    --baselines 'flyhash diehl_cook' \
    --datasets 'mnist fashion_mnist' \
    --seeds '0 1 2'
```

## 🎯 查看可用Baseline

```bash
python scripts/test_baseline.py --list
```

## 🎯 常用场景

### 场景1：开发时快速验证
```bash
python scripts/test_baseline.py flyhash --seeds 0
```

### 场景2：多个种子测试可重现性
```bash
python scripts/test_baseline.py flyhash --seeds 0 1 2
```

### 场景3：对比多个方法
```bash
python scripts/test_baseline.py flyhash diehl_cook --seeds 0
```

### 场景4：完整评估（论文用）
```bash
bash scripts/batch_test.sh --full
```

### 场景5：强制重新编码
```bash
python scripts/test_baseline.py flyhash --force
```

## 🎯 所有命令速查

| 命令 | 用途 | 时间 |
|------|------|------|
| `python scripts/test_baseline.py flyhash` | 测试单个baseline | 5-10分钟 |
| `python scripts/test_baseline.py --all` | 测试所有baseline | 15-30分钟 |
| `python scripts/test_baseline.py flyhash --seeds 0 1 2` | 多seed测试 | 15-30分钟 |
| `bash scripts/batch_test.sh --quick` | 快速批量测试 | 10-20分钟 |
| `bash scripts/batch_test.sh --full` | 完整批量测试 | 30-60分钟 |
| `make test-baseline BASELINE=flyhash` | Makefile单个测试 | 5-10分钟 |
| `make test-baselines-quick` | Makefile快速测试 | 10-20分钟 |
| `make batch-test-full` | Makefile完整批量 | 30-60分钟 |

## 🎯 输出文件位置

```
outputs/
├── codes/                      # 特征码
│   └── {baseline}/{dataset}/
│       ├── pre_code_seed0.npy
│       └── code_seed0.npy
├── results/                    # 评测结果
│   └── {baseline}_{dataset}_seed0.json
├── logs/                       # 日志
│   └── {baseline}_{dataset}_seed0.log
└── batch_results/              # 批量测试结果
    └── batch_test_20260109.txt
```

## 🎯 Makefile快捷命令

```bash
make help                      # 显示所有命令
make test-baseline BASELINE=flyhash  # 测试单个
make test-baselines-quick      # 快速测试所有
make test-baselines-full       # 完整测试所有
make batch-test-quick          # 快速批量
make batch-test-full           # 完整批量
make status                    # 查看状态
```

## 🎯 故障排除

**问题**: Baseline未找到
```bash
# 查看可用baseline
python scripts/test_baseline.py --list
```

**问题**: 测试太慢
```bash
# 使用快速模式
bash scripts/batch_test.sh --quick
```

**问题**: 内存不足
```bash
# 编辑配置文件，减少参数
vim configs/baseline.yaml
# 设置: device: "cpu", n_neurons: 200
```

---

**详细文档**: 
- `BASELINE_TESTING.md` - 完整测试指南
- `TEST_GUIDE.md` - 测试详细说明
- `TROUBLESHOOTING.md` - 故障排除
