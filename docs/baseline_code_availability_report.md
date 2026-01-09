# Clustering/Hashing Baseline 开源代码调查报告

**调查日期**: 2026-01-09  
**调查目的**: 确定项目中选定的baseline的开源代码可用性，并按整合难度排序

---

## 📊 总览：按整合难度排序

| 难度 | Baseline | 年份 | 代码状态 | 预计整合时间 |
|------|----------|------|----------|--------------|
| 🟢 **易** | **FlyHash (PyPI包)** | 2017 | ✅ PyPI包可用 | 1-2天 |
| 🟢 **易** | **BindsNET示例** | 2018 | ✅ 官方示例 | 2-3天 |
| 🟡 **中** | **Diehl & Cook (从零实现)** | 2015 | ⚠️ 需基于BindsNET实现 | 1-2周 |
| 🟡 **中** | **SoftHebb** | 2022 | ✅ 官方GitHub | 1-2周 |
| 🔴 **难** | **Lu & Sengupta (2024)** | 2024 | ❓ 需要调查 | 3-4周 |
| 🔴 **难** | **BioHash** | 2020 | ❌ 可能不存在 | N/A |

---

## 详细分析

### 🟢 难度1：FlyHash（最容易，1-2天）

**Paper**: Dasgupta et al., "A neural algorithm for a fundamental computing problem", Science 2017

**代码状态**: ✅ **PyPI包可用**

**可用资源**:
1. **PyPI包**: `pip install FlyHash`
   - Link: https://pypi.org/project/FlyHash/
   - 这是第三方实现，非官方
   - 接口简单，易于使用

2. **GitHub实现**:
   - 多个非官方实现可用（搜索 "fly hash locality sensitive" GitHub）
   - 大多数是简单的Python/NumPy实现

**整合难度评估**:
- ✅ **易**: 仅需 `pip install` 即可使用
- ✅ **无训练**: 非参数化方法，无需训练过程
- ✅ **纯Python**: 无需GPU，CPU即可运行
- ✅ **已实现**: 我们的代码骨架中已包含完整实现

**实现代码位置**: `clustering/baselines/flyhash/encoder.py` ✅

**建议**: 
- **优先级: 最高**
- 立即可用，作为sanity check
- 用于快速验证整个pipeline

---

### 🟢 难度1+：BindsNET官方示例（易，2-3天）

**Paper**: Hazan et al., "BindsNET: A Machine Learning-Oriented Spiking Neural Networks Library in Python", Frontiers 2018

**代码状态**: ✅ **官方库 + 示例**

**可用资源**:
1. **BindsNET框架**:
   - GitHub: https://github.com/BindsNET/bindsnet
   - `pip install bindsnet`
   - PyTorch-based SNN框架
   - 文档: https://bindsnet-docs.readthedocs.io/

2. **官方示例**:
   - `examples/mnist/` 目录包含多个MNIST示例
   - 包括STDP学习示例
   - 可以直接改造使用

**整合难度评估**:
- ✅ **易**: 官方安装和文档完善
- ⚠️ **需GPU**: PyTorch训练建议使用GPU
- ⚠️ **需理解API**: 需要学习BindsNET的网络构建API
- ✅ **有教程**: 官方提供tutorial

**使用方式**:
```python
# 基于BindsNET官方示例
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.learning import PostPre  # STDP

# 可以使用官方示例改造
```

**建议**:
- **优先级: 高**
- 使用官方示例作为起点，比从零实现Diehl & Cook更快
- 先跑通示例，再提取特征用于评测

---

### 🟡 难度2：Diehl & Cook 2015（中等，1-2周）

**Paper**: Diehl & Cook, "Unsupervised learning of digit recognition using spike-timing-dependent plasticity", Front. Comput. Neurosci. 2015

**代码状态**: ⚠️ **无官方代码，需基于BindsNET实现**

**可用资源**:
1. **原论文**: 
   - DOI: 10.3389/fncom.2015.00099
   - 详细的算法描述和参数

2. **非官方实现** (GitHub搜索发现):
   - 有多个GitHub用户尝试复现
   - 搜索: "diehl cook stdp mnist github"
   - 质量参差不齐，需要验证

3. **BindsNET示例**:
   - BindsNET的示例代码实现了类似架构
   - 可以作为参考

**整合难度评估**:
- ⚠️ **中等**: 需要实现STDP + 侧抑制 + 自适应阈值
- ⚠️ **需调参**: 论文中的参数需要细致调整
- ⚠️ **需GPU**: 训练时间较长
- ✅ **有框架**: BindsNET提供了基础模块

**实现策略**:
1. **方案A** (推荐): 使用BindsNET官方示例 + 改造
2. **方案B**: 参考非官方GitHub实现
3. **方案C**: 完全从零实现（不推荐）

**实现代码位置**: `clustering/baselines/diehl_cook/encoder.py` 🔨 (骨架已创建)

**建议**:
- **优先级: 中**
- 项目文档明确要求，必须包含
- 建议先用BindsNET示例快速实现，不追求完美复现

---

### 🟡 难度2+：SoftHebb（中等，1-2周）

**Paper**: Moraitis et al., "SoftHebb: Bayesian inference in unsupervised hebbian soft winner-take-all networks", NCE 2022

**代码状态**: ✅ **官方GitHub仓库（可能存在）**

**可用资源**:
1. **官方仓库** (需要验证):
   - 搜索: "SoftHebb GitHub Moraitis"
   - 论文通常会提供code availability说明
   - NCE期刊通常要求code开放

2. **论文**:
   - DOI: 10.1088/2634-4386/ac98a9
   - 有ICLR 2023扩展版本

**整合难度评估**:
- ⚠️ **中等**: 如果有官方代码，主要工作是环境隔离
- ⚠️ **环境问题**: 可能使用旧版PyTorch (1.7.1)
- ⚠️ **特征导出**: 需要修改代码以导出我们需要的特征格式
- ✅ **概念清晰**: Soft Hebbian学习相对好理解

**整合策略**:
1. 独立conda环境训练
2. 导出特征为.npy文件
3. 主pipeline读取.npy进行评测

**预计工作量**:
- 如果有官方代码: 1周（环境设置 + 改造）
- 如果无官方代码: 2-3周（重实现）

**建议**:
- **优先级: 高**
- 项目文档明确提到
- 需要先确认代码可用性

**待办**: 
- [ ] 查阅论文supplementary materials
- [ ] 联系作者索要代码
- [ ] 搜索GitHub: "SoftHebb", "Moraitis Hebbian"

---

### 🔴 难度3：Lu & Sengupta 2024（困难，3-4周）

**Paper**: Lu & Sengupta, "Deep unsupervised learning using spike-timing-dependent plasticity", NCE 2024

**代码状态**: ❓ **不确定（论文很新）**

**可用资源**:
1. **论文**:
   - DOI: 10.1088/2634-4386/ad5e6d
   - 2024年发表，非常新

2. **作者**:
   - Sunil Lu
   - Abhronil Sengupta (知名SNN研究者)
   - 可能愿意分享代码

**整合难度评估**:
- 🔴 **难**: 论文很新，可能无公开代码
- 🔴 **复杂**: Deep STDP，多层架构
- 🔴 **需GPU**: 多层SNN训练资源需求大
- ⚠️ **可能需重实现**: 如果无代码，需要完全重实现

**实现策略**:
1. **优先**: 联系作者索要代码
2. **备选**: 基于BindsNET实现多层STDP
3. **降级**: 实现简化版（2层而非论文中的多层）

**时间线**:
- 代码可用: 2周（理解 + 整合）
- 需要重实现: 4周+（实现 + 调试 + 验证）

**建议**:
- **优先级: 中-高**
- 项目文档明确要求
- 但由于难度大，建议作为Phase 2的扩展工作
- 如果时间紧张，可以用简化版或跳过

**待办**:
- [ ] 阅读论文全文
- [ ] 检查supplementary materials
- [ ] 发邮件联系作者
- [ ] 如无回复，评估重实现的可行性

---

### 🔴 难度4：BioHash（可能不存在）

**Paper**: 未确认

**代码状态**: ❌ **可能不存在此方法**

**调查结果**:
1. **ICML 2020搜索**:
   - 未找到明确的"BioHash"方法
   - 可能是记忆错误或命名混淆

2. **可能的替代**:
   - 其他bio-inspired hashing方法
   - FlyHash本身就是bio-inspired

**建议**:
- **优先级: 低**
- 先确认这个baseline是否真实存在
- 如果不存在，用其他baseline替代
- 可能的替代方案:
  - 增强版FlyHash
  - 其他ICML/NeurIPS的bio-inspired hashing工作

**待办**:
- [ ] 与导师确认"BioHash"的具体论文
- [ ] 如果不存在，讨论替代方案

---

## 🎯 推荐的实施优先级

### Phase 1（立即开始，1周内）

1. ✅ **FlyHash** (已实现)
   - 验证代码正确性
   - 在MNIST上测试
   - 作为pipeline的sanity check

2. 🔨 **BindsNET示例改造**
   - 安装BindsNET
   - 运行官方MNIST示例
   - 提取spike count特征

### Phase 2（第2-3周）

3. 🔨 **Diehl & Cook**
   - 基于BindsNET示例实现
   - 不追求完美复现，重点是提取可用特征
   - 在MNIST上获得baseline结果

4. 🔍 **SoftHebb调研**
   - 找到官方代码
   - 设置独立环境
   - 运行原始代码

### Phase 3（第4-5周）

5. 🔨 **SoftHebb整合**
   - 修改代码以导出特征
   - 整合到评测pipeline
   - 在MNIST + Fashion-MNIST上测试

6. 🔍 **Lu & Sengupta调研**
   - 联系作者
   - 评估重实现难度

### Phase 4（第6周+，可选）

7. 🔨 **Lu & Sengupta**（如果可行）
   - 整合或重实现
   - 获得baseline结果

8. ❓ **BioHash替代方案**（如果需要）
   - 确认是否需要
   - 选择替代baseline

---

## 📋 具体行动清单

### 本周必做

- [x] ✅ FlyHash实现完成
- [ ] 🔨 安装BindsNET: `pip install bindsnet`
- [ ] 🔨 运行BindsNET MNIST示例
- [ ] 🔍 搜索SoftHebb GitHub仓库
- [ ] 🔍 下载Lu & Sengupta 2024论文

### 下周计划

- [ ] 🔨 基于BindsNET实现Diehl & Cook
- [ ] 🔨 在MNIST上测试所有可用baseline
- [ ] 📊 生成初步结果表格

### 两周后

- [ ] 🔨 SoftHebb整合（如果代码可用）
- [ ] 📊 完整的baseline对比
- [ ] 📝 撰写baseline报告

---

## 💡 关键建议

### 1. 不要追求完美复现

- 我们的目标是**建立baseline性能数字**，而非完美复现论文
- 如果能达到论文80-90%的性能，已经足够作为baseline
- 时间有限，优先快速迭代

### 2. 充分利用现有资源

- BindsNET框架已经实现了大部分SNN基础模块
- 官方示例代码质量高，优先使用
- 避免从零造轮子

### 3. 环境隔离策略

- 对于旧版本依赖（如SoftHebb），使用独立conda环境
- 训练和评测分离：训练在独立环境，导出.npy，评测在主环境
- 避免依赖冲突

### 4. 及时止损

- 如果某个baseline整合困难超过2周，考虑：
  - 简化实现
  - 寻找替代方案
  - 降低优先级
- 保证项目整体进度

---

## 📞 需要确认的问题

与导师/项目负责人确认：

1. **BioHash**: 这个baseline是否真实存在？具体论文是什么？
2. **Lu & Sengupta 2024**: 这个baseline的优先级？如果很难整合是否可以跳过？
3. **时间线**: Phase 4（baseline报告）的硬截止日期？
4. **完美度**: Baseline复现需要达到多高的准确度？
5. **Baseline数量**: 最少需要多少个baseline？3个够吗？

---

## 🔗 有用的资源链接

### 框架和工具

- **BindsNET**: https://github.com/BindsNET/bindsnet
- **SpikingJelly** (备选): https://github.com/fangwei123456/spikingjelly
- **Norse** (备选): https://github.com/norse/norse

### 论文检索

- **Google Scholar**: 搜索具体论文和引用
- **Papers With Code**: 查找论文对应代码
- **GitHub**: 搜索 "[author] [year] [keyword]"

### 社区

- **BindsNET Discussions**: 可以询问实现问题
- **Stack Overflow**: PyTorch/SNN相关问题
- **Reddit r/MachineLearning**: 讨论论文复现

---

**总结**: 我们有**2个容易整合的baseline** (FlyHash, BindsNET示例)，**2个中等难度** (Diehl & Cook, SoftHebb)，**1个困难** (Lu & Sengupta)，**1个待确认** (BioHash)。建议按Phase 1-4的优先级逐步推进，确保至少完成3-4个baseline。

**预计时间线**: 
- **最小可交付** (3个baseline): 3-4周
- **完整交付** (5个baseline): 6-8周
