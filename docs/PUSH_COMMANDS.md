# 推送命令指南

## ✅ Remote 已配置

```bash
origin  https://github.com/GaiJingze/Neuroscience-baseline.git (fetch)
origin  https://github.com/GaiJingze/Neuroscience-baseline.git (push)
```

## 🚀 推送步骤

### 方案 1：使用 Personal Access Token（推荐）

#### 第 1 步：创建 Token

1. 访问：https://github.com/settings/tokens
2. 点击 **"Generate new token (classic)"**
3. 设置：
   - Note: `Neuroscience-baseline push`
   - Expiration: 选择有效期（建议 90 days 或 No expiration）
   - ✅ 勾选 **repo** (完整的仓库访问权限)
4. 点击 **"Generate token"**
5. 复制生成的 token（只显示一次，记得保存！）

#### 第 2 步：推送

```bash
cd /hy-tmp/clustering
git push -u origin main
```

当提示输入凭据时：
- **Username**: `GaiJingze`
- **Password**: `<粘贴你的 token>`

#### 第 3 步（可选）：保存凭据

如果不想每次都输入，运行：

```bash
git config --global credential.helper store
```

下次推送时输入一次 token，之后会自动记住。

---

### 方案 2：使用 SSH Key

如果你已经有 SSH key 设置好：

```bash
cd /hy-tmp/clustering
git remote set-url origin git@github.com:GaiJingze/Neuroscience-baseline.git
git push -u origin main
```

#### 如果没有 SSH key，设置步骤：

```bash
# 1. 生成 SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 查看公钥
cat ~/.ssh/id_ed25519.pub

# 3. 复制公钥内容，添加到 GitHub:
#    https://github.com/settings/keys
#    点击 "New SSH key"，粘贴公钥

# 4. 测试连接
ssh -T git@github.com

# 5. 推送
git push -u origin main
```

---

### 方案 3：快速脚本

我已经为你准备了一个快速推送脚本：

```bash
cd /hy-tmp/clustering
./quick_push_neuroscience.sh
```

这个脚本会：
1. 检查 remote 配置
2. 尝试推送
3. 提供详细的错误提示

---

## 📝 完整推送命令（复制粘贴即可）

```bash
cd /hy-tmp/clustering
git push -u origin main
```

**重要**：密码使用 Personal Access Token，不是 GitHub 密码！

---

## ❓ 常见问题

### Q: 忘记保存 Token 怎么办？
**A**: 重新生成一个新的 token：https://github.com/settings/tokens

### Q: 推送失败，显示 "Authentication failed"
**A**: 
1. 确认使用的是 token，不是密码
2. 确认 token 有 `repo` 权限
3. 确认 token 没有过期

### Q: 如何查看我的 tokens？
**A**: 访问：https://github.com/settings/tokens

### Q: 推送后可以删除 token 吗？
**A**: 不建议。如果删除，下次推送需要重新创建。建议设置较短的有效期（如 90 天）。

---

## 🎯 推送成功后

你的仓库将在这里可见：
**https://github.com/GaiJingze/Neuroscience-baseline**

包含：
- ✅ 51 个文件
- ✅ 11,490+ 行代码
- ✅ 完整的 clustering/hashing pipeline
- ✅ FlyHash baseline（已测试）
- ✅ 完整文档

建议后续操作：
1. 在 GitHub 上添加 Repository Description
2. 添加 Topics 标签：`spiking-neural-networks`, `clustering`, `pytorch`, `neuroscience`
3. 更新 README（如果需要）
4. 邀请协作者（如果有）

---

**准备好推送了吗？** 运行：

```bash
cd /hy-tmp/clustering
git push -u origin main
```
