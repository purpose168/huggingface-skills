# Hugging Face Jobs 令牌使用指南

**⚠️ 关键：** 正确使用令牌对于任何与 Hugging Face Hub 交互的作业至关重要。

## 概述

Hugging Face 令牌是允许你的作业与 Hub 交互的身份验证凭据。它们对于以下操作是必需的：
- 将模型/数据集推送到 Hub
- 访问私有仓库
- 创建新仓库
- 以编程方式使用 Hub API
- 任何需要身份验证的 Hub 操作

## 令牌类型

### 读取令牌
- **权限：** 下载模型/数据集，读取私有仓库
- **使用场景：** 仅需要下载/读取内容的作业
- **创建：** https://huggingface.co/settings/tokens

### 写入令牌
- **权限：** 推送模型/数据集，创建仓库，修改内容
- **使用场景：** 需要上传结果的作业（最常见）
- **创建：** https://huggingface.co/settings/tokens
- **⚠️ 必需用于：** 推送模型、数据集或任何上传操作

### 组织令牌
- **权限：** 代表组织行事
- **使用场景：** 在组织命名空间下运行的作业
- **创建：** 组织设置 → 令牌

## 向作业提供令牌

### 方法 1：自动令牌（推荐）⭐

```python
hf_jobs("uv", {
    "script": "your_script.py",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # ✅ 自动替换
})
```

**工作原理：**
1. `$HF_TOKEN` 是一个占位符，会被替换为你的实际令牌
2. 使用你登录会话中的令牌 (`hf auth login`)
3. 令牌作为密钥传递时在服务器端加密
4. 最安全和便捷的方法

**优点：**
- ✅ 代码中无令牌暴露
- ✅ 使用你当前的登录会话
- ✅ 重新登录时自动更新
- ✅ 与 MCP 工具无缝配合
- ✅ 令牌在服务器端加密

**要求：**
- 必须已登录：`hf auth login` 或 `hf_whoami()` 可用
- 令牌必须具有所需的权限

### 方法 2：显式令牌（不推荐）

```python
hf_jobs("uv", {
    "script": "your_script.py",
    "secrets": {"HF_TOKEN": "hf_abc123..."}  # ⚠️ 硬编码令牌
})
```

**何时使用：**
- 仅当自动令牌不起作用时
- 使用特定令牌进行测试
- 组织令牌（谨慎使用）

**安全隐患：**
- ❌ 令牌在代码中暴露
- ❌ 令牌轮换时必须手动更新
- ❌ 令牌暴露风险
- ❌ 不推荐用于生产环境

### 方法 3：环境变量（安全性较低）

```python
hf_jobs("uv", {
    "script": "your_script.py",
    "env": {"HF_TOKEN": "hf_abc123..."}  # ⚠️ 比 secrets 安全性低
})
```

**与 secrets 的区别：**
- `env` 变量在作业日志中可见
- `secrets` 在服务器端加密
- 对于令牌，始终优先使用 `secrets`

**何时使用：**
- 仅用于非敏感配置
- 永远不要用于令牌（请使用 `secrets` 替代）

## 在脚本中使用令牌

### 访问令牌

通过 `secrets` 传递的令牌在脚本中作为环境变量可用：

```python
import os

# 从环境中获取令牌
token = os.environ.get("HF_TOKEN")

# 验证令牌存在
if not token:
    raise ValueError("环境中未找到 HF_TOKEN！")
```

### 与 Hugging Face Hub 一起使用

**选项 1：显式令牌参数**
```python
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN"))
api.upload_file(...)
```

**选项 2：自动检测（推荐）**
```python
from huggingface_hub import HfApi

# 自动使用 HF_TOKEN 环境变量
api = HfApi()  # ✅ 更简单，使用环境中的令牌
api.upload_file(...)
```

**选项 3：与 transformers/datasets 一起使用**
```python
from transformers import AutoModel
from datasets import load_dataset

# 从环境中自动检测 HF_TOKEN
model = AutoModel.from_pretrained("username/model")
dataset = load_dataset("username/dataset")

# 对于推送操作，令牌会被自动检测
model.push_to_hub("username/new-model")
dataset.push_to_hub("username/new-dataset")
```

### 完整示例

```python
# /// script
# dependencies = ["huggingface-hub", "datasets"]
# ///

import os
from huggingface_hub import HfApi
from datasets import Dataset

# 验证令牌可用
assert "HF_TOKEN" in os.environ, "Hub 操作需要 HF_TOKEN！"

# 使用令牌进行 Hub 操作
api = HfApi()  # 自动检测 HF_TOKEN

# 创建并推送数据集
data = {"text": ["Hello", "World"]}
dataset = Dataset.from_dict(data)

# 推送到 Hub（令牌自动检测）
dataset.push_to_hub("username/my-dataset")

print("✅ 数据集推送成功！")
```

## 令牌验证

### 本地检查身份验证

```python
from huggingface_hub import whoami

try:
    user_info = whoami()
    print(f"✅ 已登录为：{user_info['name']}")
except Exception as e:
    print(f"❌ 未身份验证：{e}")
```

### 在作业中验证令牌

```python
import os

# 检查令牌存在
if "HF_TOKEN" not in os.environ:
    raise ValueError("环境中未找到 HF_TOKEN！")

token = os.environ["HF_TOKEN"]

# 验证令牌格式（应以 "hf_" 开头）
if not token.startswith("hf_"):
    raise ValueError(f"无效的令牌格式：{token[:10]}...")

# 测试令牌是否有效
from huggingface_hub import whoami
try:
    user_info = whoami(token=token)
    print(f"✅ 令牌对用户有效：{user_info['name']}")
except Exception as e:
    raise ValueError(f"令牌验证失败：{e}")
```

## 常见令牌问题

### 错误：401 Unauthorized

**症状：**
```
401 Client Error: Unauthorized for url: https://huggingface.co/api/...
```

**原因：**
1. 作业中缺少令牌
2. 令牌无效或已过期
3. 令牌传递不正确

**解决方案：**
1. 在作业配置中添加 `secrets={"HF_TOKEN": "$HF_TOKEN"}`
2. 验证本地 `hf_whoami()` 可用
3. 重新登录：`hf auth login`
4. 检查令牌是否已过期

**验证：**
```python
# 在你的脚本中
import os
assert "HF_TOKEN" in os.environ, "HF_TOKEN 缺失！"
```

### 错误：403 Forbidden

**症状：**
```
403 Client Error: Forbidden for url: https://huggingface.co/api/...
```

**原因：**
1. 令牌缺少所需权限（只读令牌用于写入操作）
2. 无访问私有仓库的权限
3. 组织权限不足

**解决方案：**
1. 确保令牌具有写入权限
2. 在 https://huggingface.co/settings/tokens 检查令牌类型
3. 验证对目标仓库的访问权限
4. 如需要，使用组织令牌

**检查令牌权限：**
```python
from huggingface_hub import whoami

user_info = whoami()
print(f"用户：{user_info['name']}")
print(f"类型：{user_info.get('type', 'user')}")
```

### 错误：环境中未找到令牌

**症状：**
```
KeyError: 'HF_TOKEN'
ValueError: HF_TOKEN not found
```

**原因：**
1. 作业配置中未传递 `secrets`
2. 键名错误（应为 `HF_TOKEN`）
3. 使用 `env` 而不是 `secrets`

**解决方案：**
1. 使用 `secrets={"HF_TOKEN": "$HF_TOKEN"}`（而非 `env`）
2. 验证键名确实是 `HF_TOKEN`
3. 检查作业配置语法

**正确配置：**
```python
# ✅ 正确
hf_jobs("uv", {
    "script": "...",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})

# ❌ 错误 - 使用 env 而非 secrets
hf_jobs("uv", {
    "script": "...",
    "env": {"HF_TOKEN": "$HF_TOKEN"}  # 安全性较低
})

# ❌ 错误 - 键名错误
hf_jobs("uv", {
    "script": "...",
    "secrets": {"TOKEN": "$HF_TOKEN"}  # 错误的键
})
```

### 错误：仓库访问被拒绝

**症状：**
```
403 Client Error: Forbidden
Repository not found or access denied
```

**原因：**
1. 令牌无访问私有仓库的权限
2. 仓库不存在且无法创建
3. 命名空间错误

**解决方案：**
1. 使用具有访问权限的账户的令牌
2. 验证仓库可见性（公开 vs 私有）
3. 检查命名空间是否与令牌所有者匹配
4. 如需，先创建仓库

**检查仓库访问：**
```python
from huggingface_hub import HfApi

api = HfApi()
try:
    repo_info = api.repo_info("username/repo-name")
    print(f"✅ 访问已授予：{repo_info.id}")
except Exception as e:
    print(f"❌ 访问被拒绝：{e}")
```

## 令牌安全最佳实践

### 1. 永远不要提交令牌

**❌ 错误：**
```python
# 永远不要这样做！
token = "hf_abc123xyz..."
api = HfApi(token=token)
```

**✅ 正确：**
```python
# 使用环境变量
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
```

### 2. 使用 Secrets，而非环境变量

**❌ 错误：**
```python
hf_jobs("uv", {
    "script": "...",
    "env": {"HF_TOKEN": "$HF_TOKEN"}  # 在日志中可见
})
```

**✅ 正确：**
```python
hf_jobs("uv", {
    "script": "...",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # 在服务器端加密
})
```

### 3. 使用自动令牌替换

**❌ 错误：**
```python
hf_jobs("uv", {
    "script": "...",
    "secrets": {"HF_TOKEN": "hf_abc123..."}  # 硬编码
})
```

**✅ 正确：**
```python
hf_jobs("uv", {
    "script": "...",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # 自动
})
```

### 4. 定期轮换令牌

- 定期生成新令牌
- 撤销旧令牌
- 更新作业配置
- 监控令牌使用情况

### 5. 使用最小权限

- 创建仅具有所需权限的令牌
- 不需要写入时使用只读令牌
- 不要为常规作业使用管理员令牌

### 6. 不要共享令牌

- 每个用户应使用自己的令牌
- 不要将令牌提交到仓库
- 不要在日志或消息中共享令牌

### 7. 监控令牌使用情况

- 在 Hub 设置中检查令牌活动
- 查看作业日志中的令牌问题
- 设置未授权访问的警报

## 令牌工作流示例

### 示例 1：将模型推送到 Hub

```python
hf_jobs("uv", {
    "script": """
# /// script
# dependencies = ["transformers"]
# ///

import os
from transformers import AutoModel, AutoTokenizer

# 验证令牌
assert "HF_TOKEN" in os.environ, "需要 HF_TOKEN！"

# 加载并处理模型
model = AutoModel.from_pretrained("base-model")
# ... 处理模型 ...

# 推送到 Hub（令牌自动检测）
model.push_to_hub("username/my-model")
print("✅ 模型已推送！")
""",
    "flavor": "a10g-large",
    "timeout": "2h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # ✅ 提供令牌
})
```

### 示例 2：访问私有数据集

```python
hf_jobs("uv", {
    "script": """
# /// script
# dependencies = ["datasets"]
# ///

import os
from datasets import load_dataset

# 验证令牌
assert "HF_TOKEN" in os.environ, "需要 HF_TOKEN！"

# 加载私有数据集（令牌自动检测）
dataset = load_dataset("private-org/private-dataset")
print(f"✅ 加载了 {len(dataset)} 个示例")
""",
    "flavor": "cpu-basic",
    "timeout": "30m",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # ✅ 提供令牌
})
```

### 示例 3：创建并推送数据集

```python
hf_jobs("uv", {
    "script": """
# /// script
# dependencies = ["datasets", "huggingface-hub"]
# ///

import os
from datasets import Dataset
from huggingface_hub import HfApi

# 验证令牌
assert "HF_TOKEN" in os.environ, "需要 HF_TOKEN！"

# 创建数据集
data = {"text": ["Sample 1", "Sample 2"]}
dataset = Dataset.from_dict(data)

# 推送到 Hub
api = HfApi()  # 自动检测 HF_TOKEN
dataset.push_to_hub("username/my-dataset")
print("✅ 数据集已推送！")
""",
    "flavor": "cpu-basic",
    "timeout": "30m",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # ✅ 提供令牌
})
```

## 快速参考

### 令牌检查清单

在提交使用 Hub 的作业之前：

- [ ] 作业包含 `secrets={"HF_TOKEN": "$HF_TOKEN"}`
- [ ] 脚本检查令牌：`assert "HF_TOKEN" in os.environ`
- [ ] 令牌具有所需权限（读/写）
- [ ] 用户已登录：`hf_whoami()` 可用
- [ ] 令牌未在脚本中硬编码
- [ ] 使用 `secrets` 而非 `env` 存储令牌

### 常见模式

**模式 1：自动检测令牌**
```python
from huggingface_hub import HfApi
api = HfApi()  # 使用环境中的 HF_TOKEN
```

**模式 2：显式令牌**
```python
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
```

**模式 3：验证令牌**
```python
import os
assert "HF_TOKEN" in os.environ, "需要 HF_TOKEN！"
```

## 关键要点

1. **始终使用 `secrets={"HF_TOKEN": "$HF_TOKEN"}`** 进行 Hub 操作
2. **永远不要硬编码令牌** 在脚本或作业配置中
3. **在 Hub 操作前验证令牌存在** 在脚本中
4. **尽可能使用自动检测**（不带令牌参数的 `HfApi()`）
5. **检查权限** - 确保令牌具有所需的访问权限
6. **监控令牌使用** - 定期查看活动
7. **轮换令牌** - 定期生成新令牌