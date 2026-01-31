# 将结果保存到 Hugging Face Hub

**⚠️ 关键：** 作业环境是临时的。作业完成时，所有结果都会丢失，除非持久化到 Hub 或外部存储。

## 为什么需要持久化

在 Hugging Face Jobs 上运行时：
- 环境是临时的
- 作业完成时所有文件被删除
- 没有本地磁盘持久化
- 作业结束后无法访问结果

**没有持久化，所有工作将永久丢失。**

## 持久化选项

### 选项 1：推送到 Hugging Face Hub（推荐）

**对于模型：**
```python
from transformers import AutoModel
model.push_to_hub("username/model-name", token=os.environ.get("HF_TOKEN"))
```

**对于数据集：**
```python
from datasets import Dataset
dataset.push_to_hub("username/dataset-name", token=os.environ.get("HF_TOKEN"))
```

**对于文件/制品：**
```python
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.upload_file(
    path_or_fileobj="results.json",
    path_in_repo="results.json",
    repo_id="username/results",
    repo_type="dataset"
)
```

### 选项 2：外部存储

**S3：**
```python
import boto3
s3 = boto3.client('s3')
s3.upload_file('results.json', 'my-bucket', 'results.json')
```

**Google Cloud Storage：**
```python
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('my-bucket')
blob = bucket.blob('results.json')
blob.upload_from_filename('results.json')
```

### 选项 3：API 端点

```python
import requests
requests.post("https://your-api.com/results", json=results)
```

## Hub 推送的必要配置

### 作业配置

**始终包含 HF_TOKEN：**
```python
hf_jobs("uv", {
    "script": "your_script.py",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # ✅ Hub 操作所需
})
```

### 脚本配置

**验证令牌存在：**
```python
import os
assert "HF_TOKEN" in os.environ, "Hub 操作需要 HF_TOKEN！"
```

**使用令牌进行 Hub 操作：**
```python
from huggingface_hub import HfApi

# 自动从环境中检测 HF_TOKEN
api = HfApi()

# 或显式传递令牌
api = HfApi(token=os.environ.get("HF_TOKEN"))
```

## 完整示例

### 示例 1：推送数据集

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

# 处理数据
data = {"text": ["Sample 1", "Sample 2"]}
dataset = Dataset.from_dict(data)

# 推送到 Hub
dataset.push_to_hub("username/my-dataset")
print("✅ 数据集已推送！")
""",
    "flavor": "cpu-basic",
    "timeout": "30m",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

### 示例 2：推送模型

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
tokenizer = AutoTokenizer.from_pretrained("base-model")
# ... 处理模型 ...

# 推送到 Hub
model.push_to_hub("username/my-model")
tokenizer.push_to_hub("username/my-model")
print("✅ 模型已推送！")
""",
    "flavor": "a10g-large",
    "timeout": "2h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

### 示例 3：推送制品

```python
hf_jobs("uv", {
    "script": """
# /// script
# dependencies = ["huggingface-hub", "pandas"]
# ///

import os
import json
import pandas as pd
from huggingface_hub import HfApi

# 验证令牌
assert "HF_TOKEN" in os.environ, "需要 HF_TOKEN！"

# 生成结果
results = {"accuracy": 0.95, "loss": 0.05}
df = pd.DataFrame([results])

# 保存文件
with open("results.json", "w") as f:
    json.dump(results, f)
df.to_csv("results.csv", index=False)

# 推送到 Hub
api = HfApi()
api.upload_file("results.json", "results.json", "username/results", repo_type="dataset")
api.upload_file("results.csv", "results.csv", "username/results", repo_type="dataset")
print("✅ 结果已推送！")
""",
    "flavor": "cpu-basic",
    "timeout": "30m",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

## 认证方法

### 方法 1：自动令牌（推荐）

```python
"secrets": {"HF_TOKEN": "$HF_TOKEN"}
```

自动使用你登录的 Hugging Face 令牌。

### 方法 2：显式令牌

```python
"secrets": {"HF_TOKEN": "hf_abc123..."}
```

显式提供令牌（出于安全考虑不推荐）。

### 方法 3：环境变量

```python
"env": {"HF_TOKEN": "hf_abc123..."}
```

作为常规环境变量传递（安全性低于 secrets）。

**始终优先使用方法 1** 以获得安全性和便利性。

## 验证清单

在提交任何保存到 Hub 的作业之前，验证：

- [ ] 作业配置中包含 `secrets={"HF_TOKEN": "$HF_TOKEN"}`
- [ ] 脚本检查令牌：`assert "HF_TOKEN" in os.environ`
- [ ] 脚本中包含 Hub 推送代码
- [ ] 仓库名称不与现有仓库冲突
- [ ] 你对目标命名空间有写入权限

## 仓库设置

### 自动创建

如果仓库不存在，首次推送时会自动创建（如果令牌具有写入权限）。

### 手动创建

在推送前创建仓库：

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(
    repo_id="username/repo-name",
    repo_type="model",  # 或 "dataset"
    private=False,  # 或 True 用于私有仓库
)
```

### 仓库命名

**有效名称：**
- `username/my-model`
- `username/model-name`
- `organization/model-name`

**无效名称：**
- `model-name`（缺少用户名）
- `username/model name`（不允许空格）
- `username/MODEL`（不鼓励大写）

## 故障排除

### 错误：401 Unauthorized

**原因：** 未提供 HF_TOKEN 或令牌无效

**解决方案：**
1. 验证作业配置中包含 `secrets={"HF_TOKEN": "$HF_TOKEN"}`
2. 检查你是否已登录：`hf_whoami()`
3. 重新登录：`hf auth login`

### 错误：403 Forbidden

**原因：** 对仓库没有写入权限

**解决方案：**
1. 检查仓库命名空间是否与你的用户名匹配
2. 验证你是组织的成员（如果使用组织命名空间）
3. 检查令牌是否具有写入权限

### 错误：Repository not found

**原因：** 仓库不存在且自动创建失败

**解决方案：**
1. 先手动创建仓库
2. 检查仓库名称格式
3. 验证命名空间存在

### 错误：Push failed

**原因：** 网络问题或 Hub 不可用

**解决方案：**
1. 检查日志中的具体错误
2. 验证令牌有效
3. 重试推送操作

## 最佳实践

1. **在 Hub 操作前始终验证令牌存在**
2. **使用描述性仓库名称**（例如，`my-experiment-results` 而不是 `results`）
3. **对于大型结果增量推送**（使用检查点）
4. **在作业完成前在日志中验证推送成功**
5. **使用适当的仓库类型**（model vs dataset）
6. **添加 README** 描述结果
7. **使用相关标签标记仓库**

## 监控推送进度

检查日志中的推送进度：

**MCP 工具：**
```python
hf_jobs("logs", {"job_id": "your-job-id"})
```

**CLI：**
```bash
hf jobs logs <job-id>
```

**Python API：**
```python
from huggingface_hub import fetch_job_logs
for log in fetch_job_logs(job_id="your-job-id"):
    print(log)
```

**查找：**
```
正在推送到 username/repo-name...
上传文件 results.json: 100%
✅ 推送成功
```

## 关键要点

**没有 `secrets={"HF_TOKEN": "$HF_TOKEN"}` 和持久化代码，所有结果将永久丢失。**

在提交任何产生结果的作业之前，始终验证两者都已配置。
