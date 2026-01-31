# 将训练结果保存到 Hugging Face Hub

**⚠️ 重要提示:** 训练环境是临时的。除非推送到 Hub,否则作业完成时所有结果都会丢失。

## 为什么需要推送到 Hub

在 Hugging Face Jobs 上运行时:
- 环境是临时的
- 作业完成时所有文件都会被删除
- 没有本地磁盘持久化
- 作业结束后无法访问结果

**如果不推送到 Hub,训练将完全白费。**

## 必需配置

### 1. 训练配置

在您的 SFTConfig 或训练器配置中:

```python
SFTConfig(
    push_to_hub=True,                    # 启用 Hub 推送
    hub_model_id="username/model-name",   # 目标仓库
)
```

### 2. 作业配置

提交作业时:

```python
hf_jobs("uv", {
    "script": "train.py",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # 提供身份验证
})
```

**`$HF_TOKEN` 占位符会自动替换为您的 Hugging Face 令牌。**

## 完整示例

```python
# train.py
# /// script
# dependencies = ["trl"]
# ///

from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

dataset = load_dataset("trl-lib/Capybara", split="train")

# 配置 Hub 推送
config = SFTConfig(
    output_dir="my-model",
    num_train_epochs=3,

    # ✅ 关键:Hub 推送配置
    push_to_hub=True,
    hub_model_id="myusername/my-trained-model",

    # 可选:推送策略
    push_to_hub_model_id="myusername/my-trained-model",
    push_to_hub_organization=None,
    push_to_hub_token=None,  # 使用环境令牌
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    args=config,
)

trainer.train()

# ✅ 推送最终模型
trainer.push_to_hub()

print("✅ 模型已保存到: https://huggingface.co/myusername/my-trained-model")
```

**使用身份验证提交:**

```python
hf_jobs("uv", {
    "script": "train.py",
    "flavor": "a10g-large",
    "timeout": "2h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # ✅ 必需!
})
```

## 保存的内容

当 `push_to_hub=True` 时:

1. **模型权重** - 最终训练的参数
2. **分词器** - 关联的分词器
3. **配置** - 模型配置 (config.json)
4. **训练参数** - 使用的超参数
5. **模型卡片** - 自动生成的文档
6. **检查点** - 如果启用了 `save_strategy="steps"`

## 检查点保存

在训练期间保存中间检查点:

```python
SFTConfig(
    output_dir="my-model",
    push_to_hub=True,
    hub_model_id="username/my-model",

    # 检查点配置
    save_strategy="steps",
    save_steps=100,              # 每 100 步保存一次
    save_total_limit=3,          # 仅保留最后 3 个检查点
)
```

**优势:**
- 如果作业失败,可以恢复训练
- 比较检查点性能
- 使用中间模型

**检查点推送到:** `username/my-model` (同一仓库)

## 身份验证方法

### 方法 1: 自动令牌 (推荐)

```python
"secrets": {"HF_TOKEN": "$HF_TOKEN"}
```

自动使用您登录的 Hugging Face 令牌。

### 方法 2: 显式令牌

```python
"secrets": {"HF_TOKEN": "hf_abc123..."}
```

显式提供令牌 (出于安全考虑不推荐)。

### 方法 3: 环境变量

```python
"env": {"HF_TOKEN": "hf_abc123..."}
```

作为常规环境变量传递 (不如 secrets 安全)。

**出于安全和便利考虑,始终优先使用方法 1。**

## 验证清单

提交任何训练作业之前,请验证:

- [ ] 训练配置中有 `push_to_hub=True`
- [ ] 指定了 `hub_model_id` (格式: `username/model-name`)
- [ ] 作业配置中有 `secrets={"HF_TOKEN": "$HF_TOKEN"}`
- [ ] 仓库名称不与现有仓库冲突
- [ ] 您对目标命名空间有写入权限

## 仓库设置

### 自动创建

如果仓库不存在,首次推送时会自动创建。

### 手动创建

在训练前创建仓库:

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(
    repo_id="username/model-name",
    repo_type="model",
    private=False,  # 或 True 表示私有仓库
)
```

### 仓库命名

**有效名称:**
- `username/my-model`
- `username/model-name`
- `organization/model-name`

**无效名称:**
- `model-name` (缺少用户名)
- `username/model name` (不允许空格)
- `username/MODEL` (不鼓励使用大写)

## 故障排除

### 错误: 401 未授权

**原因:** 未提供 HF_TOKEN 或令牌无效

**解决方案:**
1. 验证作业配置中有 `secrets={"HF_TOKEN": "$HF_TOKEN"}`
2. 检查您是否已登录: `huggingface-cli whoami`
3. 重新登录: `huggingface-cli login`

### 错误: 403 禁止访问

**原因:** 没有仓库的写入权限

**解决方案:**
1. 检查仓库命名空间是否与您的用户名匹配
2. 验证您是组织成员 (如果使用组织命名空间)
3. 检查仓库是否为私有 (如果访问组织仓库)

### 错误: 仓库未找到

**原因:** 仓库不存在且自动创建失败

**解决方案:**
1. 先手动创建仓库
2. 检查仓库名称格式
3. 验证命名空间存在

### 错误: 训练期间推送失败

**原因:** 网络问题或 Hub 不可用

**解决方案:**
1. 训练继续但最终推送失败
2. 检查点可能已保存
3. 作业完成后手动重新运行推送

### 问题: 模型已保存但不可见

**可能原因:**
1. 仓库是私有的 - 检查 https://huggingface.co/username
2. 命名空间错误 - 验证 `hub_model_id` 与登录信息匹配
3. 推送仍在进行中 - 等待几分钟

## 训练后手动推送

如果训练完成但推送失败,请手动推送:

```python
from transformers import AutoModel, AutoTokenizer

# 从本地检查点加载
model = AutoModel.from_pretrained("./output_dir")
tokenizer = AutoTokenizer.from_pretrained("./output_dir")

# 推送到 Hub
model.push_to_hub("username/model-name", token="hf_abc123...")
tokenizer.push_to_hub("username/model-name", token="hf_abc123...")
```

**注意:** 仅在作业未完成 (文件仍存在) 时才可能。

## 最佳实践

1. **始终启用 `push_to_hub=True`**
2. **使用检查点保存** 进行长时间训练
3. **在作业完成前验证 Hub 推送** 在日志中
4. **设置适当的 `save_total_limit`** 以避免过多的检查点
5. **使用描述性的仓库名称** (例如, `qwen-capybara-sft` 而不是 `model1`)
6. **添加模型卡片** 包含训练详细信息
7. **标记模型** 使用相关标签 (例如, `text-generation`, `fine-tuned`)

## 监控推送进度

检查日志以获取推送进度:

```python
hf_jobs("logs", {"job_id": "your-job-id"})
```

**查找:**
```
Pushing model to username/model-name...
Upload file pytorch_model.bin: 100%
✅ Model pushed successfully
```

## 示例: 完整生产环境设置

```python
# production_train.py
# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.7.0"]
# ///

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import os

# 验证令牌可用
assert "HF_TOKEN" in os.environ, "HF_TOKEN not found in environment!"

# 加载数据集
dataset = load_dataset("trl-lib/Capybara", split="train")
print(f"✅ Dataset loaded: {len(dataset)} examples")

# 配置全面的 Hub 设置
config = SFTConfig(
    output_dir="qwen-capybara-sft",

    # Hub 配置
    push_to_hub=True,
    hub_model_id="myusername/qwen-capybara-sft",
    hub_strategy="checkpoint",  # 推送检查点

    # 检查点配置
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,

    # 训练设置
    num_train_epochs=3,
    per_device_train_batch_size=4,

    # 日志记录
    logging_steps=10,
    logging_first_step=True,
)

# 使用 LoRA 训练
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    args=config,
    peft_config=LoraConfig(r=16, lora_alpha=32),
)

print("🚀 Starting training...")
trainer.train()

print("💾 Pushing final model to Hub...")
trainer.push_to_hub()

print("✅ Training complete!")
print(f"Model available at: https://huggingface.co/myusername/qwen-capybara-sft")
```

**提交:**

```python
hf_jobs("uv", {
    "script": "production_train.py",
    "flavor": "a10g-large",
    "timeout": "6h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

## 关键要点

**如果没有 `push_to_hub=True` 和 `secrets={"HF_TOKEN": "$HF_TOKEN"}`,所有训练结果将永久丢失。**

提交任何训练作业之前,请始终验证两者都已配置。
