---
name: hugging-face-model-trainer
description: 当用户想要在Hugging Face Jobs基础设施上使用TRL（Transformer Reinforcement Learning）训练或微调语言模型时，应使用此技能。涵盖SFT、DPO、GRPO和奖励建模训练方法，以及用于本地部署的GGUF转换。包括关于TRL Jobs包、带有PEP 723格式的UV脚本、数据集准备和验证、硬件选择、成本估算、Trackio监控、Hub身份验证和模型持久化的指导。当涉及云GPU训练、GGUF转换或用户提及在Hugging Face Jobs上训练而无需本地GPU设置时，应调用此技能。
license: Complete terms in LICENSE.txt
---

# 在Hugging Face Jobs上进行TRL训练

## 概述

在完全托管的Hugging Face基础设施上使用TRL（Transformer Reinforcement Learning）训练语言模型。无需本地GPU设置——模型在云GPU上训练，结果自动保存到Hugging Face Hub。

**TRL提供多种训练方法：**
- **SFT**（监督微调）- 标准指令调优
- **DPO**（直接偏好优化）- 基于偏好数据的对齐
- **GRPO**（组相对策略优化）- 在线RL训练
- **奖励建模** - 为RLHF训练奖励模型

**有关TRL方法的详细文档：**
```python
hf_doc_search("your query", product="trl")
hf_doc_fetch("https://huggingface.co/docs/trl/sft_trainer")  # SFT
hf_doc_fetch("https://huggingface.co/docs/trl/dpo_trainer")  # DPO
# 等等
```

**另见：** `references/training_methods.md`获取方法概述和选择指南

## 何时使用此技能

当用户想要：
- 在云GPU上微调语言模型，无需本地基础设施
- 使用TRL方法训练（SFT、DPO、GRPO等）
- 在Hugging Face Jobs基础设施上运行训练作业
- 将训练好的模型转换为GGUF用于本地部署（Ollama、LM Studio、llama.cpp）
- 确保训练好的模型永久保存到Hub
- 使用具有优化默认值的现代工作流

## 关键指令

在协助训练作业时：

1. **始终使用`hf_jobs()` MCP工具** - 使用`hf_jobs("uv", {...})`提交作业，而不是bash `trl-jobs`命令。`script`参数直接接受Python代码。除非用户明确要求，否则不要保存到本地文件。将脚本内容作为字符串传递给`hf_jobs()`。如果用户要求"训练模型"、"微调"或类似请求，您必须创建训练脚本并立即使用`hf_jobs()`提交作业。

2. **始终包含Trackio** - 每个训练脚本应包含Trackio用于实时监控。使用`scripts/`中的示例脚本作为模板。

3. **提交后提供作业详情** - 提交后，提供作业ID、监控URL、估计时间，并注明用户可以稍后请求状态检查。

4. **使用示例脚本作为模板** - 参考`scripts/train_sft_example.py`、`scripts/train_dpo_example.py`等作为起点。

## 本地脚本依赖项

要在本地运行脚本（如`estimate_cost.py`），安装依赖项：
```bash
pip install -r requirements.txt
```

## 前提条件清单

在开始任何训练作业之前，验证：

### ✅ **账户和身份验证**
- 具有[Pro](https://hf.co/pro)、[Team](https://hf.co/enterprise)或[Enterprise](https://hf.co/enterprise)计划的Hugging Face账户（Jobs需要付费计划）
- 已认证登录：使用`hf_whoami()`检查
- **用于Hub推送的HF_TOKEN** ⚠️ 关键 - 训练环境是临时的，必须推送到Hub，否则所有训练结果都会丢失
- 令牌必须具有写入权限
- **必须在作业配置中传递`secrets={"HF_TOKEN": "$HF_TOKEN"}`**以使令牌可用（`$HF_TOKEN`语法引用您的实际令牌值）

### ✅ **数据集要求**
- 数据集必须存在于Hub上或可通过`datasets.load_dataset()`加载
- 格式必须匹配训练方法（SFT："messages"/文本/提示-完成；DPO：选择/拒绝；GRPO：仅提示）
- **在GPU训练前始终验证未知数据集**，以防止格式失败（见下文数据集验证部分）
- 大小适合硬件（演示：t4-small上50-100个示例；生产：a10g-large/a100-large上1K-10K+）

### ⚠️ **关键设置**
- **超时必须超过预期训练时间** - 默认30分钟对于大多数训练来说太短。最低推荐：1-2小时。如果超时，作业失败并丢失所有进度。
- **必须启用Hub推送** - 配置：`push_to_hub=True`，`hub_model_id="username/model-name"`；作业：`secrets={"HF_TOKEN": "$HF_TOKEN"}`

## 异步作业指南

**⚠️ 重要：训练作业异步运行，可能需要数小时**

### 必要操作

**当用户请求训练时：**
1. **创建训练脚本**，包含Trackio（使用`scripts/train_sft_example.py`作为模板）
2. **立即提交**使用`hf_jobs()` MCP工具，脚本内容内联 - 除非用户请求，否则不要保存到文件
3. **报告提交**，提供作业ID、监控URL和估计时间
4. **等待用户**请求状态检查 - 不要自动轮询

### 基本规则
- **作业在后台运行** - 提交立即返回；训练独立继续
- **初始日志延迟** - 日志可能需要30-60秒才会出现
- **用户检查状态** - 等待用户请求状态更新
- **避免轮询** - 仅在用户请求时检查日志；提供监控链接

### 提交后

**向用户提供：**
- ✅ 作业ID和监控URL
- ✅ 预计完成时间
- ✅ Trackio仪表板URL
- ✅ 注意用户可以稍后请求状态检查

**示例响应：**
```
✅ 作业提交成功！

作业ID：abc123xyz
监控：https://huggingface.co/jobs/username/abc123xyz

预计时间：~2小时
预计成本：~$10

作业在后台运行。准备好时请告诉我检查状态！
```

## 快速开始：三种方法

**💡 演示提示：** 对于在较小GPU（t4-small）上的快速演示，省略`eval_dataset`和`eval_strategy`以节省~40%内存。您仍然会看到训练损失和学习进度。

### 序列长度配置

**TRL配置类使用`max_length`（而非`max_seq_length`）控制标记化序列长度：**

```python
# ✅ 正确 - 如果需要设置序列长度
SFTConfig(max_length=512)   # 将序列截断为512个标记
DPOConfig(max_length=2048)  # 更长上下文（2048个标记）

# ❌ 错误 - 此参数不存在
SFTConfig(max_seq_length=512)  # TypeError!
```

**默认行为：** `max_length=1024`（从右侧截断）。这对大多数训练效果很好。

**何时覆盖：**
- **更长上下文**：设置更高（例如，`max_length=2048`）
- **内存限制**：设置更低（例如，`max_length=512`）
- **视觉模型**：设置`max_length=None`（防止剪切图像标记）

**通常您根本不需要设置此参数** - 下面的示例使用合理的默认值。

### 方法1：UV脚本（推荐—默认选择）

UV脚本使用PEP 723内联依赖项进行干净、自包含的训练。**这是Claude Code的主要方法。**

```python
hf_jobs("uv", {
    "script": """
# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.7.0", "trackio"]
# ///

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import trackio

dataset = load_dataset("trl-lib/Capybara", split="train")

# 创建训练/评估分割用于监控
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
    peft_config=LoraConfig(r=16, lora_alpha=32),
    args=SFTConfig(
        output_dir="my-model",
        push_to_hub=True,
        hub_model_id="username/my-model",
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=50,
        report_to="trackio",
        project="meaningful_prject_name", # 训练名称的项目名称（trackio）
        run_name="meaningful_run_name",   # 特定训练运行的描述性名称（trackio）
    )
)

trainer.train()
trainer.push_to_hub()
""",
    "flavor": "a10g-large",
    "timeout": "2h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**优势：** 直接MCP工具使用，代码干净，依赖项内联声明（PEP 723），无需保存文件，完全控制
**何时使用：** Claude Code中所有训练任务的默认选择，自定义训练逻辑，任何需要`hf_jobs()`的场景

#### 使用脚本

⚠️ **重要：** `script`参数接受内联代码（如上所示）或URL。**本地文件路径不工作。**

**本地路径不工作的原因：**
作业在隔离的Docker容器中运行，无法访问您的本地文件系统。脚本必须是：
- 内联代码（推荐用于自定义训练）
- 可公开访问的URL
- 私有仓库URL（带HF_TOKEN）

**常见错误：**
```python
# ❌ 这些都会失败
hf_jobs("uv", {"script": "train.py"})
hf_jobs("uv", {"script": "./scripts/train.py"})
hf_jobs("uv", {"script": "/path/to/train.py"})
```

**正确方法：**
```python
# ✅ 内联代码（推荐）
hf_jobs("uv", {"script": "# /// script\n# dependencies = [...]\n# ///\n\n<your code>"})

# ✅ 来自Hugging Face Hub
hf_jobs("uv", {"script": "https://huggingface.co/user/repo/resolve/main/train.py"})

# ✅ 来自GitHub
hf_jobs("uv", {"script": "https://raw.githubusercontent.com/user/repo/main/train.py"})

# ✅ 来自Gist
hf_jobs("uv", {"script": "https://gist.githubusercontent.com/user/id/raw/train.py"})
```

**使用本地脚本：** 首先上传到HF Hub：
```bash
huggingface-cli repo create my-training-scripts --type model
huggingface-cli upload my-training-scripts ./train.py train.py
# 使用：https://huggingface.co/USERNAME/my-training-scripts/resolve/main/train.py
```

### 方法2：TRL维护的脚本（官方示例）

TRL提供所有方法的经过实战检验的脚本。可以从URL运行：

```python
hf_jobs("uv", {
    "script": "https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py",
    "script_args": [
        "--model_name_or_path", "Qwen/Qwen2.5-0.5B",
        "--dataset_name", "trl-lib/Capybara",
        "--output_dir", "my-model",
        "--push_to_hub",
        "--hub_model_id", "username/my-model"
    ],
    "flavor": "a10g-large",
    "timeout": "2h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**优势：** 无需编写代码，由TRL团队维护，经过生产测试
**何时使用：** 标准TRL训练，快速实验，不需要自定义代码
**可用：** 脚本可从https://github.com/huggingface/trl/tree/main/examples/scripts获取

### 在Hub上查找更多UV脚本

`uv-scripts`组织提供存储在Hugging Face Hub上作为数据集的即用型UV脚本：

```python
# 发现可用的UV脚本集合
dataset_search({"author": "uv-scripts", "sort": "downloads", "limit": 20})

# 浏览特定集合
hub_repo_details(["uv-scripts/classification"], repo_type="dataset", include_readme=True)
```

**流行集合：** ocr、classification、synthetic-data、vllm、dataset-creation

### 方法3：HF Jobs CLI（直接终端命令）

当`hf_jobs()` MCP工具不可用时，直接使用`hf jobs` CLI。

**⚠️ 关键：CLI语法规则**

```bash
# ✅ 正确语法 - 标志在脚本URL之前
hf jobs uv run --flavor a10g-large --timeout 2h --secrets HF_TOKEN "https://example.com/train.py"

# ❌ 错误 - "run uv"而不是"uv run"
hf jobs run uv "https://example.com/train.py" --flavor a10g-large

# ❌ 错误 - 标志在脚本URL之后（将被忽略！）
hf jobs uv run "https://example.com/train.py" --flavor a10g-large

# ❌ 错误 - "--secret"而不是"--secrets"（复数）
hf jobs uv run --secret HF_TOKEN "https://example.com/train.py"
```

**关键语法规则：**
1. 命令顺序是`hf jobs uv run`（不是`hf jobs run uv`）
2. 所有标志（`--flavor`、`--timeout`、`--secrets`）必须在脚本URL之前
3. 使用`--secrets`（复数），不是`--secret`
4. 脚本URL必须是最后一个位置参数

**完整CLI示例：**
```bash
hf jobs uv run \
  --flavor a10g-large \
  --timeout 2h \
  --secrets HF_TOKEN \
  "https://huggingface.co/user/repo/resolve/main/train.py"
```

**通过CLI检查作业状态：**
```bash
hf jobs ps                        # 列出所有作业
hf jobs logs <job-id>             # 查看日志
hf jobs inspect <job-id>          # 作业详情
hf jobs cancel <job-id>           # 取消作业
```

### 方法4：TRL Jobs包（简化训练）

`trl-jobs`包提供优化的默认值和一行式训练。

```bash
# 安装
pip install trl-jobs

# 使用SFT训练（最简单）
trl-jobs sft \
  --model_name Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/Capybara
```

**优势：** 预配置设置，自动Trackio集成，自动Hub推送，一行命令
**何时使用：** 用户直接在终端工作（非Claude Code上下文），快速本地实验
**仓库：** https://github.com/huggingface/trl-jobs

⚠️ **在Claude Code上下文中，当可用时，优先使用`hf_jobs()` MCP工具（方法1）。**

## 硬件选择

| 模型大小 | 推荐硬件 | 成本（约/小时） | 用例 |
|------------|---------------------|------------------|----------|
| <1B参数 | `t4-small` | ~$0.75 | 演示，快速测试（无评估步骤） |
| 1-3B参数 | `t4-medium`, `l4x1` | ~$1.50-2.50 | 开发 |
| 3-7B参数 | `a10g-small`, `a10g-large` | ~$3.50-5.00 | 生产训练 |
| 7-13B参数 | `a10g-large`, `a100-large` | ~$5-10 | 大型模型（使用LoRA） |
| 13B+参数 | `a100-large`, `a10g-largex2` | ~$10-20 | 非常大（使用LoRA） |

**GPU类型：** cpu-basic/upgrade/performance/xl, t4-small/medium, l4x1/x4, a10g-small/large/largex2/largex4, a100-large, h100/h100x8

**指南：**
- 对于>7B模型，使用**LoRA/PEFT**减少内存
- TRL/Accelerate自动处理多GPU
- 从小型硬件开始测试

**参见：** `references/hardware_guide.md`获取详细规格

## 关键：将结果保存到Hub

**⚠️ 临时环境—必须推送到Hub**

Jobs环境是临时的。作业结束时所有文件都会被删除。如果模型没有推送到Hub，**所有训练都将丢失**。

### 必要配置

**在训练脚本/配置中：**
```python
SFTConfig(
    push_to_hub=True,
    hub_model_id="username/model-name",  # 必须指定
    hub_strategy="every_save",  # 可选：推送检查点
)
```

**在作业提交中：**
```python
{
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # 启用身份验证
}
```

### 验证清单

提交前：
- [ ] 在配置中设置`push_to_hub=True`
- [ ] `hub_model_id`包含用户名/仓库名
- [ ] `secrets`参数包含HF_TOKEN
- [ ] 用户对目标仓库有写入权限

**参见：** `references/hub_saving.md`获取详细故障排除

## 超时管理

**⚠️ 默认：30分钟—对训练来说太短**

### 设置超时

```python
{
    "timeout": "2h"   # 2小时（格式："90m"，"2h"，"1.5h"，或整数秒）
}
```

### 超时指南

| 场景 | 推荐 | 说明 |
|----------|-------------|-------|
| 快速演示（50-100个示例） | 10-30分钟 | 验证设置 |
| 开发训练 | 1-2小时 | 小型数据集 |
| 生产（3-7B模型） | 4-6小时 | 完整数据集 |
| 带LoRA的大型模型 | 3-6小时 | 取决于数据集 |

**始终添加20-30%的缓冲**用于模型/数据集加载、检查点保存、Hub推送操作和网络延迟。

**超时后果：** 作业立即终止，所有未保存的进度丢失，必须从头开始重新启动

## 成本估算

**当使用已知参数规划作业时，提供成本估算。** 使用`scripts/estimate_cost.py`：

```bash
uv run scripts/estimate_cost.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset trl-lib/Capybara \
  --hardware a10g-large \
  --dataset-size 16000 \
  --epochs 3
```

输出包括估计时间、成本、推荐超时（带缓冲）和优化建议。

**何时提供：** 用户计划作业，询问成本/时间，选择硬件，作业将运行>1小时或成本>$5

## 示例训练脚本

**具有所有最佳实践的生产就绪模板：**

正确加载这些脚本：

- **`scripts/train_sft_example.py`** - 完整的SFT训练，包含Trackio、LoRA、检查点
- **`scripts/train_dpo_example.py`** - 用于偏好学习的DPO训练
- **`scripts/train_grpo_example.py`** - 用于在线RL的GRPO训练

这些脚本展示了正确的Hub保存、Trackio集成、检查点管理和优化参数。将其内容内联传递给`hf_jobs()`或用作自定义脚本的模板。

## 监控和跟踪

**Trackio**提供实时指标可视化。参见`references/trackio_guide.md`获取完整设置指南。

**关键点：**
- 将`trackio`添加到依赖项
- 使用`report_to="trackio"`和`run_name="meaningful_name"`配置训练器

### Trackio配置默认值

**除非用户指定，否则使用合理的默认值。** 当生成带有Trackio的训练脚本时：

**默认配置：**
- **空间ID**：`{username}/trackio`（使用"trackio"作为默认空间名称）
- **运行命名**：除非另有指定，否则以用户可识别的方式命名运行（例如，描述任务、模型或目的）
- **配置**：保持最小 - 仅包含超参数和模型/数据集信息
- **项目名称**：使用项目名称将运行与特定项目关联

**用户覆盖：** 如果用户请求特定的trackio配置（自定义空间、运行命名、分组或附加配置），应用他们的偏好而不是默认值。

这对于管理具有相同配置的多个作业或保持训练脚本可移植性很有用。

参见`references/trackio_guide.md`获取完整文档，包括实验的运行分组。

### 检查作业状态

```python
# 列出所有作业
hf_jobs("ps")

# 检查特定作业
hf_jobs("inspect", {"job_id": "your-job-id"})

# 查看日志
hf_jobs("logs", {"job_id": "your-job-id"})
```

**记住：** 等待用户请求状态检查。避免重复轮询。

## 数据集验证

**在启动GPU训练前验证数据集格式，以防止训练失败的主要原因：格式不匹配。**

### 为什么验证

- 50%+的训练失败是由于数据集格式问题
- DPO特别严格：需要确切的列名（`prompt`、`chosen`、`rejected`）
- 失败的GPU作业浪费$1-10和30-60分钟
- 在CPU上验证成本~$0.01，耗时<1分钟

### 何时验证

**始终验证：**
- 未知或自定义数据集
- DPO训练（关键 - 90%的数据集需要映射）
- 任何未明确标记为TRL兼容的数据集

**跳过已知TRL数据集的验证：**
- `trl-lib/ultrachat_200k`、`trl-lib/Capybara`、`HuggingFaceH4/ultrachat_200k`等

### 使用方法

```python
hf_jobs("uv", {
    "script": "https://huggingface.co/datasets/mcp-tools/skills/raw/main/dataset_inspector.py",
    "script_args": ["--dataset", "username/dataset-name", "--split", "train"]
})
```

脚本速度快，通常会同步完成。

### 读取结果

输出显示每种训练方法的兼容性：

- **`✓ READY`** - 数据集兼容，直接使用
- **`✗ NEEDS MAPPING`** - 兼容但需要预处理（提供映射代码）
- **`✗ INCOMPATIBLE`** - 不能用于此方法

当需要映射时，输出包括**"MAPPING CODE"**部分，包含可直接复制粘贴的Python代码。

### 示例工作流

```python
# 1. 检查数据集（成本~$0.01，CPU上<1分钟）
hf_jobs("uv", {
    "script": "https://huggingface.co/datasets/mcp-tools/skills/raw/main/dataset_inspector.py",
    "script_args": ["--dataset", "argilla/distilabel-math-preference-dpo", "--split", "train"]
})

# 2. 检查输出标记：
#    ✓ READY → 继续训练
#    ✗ NEEDS MAPPING → 应用下面的映射代码
#    ✗ INCOMPATIBLE → 选择不同方法/数据集

# 3. 如果需要映射，在训练前应用：
def format_for_dpo(example):
    return {
        'prompt': example['instruction'],
        'chosen': example['chosen_response'],
        'rejected': example['rejected_response'],
    }
dataset = dataset.map(format_for_dpo, remove_columns=dataset.column_names)

# 4. 自信启动训练作业
```

### 常见场景：DPO格式不匹配

大多数DPO数据集使用非标准列名。示例：

```
数据集有：instruction, chosen_response, rejected_response
DPO期望：prompt, chosen, rejected
```

验证器检测到这一点并提供确切的映射代码来修复它。

## 将模型转换为GGUF

训练后，将模型转换为**GGUF格式**，用于llama.cpp、Ollama、LM Studio和其他本地推理工具。

**什么是GGUF：**
- 为llama.cpp的CPU/GPU推理优化
- 支持量化（4位、5位、8位）以减少模型大小
- 兼容Ollama、LM Studio、Jan、GPT4All、llama.cpp
- 7B模型通常为2-8GB（相比未量化的14GB）

**何时转换：**
- 使用Ollama或LM Studio在本地运行模型
- 通过量化减小模型大小
- 部署到边缘设备
- 共享模型用于本地优先使用

**参见：** `references/gguf_conversion.md`获取完整转换指南，包括生产就绪的转换脚本、量化选项、硬件要求、使用示例和故障排除。

**快速转换：**
```python
hf_jobs("uv", {
    "script": "<see references/gguf_conversion.md for complete script>",
    "flavor": "a10g-large",
    "timeout": "45m",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"},
    "env": {
        "ADAPTER_MODEL": "username/my-finetuned-model",
        "BASE_MODEL": "Qwen/Qwen2.5-0.5B",
        "OUTPUT_REPO": "username/my-model-gguf"
    }
})
```

## 常见训练模式

参见`references/training_patterns.md`获取详细示例，包括：
- 快速演示（5-10分钟）
- 带检查点的生产
- 多GPU训练
- DPO训练（偏好学习）
- GRPO训练（在线RL）

## 常见失败模式

### 内存不足（OOM）

**修复（按顺序尝试）：**
1. 减少批量大小：`per_device_train_batch_size=1`，增加`gradient_accumulation_steps=8`。有效批量大小为`per_device_train_batch_size` x `gradient_accumulation_steps`。为获得最佳性能，保持有效批量大小接近128。
2. 启用：`gradient_checkpointing=True`
3. 升级硬件：t4-small → l4x1, a10g-small → a10g-large等

### 数据集格式错误

**修复：**
1. 首先使用数据集检查器验证：
   ```bash
   uv run https://huggingface.co/datasets/mcp-tools/skills/raw/main/dataset_inspector.py \
     --dataset name --split train
   ```
2. 检查输出兼容性标记（✓ READY, ✗ NEEDS MAPPING, ✗ INCOMPATIBLE）
3. 如果需要，应用检查器输出中的映射代码

### 作业超时

**修复：**
1. 检查日志中的实际运行时间：`hf_jobs("logs", {"job_id": "..."})`
2. 增加超时并添加缓冲：`"timeout": "3h"`（在估计时间上增加30%）
3. 或减少训练：降低`num_train_epochs`，使用较小的数据集，启用`max_steps`
4. 保存检查点：`save_strategy="steps"`，`save_steps=500`，`hub_strategy="every_save"`

**注意：** 默认30分钟对实际训练不足。推荐最低1-2小时。

### Hub推送失败

**修复：**
1. 添加到作业：`secrets={"HF_TOKEN": "$HF_TOKEN"}`
2. 添加到配置：`push_to_hub=True`，`hub_model_id="username/model-name"`
3. 验证认证：`mcp__huggingface__hf_whoami()`
4. 检查令牌是否具有写入权限且仓库存在（或设置`hub_private_repo=True`）

### 缺少依赖项

**修复：**
添加到PEP 723头部：
```python
# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.7.0", "trackio", "missing-package"]
# ///
```

## 故障排除

**常见问题：**
- 作业超时 → 增加超时，减少轮次/数据集，使用较小模型/LoRA
- 模型未保存到Hub → 检查push_to_hub=True，hub_model_id，secrets=HF_TOKEN
- 内存不足（OOM） → 减少批量大小，增加梯度累积，启用LoRA，使用更大的GPU
- 数据集格式错误 → 使用数据集检查器验证（见数据集验证部分）
- 导入/模块错误 → 添加带依赖项的PEP 723头部，验证格式
- 认证错误 → 检查`mcp__huggingface__hf_whoami()`，令牌权限，secrets参数

**参见：** `references/troubleshooting.md`获取完整故障排除指南

## 资源

### 参考（本技能中）
- `references/training_methods.md` - SFT、DPO、GRPO、KTO、PPO、奖励建模概述
- `references/training_patterns.md` - 常见训练模式和示例
- `references/gguf_conversion.md` - 完整的GGUF转换指南
- `references/trackio_guide.md` - Trackio监控设置
- `references/hardware_guide.md` - 硬件规格和选择
- `references/hub_saving.md` - Hub认证故障排除
- `references/troubleshooting.md` - 常见问题和解决方案

### 脚本（本技能中）
- `scripts/train_sft_example.py` - 生产SFT模板
- `scripts/train_dpo_example.py` - 生产DPO模板
- `scripts/train_grpo_example.py` - 生产GRPO模板
- `scripts/estimate_cost.py` - 估计时间和成本（适当时提供）
- `scripts/convert_to_gguf.py` - 完整的GGUF转换脚本

### 外部脚本
- [Dataset Inspector](https://huggingface.co/datasets/mcp-tools/skills/raw/main/dataset_inspector.py) - 训练前验证数据集格式（通过`uv run`或`hf_jobs`使用）

### 外部链接
- [TRL文档](https://huggingface.co/docs/trl)
- [TRL Jobs训练指南](https://huggingface.co/docs/trl/en/jobs_training)
- [TRL Jobs包](https://github.com/huggingface/trl-jobs)
- [HF Jobs文档](https://huggingface.co/docs/huggingface_hub/guides/jobs)
- [TRL示例脚本](https://github.com/huggingface/trl/tree/main/examples/scripts)
- [UV脚本指南](https://docs.astral.sh/uv/guides/scripts/)
- [UV脚本组织](https://huggingface.co/uv-scripts)

## 关键要点

1. **内联提交脚本** - `script`参数直接接受Python代码；除非用户请求，否则无需保存文件
2. **作业是异步的** - 不要等待/轮询；让用户在准备好时检查
3. **始终设置超时** - 默认30分钟不足；推荐最低1-2小时
4. **始终启用Hub推送** - 环境是临时的；不推送则所有结果丢失
5. **包含Trackio** - 使用示例脚本作为实时监控的模板
6. **提供成本估算** - 当参数已知时，使用`scripts/estimate_cost.py`
7. **使用UV脚本（方法1）** - 默认使用`hf_jobs("uv", {...})`和内联脚本；标准训练使用TRL维护的脚本；避免在Claude Code中使用bash `trl-jobs`命令
8. **使用hf_doc_fetch/hf_doc_search**获取最新的TRL文档
9. **训练前验证数据集格式** - 使用数据集检查器（见数据集验证部分）
10. **为模型大小选择适当的硬件**；对>7B模型使用LoRA