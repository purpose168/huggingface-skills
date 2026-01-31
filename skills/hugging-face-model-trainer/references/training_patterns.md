# 常见训练模式

本指南提供了在 Hugging Face Jobs 上使用 TRL 的常见训练模式和用例。

## 多 GPU 训练

自动跨多个 GPU 进行分布式训练。TRL/Accelerate 会自动处理分布:

```python
hf_jobs("uv", {
    "script": """
# 您的训练脚本在这里(与单 GPU 相同)
# 无需更改 - Accelerate 会自动检测多个 GPU
""",
    "flavor": "a10g-largex2",  # 2x A10G GPU
    "timeout": "4h",  # 超时时间为 4 小时
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # 使用环境变量中的 HF_TOKEN
})
```

**多 GPU 训练提示:**
- 无需更改代码
- 使用 `per_device_train_batch_size`(每个 GPU,而非总数)
- 有效批次大小 = `per_device_train_batch_size` × `num_gpus` × `gradient_accumulation_steps`
- 监控 GPU 利用率以确保两个 GPU 都在使用

## DPO 训练(偏好学习)

使用偏好数据进行对齐训练:

```python
hf_jobs("uv", {
    "script": """
# /// script
# dependencies = ["trl>=0.12.0", "trackio"]
# ///

from datasets import load_dataset  # 从 Hugging Face 加载数据集
from trl import DPOTrainer, DPOConfig  # 导入 DPO 训练器和配置
import trackio  # 用于训练进度跟踪和可视化

# 加载偏好数据集(包含偏好对的数据)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# 创建训练集/验证集分割(90% 训练,10% 验证)
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)

# 配置 DPO 训练参数
config = DPOConfig(
    output_dir="dpo-model",  # 输出目录
    push_to_hub=True,  # 训练完成后推送到 Hugging Face Hub
    hub_model_id="username/dpo-model",  # Hub 上的模型 ID
    num_train_epochs=1,  # 训练轮数
    beta=0.1,  # KL 散度惩罚系数,控制模型偏离原始策略的程度
    eval_strategy="steps",  # 评估策略:按步骤评估
    eval_steps=50,  # 每 50 步评估一次
    report_to="trackio",  # 使用 trackio 记录训练指标
    run_name="baseline_run",  # 使用有意义的运行名称
    # max_length=1024,  # 默认值 - 仅在需要不同序列长度时设置
)

# 创建 DPO 训练器
trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",  # 使用指令模型作为基础模型
    train_dataset=dataset_split["train"],  # 训练数据集
    eval_dataset=dataset_split["test"],  # 验证数据集 - 重要:启用 eval_strategy 时必须提供
    args=config,  # 训练配置
)

# 开始训练
trainer.train()

# 训练完成后推送到 Hub
trainer.push_to_hub()

# 完成 trackio 记录
trackio.finish()
""",
    "flavor": "a10g-large",  # 使用单个 A10G GPU
    "timeout": "3h",  # 超时时间为 3 小时
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # 使用环境变量中的 HF_TOKEN
})
```

**DPO 文档:** 使用 `hf_doc_fetch("https://huggingface.co/docs/trl/dpo_trainer")`

## GRPO 训练(在线强化学习)

用于在线强化学习的组相对策略优化:

```python
hf_jobs("uv", {
    "script": "https://raw.githubusercontent.com/huggingface/trl/main/examples/scripts/grpo.py",  # 使用官方 GRPO 训练脚本
    "script_args": [  # 脚本参数
        "--model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct",  # 基础模型路径
        "--dataset_name", "trl-lib/math_shepherd",  # 数据集名称
        "--output_dir", "grpo-model",  # 输出目录
        "--push_to_hub",  # 推送到 Hub
        "--hub_model_id", "username/grpo-model"  # Hub 上的模型 ID
    ],
    "flavor": "a10g-large",  # 使用单个 A10G GPU
    "timeout": "4h",  # 超时时间为 4 小时
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # 使用环境变量中的 HF_TOKEN
})
```

**GRPO 文档:** 使用 `hf_doc_fetch("https://huggingface.co/docs/trl/grpo_trainer")`

## Trackio 配置

**为 trackio 设置使用合理的默认值。** 有关完整文档,包括为实验分组运行,请参阅 `references/trackio_guide.md`。

### 基本模式

```python
import trackio  # 导入 trackio 用于训练跟踪

# 初始化 trackio
trackio.init(
    project="my-training",  # 项目名称
    run_name="baseline-run",  # 运行名称 - 用户能识别的描述性名称
    space_id="username/trackio",  # Space ID - 默认为 {username}/trackio
    config={
        # 保持配置最小化 - 仅包含超参数和模型/数据集信息
        "model": "Qwen/Qwen2.5-0.5B",  # 使用的模型
        "dataset": "trl-lib/Capybara",  # 使用的数据集
        "learning_rate": 2e-5,  # 学习率
    }
)

# 您的训练代码...

# 完成 trackio 记录
trackio.finish()
```

### 实验分组(可选)

当用户想要比较相关运行时,使用 `group` 参数:

```python
# 超参数扫描
trackio.init(project="hyperparam-sweep", run_name="lr-0.001", group="lr_0.001")  # 学习率为 0.001 的运行
trackio.init(project="hyperparam-sweep", run_name="lr-0.01", group="lr_0.01")    # 学习率为 0.01 的运行
```

## 模式选择指南

| 用例 | 模式 | 硬件 | 时间 |
|----------|---------|----------|------|
| SFT 训练 | `scripts/train_sft_example.py` | a10g-large | 2-6 小时 |
| 大数据集(>10K) | 多 GPU | a10g-largex2 | 4-12 小时 |
| 偏好学习 | DPO 训练 | a10g-large | 2-4 小时 |
| 在线强化学习 | GRPO 训练 | a10g-large | 3-6 小时 |

## 关键:评估数据集要求

**⚠️ 重要**: 如果您设置了 `eval_strategy="steps"` 或 `eval_strategy="epoch"`,您**必须**向训练器提供一个 `eval_dataset`,否则训练将挂起。

### ✅ 正确 - 带有评估数据集:
```python
# 将数据集分割为训练集和验证集
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",  # 基础模型
    train_dataset=dataset_split["train"],  # 训练数据集
    eval_dataset=dataset_split["test"],  # ← 启用 eval_strategy 时必须提供验证数据集
    args=SFTConfig(eval_strategy="steps", ...),  # 按步骤进行评估
)
```

### ❌ 错误 - 将挂起:
```python
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",  # 基础模型
    train_dataset=dataset,  # 训练数据集
    # 没有提供 eval_dataset 但设置了 eval_strategy="steps" ← 将导致训练挂起
    args=SFTConfig(eval_strategy="steps", ...),
)
```

### 选项:如果没有评估数据集则禁用评估
```python
# 配置训练参数
config = SFTConfig(
    eval_strategy="no",  # ← 显式禁用评估
    # ... 其他配置
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",  # 基础模型
    train_dataset=dataset,  # 训练数据集
    # 不需要 eval_dataset
    args=config,
)
```

## 最佳实践

1. **使用训练/验证分割** - 创建评估分割以监控训练进度
2. **启用 Trackio** - 实时监控训练进度
3. **为超时时间添加 20-30% 的缓冲** - 考虑加载/保存的开销
4. **首先使用 TRL 官方脚本测试** - 在自定义代码之前使用维护的示例
5. **始终提供 eval_dataset** - 当使用 eval_strategy 时,或设置为 "no"
6. **对大模型使用多 GPU** - 7B+ 模型受益显著

## 另请参阅

- `scripts/train_sft_example.py` - 包含 Trackio 和评估分割的完整 SFT 模板
- `scripts/train_dpo_example.py` - 完整的 DPO 模板
- `scripts/train_grpo_example.py` - 完整的 GRPO 模板
- `references/hardware_guide.md` - 详细的硬件规格
- `references/training_methods.md` - 所有 TRL 训练方法的概述
- `references/troubleshooting.md` - 常见问题和解决方案
