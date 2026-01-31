# TRL 训练方法概述

TRL (Transformer Reinforcement Learning) 提供多种训练方法用于微调和对齐语言模型。本参考文档简要概述了每种方法。

## 监督微调 (SFT)

**什么是:** 使用监督学习在演示数据上进行标准指令微调。

**何时使用:**
- 在任务特定数据上对基础模型进行初始微调
- 教授新的能力或领域
- 最常见的微调起点

**数据集格式:** 带有 "messages" 字段的对话格式,或文本字段,或提示/完成对

**示例:**
```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="my-model",
        push_to_hub=True,
        hub_model_id="username/my-model",
        eval_strategy="no",  # 为简单示例禁用评估
        # max_length=1024 是默认值 - 仅在需要不同长度时设置
    )
)
trainer.train()
```

**注意:** 对于带有评估监控的生产训练,请参见 `scripts/train_sft_example.py`

**文档:** `hf_doc_fetch("https://huggingface.co/docs/trl/sft_trainer")`

## 直接偏好优化 (DPO)

**什么是:** 直接在偏好对(选中 vs 拒绝的响应)上进行训练的对齐方法,无需奖励模型。

**何时使用:**
- 将模型对齐到人类偏好
- 在 SFT 后提高响应质量
- 拥有成对偏好数据(选中/拒绝的响应)

**数据集格式:** 带有 "chosen" 和 "rejected" 字段的偏好对

**示例:**
```python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",  # 使用指令模型
    train_dataset=dataset,
    args=DPOConfig(
        output_dir="dpo-model",
        beta=0.1,  # KL 惩罚系数
        eval_strategy="no",  # 为简单示例禁用评估
        # max_length=1024 是默认值 - 仅在需要不同长度时设置
    )
)
trainer.train()
```

**注意:** 对于带有评估监控的生产训练,请参见 `scripts/train_dpo_example.py`

**文档:** `hf_doc_fetch("https://huggingface.co/docs/trl/dpo_trainer")`

## 组相对策略优化 (GRPO)

**什么是:** 相对于组性能进行优化的在线强化学习方法,适用于具有可验证奖励的任务。

**何时使用:**
- 具有自动奖励信号的任务(代码执行、数学验证)
- 在线学习场景
- 当 DPO 离线数据不足时

**数据集格式:** 仅提示格式(模型生成响应,奖励在线计算)

**示例:**
```python
# 使用 TRL 维护的脚本
hf_jobs("uv", {
    "script": "https://raw.githubusercontent.com/huggingface/trl/main/examples/scripts/grpo.py",
    "script_args": [
        "--model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct",
        "--dataset_name", "trl-lib/math_shepherd",
        "--output_dir", "grpo-model"
    ],
    "flavor": "a10g-large",
    "timeout": "4h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**文档:** `hf_doc_fetch("https://huggingface.co/docs/trl/grpo_trainer")`

## 奖励建模

**什么是:** 训练奖励模型来对响应进行评分,用作 RLHF 流水线中的组件。

**何时使用:**
- 构建 RLHF 流水线
- 需要自动质量评分
- 为 PPO 训练创建奖励信号

**数据集格式:** 带有 "chosen" 和 "rejected" 响应的偏好对

**文档:** `hf_doc_fetch("https://huggingface.co/docs/trl/reward_trainer")`

## 方法选择指南

| 方法 | 复杂度 | 所需数据 | 使用场景 |
|--------|-----------|---------------|----------|
| **SFT** | 低 | 演示数据 | 初始微调 |
| **DPO** | 中等 | 成对偏好 | SFT 后对齐 |
| **GRPO** | 中等 | 提示 + 奖励函数 | 带自动奖励的在线 RL |
| **Reward** | 中等 | 成对偏好 | 构建 RLHF 流水线 |

## 推荐流水线

**对于大多数用例:**
1. **从 SFT 开始** - 在任务数据上微调基础模型
2. **接着使用 DPO** - 使用成对数据对齐偏好
3. **可选: GGUF 转换** - 部署用于本地推理

**对于高级 RL 场景:**
1. **从 SFT 开始** - 微调基础模型
2. **训练奖励模型** - 在偏好数据上

## 数据集格式参考

对于完整的数据集格式规范,请使用:
```python
hf_doc_fetch("https://huggingface.co/docs/trl/dataset_formats")
```

或验证您的数据集:
```bash
uv run https://huggingface.co/datasets/mcp-tools/skills/raw/main/dataset_inspector.py \
  --dataset your/dataset --split train
```

## 另请参阅

- `references/training_patterns.md` - 常见训练模式和示例
- `scripts/train_sft_example.py` - 完整的 SFT 模板
- `scripts/train_dpo_example.py` - 完整的 DPO 模板
- [Dataset Inspector](https://huggingface.co/datasets/mcp-tools/skills/raw/main/dataset_inspector.py) - 数据集格式验证工具
