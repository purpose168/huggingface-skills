#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.36.0",
#     "accelerate>=0.24.0",
#     "trackio",
# ]
# ///

"""
生产就绪的 SFT（监督微调）训练示例，包含所有最佳实践。

本脚本演示了：
- Trackio 集成，用于实时监控
- LoRA/PEFT 用于高效训练
- 正确的 Hub 保存配置
- 训练/评估分割，用于监控
- 检查点管理
- 优化的训练参数

使用 hf_jobs MCP 工具的用法：
    hf_jobs("uv", {
        "script": '''<粘贴整个文件内容>''',
        "flavor": "a10g-large",
        "timeout": "3h",
        "secrets": {"HF_TOKEN": "$HF_TOKEN"},
    })

或者直接内联提交脚本内容，无需保存到文件。
"""

import trackio
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# 加载数据集
print("📦 正在加载数据集...")
dataset = load_dataset("trl-lib/Capybara", split="train")
print(f"✅ 数据集已加载：{len(dataset)} 个示例")

# 创建训练/评估分割
print("🔀 正在创建训练/评估分割...")
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]
print(f"   训练集：{len(train_dataset)} 个示例")
print(f"   评估集：{len(eval_dataset)} 个示例")

# 注意：对于内存受限的演示，可以跳过评估，将完整数据集用作 train_dataset
# 并从下面的配置中移除 eval_dataset、eval_strategy 和 eval_steps

# 训练配置
config = SFTConfig(
    # 关键设置：Hub 配置
    output_dir="qwen-capybara-sft",  # 输出目录名称
    push_to_hub=True,  # 是否推送到 Hugging Face Hub
    hub_model_id="username/qwen-capybara-sft",  # Hub 上的模型 ID（需要替换为您的用户名）
    hub_strategy="every_save",  # 推送策略：每次保存时都推送检查点

    # 训练参数
    num_train_epochs=3,  # 训练轮数
    per_device_train_batch_size=4,  # 每个设备的训练批次大小
    gradient_accumulation_steps=4,  # 梯度累积步数（有效批次大小 = per_device_train_batch_size * gradient_accumulation_steps）
    learning_rate=2e-5,  # 学习率
    # max_length=1024,  # 最大序列长度（默认值 - 仅在需要不同的序列长度时设置）

    # 日志记录和检查点保存
    logging_steps=10,  # 每隔多少步记录一次日志
    save_strategy="steps",  # 保存策略：按步数保存
    save_steps=100,  # 每隔多少步保存一次检查点
    save_total_limit=2,  # 最多保留的检查点数量

    # 评估 - 重要：仅当提供了 eval_dataset 时才启用
    eval_strategy="steps",  # 评估策略：按步数评估
    eval_steps=100,  # 每隔多少步评估一次

    # 优化设置
    warmup_ratio=0.1,  # 预热比例（总训练步数的 10% 用于学习率预热）
    lr_scheduler_type="cosine",  # 学习率调度器类型：余弦退火

    # 监控设置
    report_to="trackio",  # 集成 Trackio 进行监控
    project="meaningful_project_name",  # 项目名称（用于 Trackio，建议使用有意义的名称）
    run_name="baseline-run",  # 本次训练运行的描述性名称
)

# LoRA（低秩自适应）配置
# LoRA 是一种参数高效的微调方法，通过添加低秩矩阵来减少可训练参数数量
peft_config = LoraConfig(
    r=16,  # LoRA 秩（低秩矩阵的维度），值越大可训练参数越多，但显存占用也越大
    lora_alpha=32,  # LoRA 缩放因子（通常设置为 r 的 2 倍），用于控制 LoRA 更新的权重
    lora_dropout=0.05,  # LoRA 层的 dropout 概率，用于防止过拟合
    bias="none",  # 偏置项的训练方式："none"（不训练）、"all"（全部训练）、"lora_only"（仅训练 LoRA 层的偏置）
    task_type="CAUSAL_LM",  # 任务类型：因果语言模型（用于生成式模型）
    target_modules=["q_proj", "v_proj"],  # 要应用 LoRA 的目标模块（通常选择注意力机制中的查询和值投影层）
)

# 初始化训练器并开始训练
print("🎯 正在初始化训练器...")
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",  # 基础模型：Qwen 2.5 0.5B 参数版本
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=eval_dataset,  # 评估数据集（关键：当启用 eval_strategy 时必须提供）
    args=config,  # 训练配置
    peft_config=peft_config,  # LoRA/PEFT 配置
)

print("🚀 开始训练...")
trainer.train()

print("💾 正在推送到 Hub...")
trainer.push_to_hub()

# 完成 Trackio 跟踪
trackio.finish()

print("✅ 完成！模型位于：https://huggingface.co/username/qwen-capybara-sft")
print("📊 查看指标：https://huggingface.co/spaces/username/trackio")
