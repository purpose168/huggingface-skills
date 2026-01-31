#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "transformers>=4.36.0",
#     "accelerate>=0.24.0",
#     "trackio",
# ]
# ///

"""
生产级 DPO 训练示例，用于偏好学习。

DPO（直接偏好优化）在偏好对（chosen vs rejected responses，即选中与拒绝的响应）上训练模型，
无需奖励模型（reward model）。

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
from trl import DPOTrainer, DPOConfig


# 加载偏好数据集
# 偏好数据集包含成对的样本，每个样本包含一个"选中"（chosen）响应和一个"拒绝"（rejected）响应
print("📦 正在加载数据集...")
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
print(f"✅ 数据集加载完成：{len(dataset)} 个偏好对")

# 创建训练集/验证集划分
# 将数据集按 90:10 的比例划分为训练集和验证集，seed=42 确保可重复性
print("🔀 正在创建训练集/验证集划分...")
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]  # 训练集，包含 90% 的数据
eval_dataset = dataset_split["test"]   # 验证集，包含 10% 的数据
print(f"   训练集：{len(train_dataset)} 个偏好对")
print(f"   验证集：{len(eval_dataset)} 个偏好对")

# 训练配置
# DPOConfig 包含了 DPO 训练所需的所有超参数和设置
config = DPOConfig(
    # 关键设置：Hub 配置
    # output_dir: 模型输出目录，用于保存训练过程中的检查点和最终模型
    output_dir="qwen-dpo-aligned",
    # push_to_hub: 是否将模型推送到 Hugging Face Hub
    push_to_hub=True,
    # hub_model_id: Hub 上的模型 ID，格式为 "username/model-name"
    hub_model_id="username/qwen-dpo-aligned",
    # hub_strategy: 推送策略，"every_save" 表示每次保存检查点时都推送到 Hub
    hub_strategy="every_save",

    # DPO 特定参数
    # beta: KL 散度惩罚系数，控制模型与参考模型的偏离程度
    # 较高的 beta 值会使模型更接近参考模型，较低的 beta 值允许更大的偏离
    beta=0.1,  # KL 惩罚系数（值越高 = 越接近参考模型）

    # 训练参数
    # num_train_epochs: 训练轮数，DPO 通常需要的轮数比 SFT（监督微调）少
    num_train_epochs=1,  # DPO 通常需要的轮数比 SFT 少
    # per_device_train_batch_size: 每个设备（GPU/CPU）上的训练批次大小
    per_device_train_batch_size=4,
    # gradient_accumulation_steps: 梯度累积步数，用于模拟更大的批次大小
    # 实际批次大小 = per_device_train_batch_size * gradient_accumulation_steps * num_devices
    gradient_accumulation_steps=4,
    # learning_rate: 学习率，DPO 使用的学习率通常比 SFT 低得多
    learning_rate=5e-7,  # DPO 使用的学习率比 SFT 低得多
    # max_length=1024,  # 默认值 - 仅在需要不同的序列长度时设置

    # 日志记录和检查点保存
    # logging_steps: 每隔多少步记录一次训练日志
    logging_steps=10,
    # save_strategy: 保存策略，"steps" 表示按步数保存
    save_strategy="steps",
    # save_steps: 每隔多少步保存一次检查点
    save_steps=100,
    # save_total_limit: 最多保留多少个检查点，旧的会被删除
    save_total_limit=2,

    # 评估 - 重要提示：仅在提供 eval_dataset 时启用
    # eval_strategy: 评估策略，"steps" 表示按步数进行评估
    eval_strategy="steps",
    # eval_steps: 每隔多少步进行一次评估
    eval_steps=100,

    # 优化器设置
    # warmup_ratio: 学习率预热比例，在前 warmup_ratio * total_steps 步中线性增加学习率
    warmup_ratio=0.1,
    # lr_scheduler_type: 学习率调度器类型，"cosine" 表示余弦退火调度器
    lr_scheduler_type="cosine",

    # 监控和日志
    # report_to: 报告目标，"trackio" 表示将训练指标报告到 Trackio 平台
    report_to="trackio",  # 与 Trackio 集成
    # project: 项目名称，用于在 Trackio 中组织训练任务
    project="meaningful_project_name", # 训练的项目名称（trackio）
    # run_name: 运行名称，用于标识这次特定的训练运行
    run_name="baseline-run", # 这次训练运行的描述性名称

)

# 初始化并开始训练
# 注意：DPO 需要一个经过指令微调（instruct-tuned）的模型作为基础模型
# 基础模型应该已经理解如何遵循指令，DPO 将进一步优化其响应质量
print("🎯 正在初始化训练器...")
trainer = DPOTrainer(
    # model: 基础模型路径，必须使用 instruct 模型，而不是 base 模型
    # instruct 模型已经过指令微调，能够理解并遵循指令
    model="Qwen/Qwen2.5-0.5B-Instruct",  # 使用 instruct 模型，而不是 base 模型
    # train_dataset: 训练数据集，包含偏好对
    train_dataset=train_dataset,
    # eval_dataset: 验证数据集，用于评估模型性能
    # 关键提示：当启用 eval_strategy 时必须提供 eval_dataset
    eval_dataset=eval_dataset,  # 关键提示：启用 eval_strategy 时必须提供 eval_dataset
    # args: 训练配置对象
    args=config,
)

print("🚀 正在启动 DPO 训练...")
# 开始训练过程
trainer.train()

print("💾 正在推送到 Hub...")
# 将训练好的模型推送到 Hugging Face Hub
trainer.push_to_hub()

# 完成 Trackio 跟踪
# 结束 Trackio 的实验跟踪，确保所有指标和日志都已保存
trackio.finish()

print("✅ 训练完成！模型地址：https://huggingface.co/username/qwen-dpo-aligned")
print("📊 查看训练指标：https://huggingface.co/spaces/username/trackio")
