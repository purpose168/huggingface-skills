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
用于在线强化学习的生产级 GRPO 训练示例。

GRPO (Group Relative Policy Optimization, 组相对策略优化) 是一种在线强化学习方法,
它相对于组性能进行优化。最适合具有自动奖励信号的任务,如代码执行或数学验证。

使用 hf_jobs MCP 工具的用法:
    hf_jobs("uv", {
        "script": '''<粘贴整个文件内容>''',
        "flavor": "a10g-large",
        "timeout": "4h",
        "secrets": {"HF_TOKEN": "$HF_TOKEN"},
    })

或者直接内联提交脚本内容而无需保存到文件。

注意: 对于大多数 GRPO 使用场景,推荐使用 TRL 维护的脚本:
    https://raw.githubusercontent.com/huggingface/trl/main/examples/scripts/grpo.py
"""

import trackio  # 导入 trackio 库用于训练监控和指标追踪
from datasets import load_dataset  # 导入 Hugging Face datasets 库用于加载数据集
from trl import GRPOTrainer, GRPOConfig  # 导入 TRL 库中的 GRPO 训练器和配置类


# 加载数据集 (GRPO 使用仅包含提示词的格式)
# math_shepherd 数据集包含数学问题,适合用于验证 GRPO 在数学推理任务上的表现
dataset = load_dataset("trl-lib/math_shepherd", split="train")
print(f"✅ 数据集已加载: {len(dataset)} 个提示词")

# 训练配置
config = GRPOConfig(
    # 关键设置: Hub (Hugging Face Hub) 配置
    output_dir="qwen-grpo-math",  # 输出目录,用于保存训练结果
    push_to_hub=True,  # 是否将模型推送到 Hugging Face Hub
    hub_model_id="username/qwen-grpo-math",  # Hub 上的模型 ID (需要替换为您的用户名)
    hub_strategy="every_save",  # 推送策略: 每次保存时都推送到 Hub

    # 训练参数
    num_train_epochs=1,  # 训练轮数: 完整遍历数据集 1 次
    per_device_train_batch_size=4,  # 每个设备(如 GPU)的训练批次大小
    gradient_accumulation_steps=4,  # 梯度累积步数: 累积 4 个批次后再更新一次梯度,相当于有效批次大小为 16
    learning_rate=1e-6,  # 学习率: 0.000001,较小的学习率有助于稳定训练

    # 日志记录和检查点保存
    logging_steps=10,  # 每 10 步记录一次日志
    save_strategy="steps",  # 按步数保存检查点
    save_steps=100,  # 每 100 步保存一次检查点
    save_total_limit=2,  # 最多保留 2 个检查点,自动删除旧的检查点以节省空间

    # 优化器设置
    warmup_ratio=0.1,  # 预热比例: 前 10% 的训练步数使用线性预热,学习率从 0 逐渐增加到设定值
    lr_scheduler_type="cosine",  # 学习率调度器类型: 余弦退火,使学习率在训练过程中平滑下降

    # 监控设置
    report_to="trackio",  # 集成 Trackio 进行训练监控和指标追踪
    project="meaningful_project_name",  # 项目名称,用于在 trackio 中组织训练任务
    run_name="baseline-run",  # 本次训练运行的描述性名称,便于区分不同的实验

)

# 初始化并开始训练
# 注意: GRPO 需要一个经过指令微调的模型作为基础模型
# Qwen2.5-0.5B-Instruct 是一个已经过指令微调的小型模型,适合快速实验
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",  # 基础模型路径或 Hub ID
    train_dataset=dataset,  # 训练数据集
    args=config,  # 训练配置
)

print("🚀 开始 GRPO 训练...")
trainer.train()  # 执行训练

print("💾 正在推送到 Hub...")
trainer.push_to_hub()  # 将训练好的模型推送到 Hugging Face Hub


print("✅ 完成! 模型位置: https://huggingface.co/username/qwen-grpo-math")
print("📊 查看指标: https://huggingface.co/spaces/username/trackio")
