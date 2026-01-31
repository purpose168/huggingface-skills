#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
估算 TRL 任务的训练时间和成本。

使用 uv 运行：
    uv run estimate_cost.py --model <模型> --dataset <数据集> --hardware <硬件配置>

示例：
    uv run estimate_cost.py --model Qwen/Qwen2.5-0.5B --dataset trl-lib/Capybara --hardware a10g-large
"""

import argparse

# 每小时硬件成本（近似值，单位：美元）
# 这些价格基于云服务提供商的定价，实际价格可能因供应商和地区而异
HARDWARE_COSTS = {
    "t4-small": 0.75,      # NVIDIA T4 GPU，小规格配置
    "t4-medium": 1.50,     # NVIDIA T4 GPU，中等规格配置
    "l4x1": 2.50,          # NVIDIA L4 GPU，单卡配置
    "a10g-small": 3.50,   # NVIDIA A10G GPU，小规格配置
    "a10g-large": 5.00,   # NVIDIA A10G GPU，大规格配置
    "a10g-largex2": 10.00, # NVIDIA A10G GPU，双卡配置
    "a10g-largex4": 20.00, # NVIDIA A10G GPU，四卡配置
    "a100-large": 10.00,   # NVIDIA A100 GPU，大规格配置
}

# 模型大小（以十亿参数为单位）
# 这些是常见的大语言模型规模
MODEL_SIZES = {
    "0.5B": 0.5,   # 5亿参数，适合快速实验和测试
    "1.5B": 1.5,   # 15亿参数，适合轻量级应用
    "3B": 3,       # 30亿参数，中等规模模型
    "7B": 7,       # 70亿参数，通用大语言模型
    "13B": 13,     # 130亿参数，高性能大语言模型
}

def estimate_training_time(model_params, dataset_size, epochs, hardware):
    """
    估算训练时间（以小时为单位）。

    该函数基于经验观察提供粗略估算，实际训练时间会因多种因素而有所不同。

    参数：
        model_params (float): 模型参数量（单位：十亿）
        dataset_size (int): 数据集大小（样本数量）
        epochs (int): 训练轮数
        hardware (str): 硬件配置类型

    返回：
        float: 估算的训练时间（小时）

    注意：
        - 这是一个粗略估算，实际时间可能因优化程度、数据预处理等因素而变化
        - 估算基于 a10g-large 硬件配置的基准
    """
    # 基于经验观察的粗略估算
    # 这些是近似值，实际时间会有所不同

    # 基准时间：在 a10g-large 上训练 1B 模型处理 1000 个样本需要的时间（小时）
    # 这个基准值可以根据实际使用情况进行调整
    base_time_per_1k_examples = 0.1  # hours for 1B model on a10g-large

    # 根据模型大小调整时间
    # 训练时间与模型参数量、数据集大小和训练轮数成正比
    time = base_time_per_1k_examples * model_params * (dataset_size / 1000) * epochs

    # 根据硬件配置调整（相对于 a10g-large 基准）
    # 硬件乘数表示相对于基准硬件的性能差异
    # 值越小表示硬件性能越好，训练时间越短
    hardware_multipliers = {
        "t4-small": 2.0,       # T4 GPU 性能较慢，需要 2 倍时间
        "t4-medium": 1.5,      # T4 GPU 中等配置
        "l4x1": 1.2,           # L4 GPU 单卡
        "a10g-small": 1.3,     # A10G GPU 小规格
        "a10g-large": 1.0,     # A10G GPU 大规格（基准）
        "a10g-largex2": 0.6,   # A10G GPU 双卡，性能提升约 1.67 倍
        "a10g-largex4": 0.4,   # A10G GPU 四卡，性能提升约 2.5 倍
        "a100-large": 0.7,     # A100 GPU，高性能但未充分利用
    }

    # 获取对应硬件的乘数，默认为 1.0
    multiplier = hardware_multipliers.get(hardware, 1.0)
    time *= multiplier

    return time

def parse_args():
    """
    解析命令行参数。

    该函数定义并解析脚本运行所需的命令行参数。

    返回：
        argparse.Namespace: 包含解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="估算 TRL 任务的训练成本")
    parser.add_argument("--model", required=True,
                       help="模型名称或大小（例如：'Qwen/Qwen2.5-0.5B' 或 '0.5B'）")
    parser.add_argument("--dataset", required=True,
                       help="数据集名称")
    parser.add_argument("--hardware", required=True, choices=HARDWARE_COSTS.keys(),
                       help="硬件配置类型")
    parser.add_argument("--dataset-size", type=int,
                       help="覆盖数据集大小（样本数量）")
    parser.add_argument("--epochs", type=int, default=3,
                       help="训练轮数（默认：3）")
    return parser.parse_args()

def extract_model_size(model_name):
    """
    从模型名称中提取模型大小，或返回解析后的值。

    该函数尝试从模型名称字符串中提取参数量信息。

    参数：
        model_name (str): 模型名称或大小字符串

    返回：
        float: 模型参数量（单位：十亿）

    示例：
        >>> extract_model_size("Qwen/Qwen2.5-0.5B")
        0.5
        >>> extract_model_size("7B")
        7.0
    """
    # 首先尝试匹配预定义的模型大小
    for size_str, size_val in MODEL_SIZES.items():
        if size_str in model_name:
            return size_val

    # 尝试直接解析
    try:
        if "B" in model_name:
            return float(model_name.replace("B", ""))
    except:
        pass

    # 如果无法确定，默认返回 1B
    return 1.0  # Default to 1B if can't determine

def main():
    """
    主函数：执行训练成本估算流程。

    该函数协调整个估算过程：
    1. 解析命令行参数
    2. 提取模型参数信息
    3. 估算数据集大小
    4. 计算训练时间和成本
    5. 输出估算结果和建议
    """
    # 解析命令行参数
    args = parse_args()

    # 提取模型参数量
    model_params = extract_model_size(args.model)
    print(f"📊 模型：{args.model}（约 {model_params}B 参数）")

    # 估算数据集大小（实际需要加载数据集才能获得真实大小）
    if args.dataset_size:
        # 如果用户指定了数据集大小，使用用户指定的值
        dataset_size = args.dataset_size
    else:
        # 使用常见数据集大小（近似值）
        # 这些是预定义的常见数据集大小，用于快速估算
        dataset_sizes = {
            "trl-lib/Capybara": 16000,      # Capybara 数据集
            "Anthropic/hh-rlhf": 160000,    # Anthropic HH-RLHF 数据集
        }
        dataset_size = dataset_sizes.get(args.dataset, 10000)  # 默认 10000 个样本

    print(f"📦 数据集：{args.dataset}（约 {dataset_size} 个样本）")
    print(f"🔄 训练轮数：{args.epochs}")
    print(f"💻 硬件配置：{args.hardware}")
    print()

    # 估算训练时间
    estimated_hours = estimate_training_time(model_params, dataset_size, args.epochs, args.hardware)
    estimated_cost = estimated_hours * HARDWARE_COSTS[args.hardware]

    # 推荐超时时间（包含缓冲时间）
    # 增加 30% 的缓冲时间以应对不可预见的情况
    recommended_timeout_hours = estimated_hours * 1.3  # 30% buffer

    print(f"⏱️  预估训练时间：{estimated_hours:.1f} 小时")
    print(f"💰 预估成本：${estimated_cost:.2f}")
    print(f"⏰ 推荐超时时间：{recommended_timeout_hours:.1f} 小时（包含 30% 缓冲）")
    print()

    # 警告和建议
    # 当训练时间过长时，提供优化建议
    if estimated_hours > 4:
        print("⚠️  训练时间较长 - 建议考虑：")
        print("   - 使用更快的硬件配置")
        print("   - 减少训练轮数")
        print("   - 使用较小的数据集子集进行测试")

    # 当模型较大但硬件配置不足时，提供升级建议
    if model_params >= 7 and args.hardware not in ["a10g-largex2", "a10g-largex4", "a100-large"]:
        print("⚠️  大型模型 - 建议使用：")
        print("   - 更大的 GPU（a100-large）")
        print("   - 多 GPU 配置（a10g-largex2 或 a10g-largex4）")
        print("   - LoRA/PEFT 技术以提高内存效率")

    print()
    print("📋 示例任务配置：")
    print(f"""
hf_jobs("uv", {{
    "script": "your_training_script.py",
    "flavor": "{args.hardware}",
    "timeout": "{recommended_timeout_hours:.0f}h",
    "secrets": {{"HF_TOKEN": "$HF_TOKEN"}}
}})
""")

if __name__ == "__main__":
    main()
