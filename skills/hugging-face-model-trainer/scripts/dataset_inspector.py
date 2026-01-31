#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
用于 TRL 训练的数据集格式检查器（针对大语言模型优化的输出）

检查 Hugging Face 数据集以确定 TRL 训练的兼容性。
使用数据集服务器 API 获取即时结果 - 无需下载数据集！

超高效：使用 HF 数据集服务器 API - 在 2 秒内完成。

在 HF Jobs 中使用：
    hf_jobs("uv", {
        "script": "https://huggingface.co/datasets/evalstate/trl-helpers/raw/main/dataset_inspector.py",
        "script_args": ["--dataset", "your/dataset", "--split", "train"]
    })
"""

import argparse  # 命令行参数解析模块
import sys  # 系统相关功能模块
import json  # JSON 数据处理模块
import urllib.request  # HTTP 请求模块
import urllib.parse  # URL 解析模块
from typing import List, Dict, Any  # 类型提示模块


def parse_args():
    """
    解析命令行参数

    Returns:
        argparse.Namespace: 包含解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="检查 TRL 训练的数据集格式")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--split", type=str, default="train", help="数据集划分（默认：train）")
    parser.add_argument("--config", type=str, default="default", help="数据集配置名称（默认：default）")
    parser.add_argument("--preview", type=int, default=150, help="每个字段预览的最大字符数")
    parser.add_argument("--samples", type=int, default=5, help="要获取的样本数量（默认：5）")
    parser.add_argument("--json-output", action="store_true", help="以 JSON 格式输出")
    return parser.parse_args()


def api_request(url: str) -> Dict:
    """
    向数据集服务器发送 API 请求

    Args:
        url (str): API 请求的 URL 地址

    Returns:
        Dict: API 返回的 JSON 数据，解析为字典

    Raises:
        Exception: 当 API 请求失败时抛出异常
    """
    try:
        # 使用 urlopen 发送 HTTP GET 请求，设置 10 秒超时
        with urllib.request.urlopen(url, timeout=10) as response:
            # 读取响应内容并解码为 JSON
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        # 处理 404 错误（资源未找到）
        if e.code == 404:
            return None
        # 其他 HTTP 错误抛出异常
        raise Exception(f"API 请求失败：{e.code} {e.reason}")
    except Exception as e:
        # 其他异常处理
        raise Exception(f"API 请求失败：{str(e)}")


def get_splits(dataset: str) -> Dict:
    """
    获取数据集的可用划分信息

    Args:
        dataset (str): Hugging Face 数据集名称

    Returns:
        Dict: 包含数据集划分信息的字典
    """
    # 构建查询数据集划分的 API URL
    url = f"https://datasets-server.huggingface.co/splits?dataset={urllib.parse.quote(dataset)}"
    return api_request(url)


def get_rows(dataset: str, config: str, split: str, offset: int = 0, length: int = 5) -> Dict:
    """
    从数据集中获取行数据

    Args:
        dataset (str): Hugging Face 数据集名称
        config (str): 数据集配置名称
        split (str): 数据集划分名称（如 train、test、validation）
        offset (int): 起始偏移量，默认为 0
        length (int): 要获取的行数，默认为 5

    Returns:
        Dict: 包含数据集行信息的字典
    """
    # 构建查询数据集行的 API URL，包含所有必要参数
    url = f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(dataset)}&config={config}&split={split}&offset={offset}&length={length}"
    return api_request(url)


def find_columns(columns: List[str], patterns: List[str]) -> List[str]:
    """
    在列名列表中查找匹配指定模式的列

    Args:
        columns (List[str]): 数据集的所有列名列表
        patterns (List[str]): 要匹配的模式列表（不区分大小写）

    Returns:
        List[str]: 匹配的列名列表
    """
    # 列表推导式：返回所有包含任一模式（不区分大小写）的列名
    return [c for c in columns if any(p in c.lower() for p in patterns)]


def check_sft_compatibility(columns: List[str]) -> Dict[str, Any]:
    """
    检查数据集是否与 SFT（监督微调）训练兼容

    SFT 训练需要以下任一字段：
    - messages: 对话消息格式
    - text: 纯文本格式
    - prompt + completion: 提示词和完成文本

    Args:
        columns (List[str]): 数据集的所有列名列表

    Returns:
        Dict[str, Any]: 包含 SFT 兼容性检查结果的字典，包括：
            - ready (bool): 是否直接可用于 SFT 训练
            - reason (str): 兼容的原因（messages/text/prompt+completion）
            - possible_prompt (str): 可能的提示词列名
            - possible_response (str): 可能的响应列名
            - has_context (bool): 是否包含上下文字段
    """
    # 检查是否存在标准字段
    has_messages = "messages" in columns  # 对话消息格式
    has_text = "text" in columns  # 纯文本格式
    has_prompt_completion = "prompt" in columns and "completion" in columns  # 提示词+完成格式

    # 判断是否可直接用于 SFT 训练
    ready = has_messages or has_text or has_prompt_completion

    # 查找可能的提示词列（支持多种命名方式）
    possible_prompt = find_columns(columns, ["prompt", "instruction", "question", "input"])

    # 查找可能的响应列（支持多种命名方式）
    possible_response = find_columns(columns, ["response", "completion", "output", "answer"])

    return {
        "ready": ready,
        "reason": "messages" if has_messages else "text" if has_text else "prompt+completion" if has_prompt_completion else None,
        "possible_prompt": possible_prompt[0] if possible_prompt else None,
        "possible_response": possible_response[0] if possible_response else None,
        "has_context": "context" in columns,  # 是否包含上下文信息
    }


def check_dpo_compatibility(columns: List[str]) -> Dict[str, Any]:
    """
    检查数据集是否与 DPO（直接偏好优化）训练兼容

    DPO 训练需要以下字段：
    - prompt: 提示词
    - chosen: 偏好的响应（更好的回答）
    - rejected: 不偏好的响应（较差的回答）

    Args:
        columns (List[str]): 数据集的所有列名列表

    Returns:
        Dict[str, Any]: 包含 DPO 兼容性检查结果的字典，包括：
            - ready (bool): 是否直接可用于 DPO 训练
            - can_map (bool): 是否可以通过字段映射来使用
            - prompt_col (str): 提示词列名
            - chosen_col (str): 偏好响应列名
            - rejected_col (str): 不偏好响应列名
    """
    # 检查是否存在标准 DPO 字段
    has_standard = "prompt" in columns and "chosen" in columns and "rejected" in columns

    # 查找可能的提示词列
    possible_prompt = find_columns(columns, ["prompt", "instruction", "question", "input"])

    # 查找可能的偏好响应列
    possible_chosen = find_columns(columns, ["chosen", "preferred", "winner"])

    # 查找可能的不偏好响应列
    possible_rejected = find_columns(columns, ["rejected", "dispreferred", "loser"])

    # 判断是否可以通过映射来使用
    can_map = bool(possible_prompt and possible_chosen and possible_rejected)

    return {
        "ready": has_standard,
        "can_map": can_map,
        "prompt_col": possible_prompt[0] if possible_prompt else None,
        "chosen_col": possible_chosen[0] if possible_chosen else None,
        "rejected_col": possible_rejected[0] if possible_rejected else None,
    }


def check_grpo_compatibility(columns: List[str]) -> Dict[str, Any]:
    """
    检查数据集是否与 GRPO（群体相对策略优化）训练兼容

    GRPO 训练需要：
    - prompt: 提示词
    - 不应包含 chosen 或 rejected 字段（这些是 DPO 专用的）

    Args:
        columns (List[str]): 数据集的所有列名列表

    Returns:
        Dict[str, Any]: 包含 GRPO 兼容性检查结果的字典，包括：
            - ready (bool): 是否直接可用于 GRPO 训练
            - can_map (bool): 是否可以通过字段映射来使用
            - prompt_col (str): 提示词列名
    """
    # 检查是否有提示词字段，且没有 DPO 专用字段
    has_prompt = "prompt" in columns
    has_no_responses = "chosen" not in columns and "rejected" not in columns

    # 查找可能的提示词列
    possible_prompt = find_columns(columns, ["prompt", "instruction", "question", "input"])

    return {
        "ready": has_prompt and has_no_responses,
        "can_map": bool(possible_prompt) and has_no_responses,
        "prompt_col": possible_prompt[0] if possible_prompt else None,
    }


def check_kto_compatibility(columns: List[str]) -> Dict[str, Any]:
    """
    检查数据集是否与 KTO（卡尼曼-塔克斯基优化）训练兼容

    KTO 训练需要以下字段：
    - prompt: 提示词
    - completion: 完成文本
    - label: 标签（表示偏好或质量）

    Args:
        columns (List[str]): 数据集的所有列名列表

    Returns:
        Dict[str, Any]: 包含 KTO 兼容性检查结果的字典，包括：
            - ready (bool): 是否直接可用于 KTO 训练
    """
    # KTO 需要三个特定字段同时存在
    return {"ready": "prompt" in columns and "completion" in columns and "label" in columns}


def generate_mapping_code(method: str, info: Dict[str, Any]) -> str:
    """
    为指定的训练方法生成数据映射代码

    当数据集格式不完全符合训练要求时，生成映射代码将数据转换为标准格式。

    Args:
        method (str): 训练方法名称（SFT/DPO/GRPO）
        info (Dict[str, Any]): 兼容性检查信息字典

    Returns:
        str: 生成的映射代码字符串，如果不需要映射则返回 None
    """
    # 处理 SFT 训练的映射代码生成
    if method == "SFT":
        # 如果数据集已经准备好，无需映射
        if info["ready"]:
            return None

        # 获取可能的列名
        prompt_col = info.get("possible_prompt")
        response_col = info.get("possible_response")
        has_context = info.get("has_context", False)

        # 如果没有提示词列，无法生成映射
        if not prompt_col:
            return None

        # 根据不同的字段组合生成不同的映射代码
        if has_context and response_col:
            # 有上下文和响应的完整格式
            return f"""def format_for_sft(example):
    text = f"Instruction: {{example['{prompt_col}']}}\\n\\n"
    if example.get('context'):
        text += f"Context: {{example['context']}}\\n\\n"
    text += f"Response: {{example['{response_col}']}}"
    return {{'text': text}}

dataset = dataset.map(format_for_sft, remove_columns=dataset.column_names)"""
        elif response_col:
            # 有提示词和响应的格式
            return f"""def format_for_sft(example):
    return {{'text': f"{{example['{prompt_col}']}}\\n\\n{{example['{response_col}']}}}}

dataset = dataset.map(format_for_sft, remove_columns=dataset.column_names)"""
        else:
            # 只有提示词的格式
            return f"""def format_for_sft(example):
    return {{'text': example['{prompt_col}']}}

dataset = dataset.map(format_for_sft, remove_columns=dataset.column_names)}"""

    # 处理 DPO 训练的映射代码生成
    elif method == "DPO":
        # 如果数据集已经准备好或无法映射，无需生成代码
        if info["ready"] or not info["can_map"]:
            return None

        # 生成 DPO 格式的映射代码
        return f"""def format_for_dpo(example):
    return {{
        'prompt': example['{info['prompt_col']}'],
        'chosen': example['{info['chosen_col']}'],
        'rejected': example['{info['rejected_col']}'],
    }}

dataset = dataset.map(format_for_dpo, remove_columns=dataset.column_names)"""

    # 处理 GRPO 训练的映射代码生成
    elif method == "GRPO":
        # 如果数据集已经准备好或无法映射，无需生成代码
        if info["ready"] or not info["can_map"]:
            return None

        # 生成 GRPO 格式的映射代码
        return f"""def format_for_grpo(example):
    return {{'prompt': example['{info['prompt_col}']}}

dataset = dataset.map(format_for_grpo, remove_columns=dataset.column_names)}"""

    return None


def format_value_preview(value: Any, max_chars: int) -> str:
    """
    格式化值的预览显示

    根据值的类型和长度生成适合显示的预览字符串。

    Args:
        value (Any): 要格式化的值，可以是任意类型
        max_chars (int): 最大显示字符数

    Returns:
        str: 格式化后的预览字符串
    """
    # 处理 None 值
    if value is None:
        return "None"
    # 处理字符串类型
    elif isinstance(value, str):
        return value[:max_chars] + ("..." if len(value) > max_chars else "")
    # 处理列表类型
    elif isinstance(value, list):
        # 如果是字典列表，显示项数和键
        if len(value) > 0 and isinstance(value[0], dict):
            return f"[{len(value)} items] Keys: {list(value[0].keys())}"
        # 普通列表，转换为字符串并截断
        preview = str(value)
        return preview[:max_chars] + ("..." if len(preview) > max_chars else "")
    # 处理其他类型
    else:
        preview = str(value)
        return preview[:max_chars] + ("..." if len(preview) > max_chars else "")


def main():
    """
    主函数：执行数据集检查流程

    流程：
    1. 解析命令行参数
    2. 通过数据集服务器 API 获取数据集信息
    3. 检查各种训练方法的兼容性
    4. 生成并输出检查结果（JSON 或人类可读格式）
    """
    # 解析命令行参数
    args = parse_args()

    print(f"通过数据集服务器 API 获取数据集信息...")

    try:
        # 步骤 1: 获取数据集的划分信息
        splits_data = get_splits(args.dataset)
        if not splits_data or "splits" not in splits_data:
            print(f"错误：无法获取数据集 '{args.dataset}' 的划分信息")
            print(f"       数据集可能不存在或无法通过数据集服务器 API 访问")
            sys.exit(1)

        # 步骤 2: 查找正确的配置
        available_configs = set()  # 可用的配置集合
        split_found = False  # 是否找到指定的划分
        config_to_use = args.config  # 要使用的配置

        # 遍历所有划分信息
        for split_info in splits_data["splits"]:
            available_configs.add(split_info["config"])
            # 检查是否找到指定的配置和划分
            if split_info["config"] == args.config and split_info["split"] == args.split:
                split_found = True

        # 如果默认配置未找到，尝试使用第一个可用配置
        if not split_found and available_configs:
            config_to_use = list(available_configs)[0]
            print(f"配置 '{args.config}' 未找到，尝试使用 '{config_to_use}'...")

        # 步骤 3: 获取数据行
        rows_data = get_rows(args.dataset, config_to_use, args.split, offset=0, length=args.samples)

        if not rows_data or "rows" not in rows_data:
            print(f"错误：无法获取数据集 '{args.dataset}' 的行数据")
            print(f"       划分 '{args.split}' 可能不存在")
            print(f"       可用的配置：{', '.join(sorted(available_configs))}")
            sys.exit(1)

        rows = rows_data["rows"]
        if not rows:
            print(f"错误：划分 '{args.split}' 中未找到行数据")
            sys.exit(1)

        # 步骤 4: 从第一行提取列信息
        first_row = rows[0]["row"]
        columns = list(first_row.keys())  # 所有列名
        features = rows_data.get("features", [])  # 特征信息（包含类型）

        # 步骤 5: 获取总样本数（如果可用）
        total_examples = "Unknown"  # 默认为未知
        for split_info in splits_data["splits"]:
            if split_info["config"] == config_to_use and split_info["split"] == args.split:
                # 格式化数字，添加千位分隔符
                total_examples = f"{split_info.get('num_examples', 'Unknown'):,}" if isinstance(split_info.get('num_examples'), int) else "Unknown"
                break

    except Exception as e:
        print(f"错误：{str(e)}")
        sys.exit(1)

    # 步骤 6: 运行兼容性检查
    sft_info = check_sft_compatibility(columns)  # SFT 兼容性
    dpo_info = check_dpo_compatibility(columns)  # DPO 兼容性
    grpo_info = check_grpo_compatibility(columns)  # GRPO 兼容性
    kto_info = check_kto_compatibility(columns)  # KTO 兼容性

    # 步骤 7: 确定推荐的训练方法
    recommended = []
    if sft_info["ready"]:
        recommended.append("SFT")
    elif sft_info["possible_prompt"]:
        recommended.append("SFT (需要映射)")

    if dpo_info["ready"]:
        recommended.append("DPO")
    elif dpo_info["can_map"]:
        recommended.append("DPO (需要映射)")

    if grpo_info["ready"]:
        recommended.append("GRPO")
    elif grpo_info["can_map"]:
        recommended.append("GRPO (需要映射)")

    if kto_info["ready"]:
        recommended.append("KTO")

    # 步骤 8: JSON 输出模式
    if args.json_output:
        result = {
            "dataset": args.dataset,
            "config": config_to_use,
            "split": args.split,
            "total_examples": total_examples,
            "columns": columns,
            "features": [{"name": f["name"], "type": f["type"]} for f in features] if features else [],
            "compatibility": {
                "SFT": sft_info,
                "DPO": dpo_info,
                "GRPO": grpo_info,
                "KTO": kto_info,
            },
            "recommended_methods": recommended,
        }
        print(json.dumps(result, indent=2))
        sys.exit(0)

    # 步骤 9: 人类可读的输出格式（针对大语言模型解析优化）
    print("=" * 80)
    print(f"数据集检查结果")
    print("=" * 80)

    print(f"\n数据集：{args.dataset}")
    print(f"配置：{config_to_use}")
    print(f"划分：{args.split}")
    print(f"总样本数：{total_examples}")
    print(f"获取的样本数：{len(rows)}")

    print(f"\n{'列信息':-<80}")
    if features:
        # 显示列名和类型
        for feature in features:
            print(f"  {feature['name']}: {feature['type']}")
    else:
        # 如果类型信息不可用，只显示列名
        for col in columns:
            print(f"  {col}: (类型信息不可用)")

    print(f"\n{'示例数据':-<80}")
    example = first_row
    for col in columns:
        value = example.get(col)
        display = format_value_preview(value, args.preview)
        print(f"\n{col}:")
        print(f"  {display}")

    print(f"\n{'训练方法兼容性':-<80}")

    # SFT 检查结果
    print(f"\n[SFT] {'✓ 就绪' if sft_info['ready'] else '✗ 需要映射'}")
    if sft_info["ready"]:
        print(f"  原因：数据集包含 '{sft_info['reason']}' 字段")
        print(f"  操作：可直接与 SFTTrainer 一起使用")
    elif sft_info["possible_prompt"]:
        print(f"  检测到：prompt='{sft_info['possible_prompt']}' response='{sft_info['possible_response']}'")
        print(f"  操作：应用映射代码（见下方）")
    else:
        print(f"  状态：无法确定映射 - 需要手动检查")

    # DPO 检查结果
    print(f"\n[DPO] {'✓ 就绪' if dpo_info['ready'] else '✗ 需要映射' if dpo_info['can_map'] else '✗ 不兼容'}")
    if dpo_info["ready"]:
        print(f"  原因：数据集包含 'prompt'、'chosen'、'rejected' 字段")
        print(f"  操作：可直接与 DPOTrainer 一起使用")
    elif dpo_info["can_map"]:
        print(f"  检测到：prompt='{dpo_info['prompt_col']}' chosen='{dpo_info['chosen_col']}' rejected='{dpo_info['rejected_col']}'")
        print(f"  操作：应用映射代码（见下方）")
    else:
        print(f"  状态：缺少必需字段（prompt + chosen + rejected）")

    # GRPO 检查结果
    print(f"\n[GRPO] {'✓ 就绪' if grpo_info['ready'] else '✗ 需要映射' if grpo_info['can_map'] else '✗ 不兼容'}")
    if grpo_info["ready"]:
        print(f"  原因：数据集包含 'prompt' 字段")
        print(f"  操作：可直接与 GRPOTrainer 一起使用")
    elif grpo_info["can_map"]:
        print(f"  检测到：prompt='{grpo_info['prompt_col']}'")
        print(f"  操作：应用映射代码（见下方）")
    else:
        print(f"  状态：缺少 prompt 字段")

    # KTO 检查结果
    print(f"\n[KTO] {'✓ 就绪' if kto_info['ready'] else '✗ 不兼容'}")
    if kto_info["ready"]:
        print(f"  原因：数据集包含 'prompt'、'completion'、'label' 字段")
        print(f"  操作：可直接与 KTOTrainer 一起使用")
    else:
        print(f"  状态：缺少必需字段（prompt + completion + label）")

    # 映射代码
    print(f"\n{'映射代码（如需要）':-<80}")

    mapping_needed = False

    # 生成 SFT 映射代码
    sft_mapping = generate_mapping_code("SFT", sft_info)
    if sft_mapping:
        print(f"\n# 用于 SFT 训练：")
        print(sft_mapping)
        mapping_needed = True

    # 生成 DPO 映射代码
    dpo_mapping = generate_mapping_code("DPO", dpo_info)
    if dpo_mapping:
        print(f"\n# 用于 DPO 训练：")
        print(dpo_mapping)
        mapping_needed = True

    # 生成 GRPO 映射代码
    grpo_mapping = generate_mapping_code("GRPO", grpo_info)
    if grpo_mapping:
        print(f"\n# 用于 GRPO 训练：")
        print(grpo_mapping)
        mapping_needed = True

    # 如果不需要映射
    if not mapping_needed:
        print("\n无需映射 - 数据集已准备好用于训练！")

    print(f"\n{'摘要':-<80}")
    print(f"推荐的训练方法：{', '.join(recommended) if recommended else '无（数据集需要格式化）'}")
    print(f"\n注意：使用了数据集服务器 API（即时获取，无需下载）")

    print("\n" + "=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # 用户中断程序（Ctrl+C）
        sys.exit(0)
    except Exception as e:
        # 捕获其他异常并输出错误信息
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)
