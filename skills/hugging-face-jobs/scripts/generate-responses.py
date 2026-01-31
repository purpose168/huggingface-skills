# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "flashinfer-python",
#     "huggingface-hub[hf_transfer]",
#     "hf-xet>= 1.1.7",
#     "torch",
#     "transformers",
#     "vllm>=0.8.5",
# ]
#
# ///
"""
使用 vLLM 为数据集中的提示词生成响应,以实现高效的 GPU 推理。

此脚本从 Hugging Face Hub 加载包含聊天格式消息的数据集,
应用模型的聊天模板,使用 vLLM 生成响应,并将结果
保存回 Hub,附带完整的数据集说明卡片。

使用示例:
    # 本地执行,自动检测 GPU
    uv run generate-responses.py \\
        username/input-dataset \\
        username/output-dataset \\
        --messages-column messages

    # 使用自定义模型和采样参数
    uv run generate-responses.py \\
        username/input-dataset \\
        username/output-dataset \\
        --model-id meta-llama/Llama-3.1-8B-Instruct \\
        --temperature 0.9 \\
        --top-p 0.95 \\
        --max-tokens 2048

    # HF Jobs 执行(完整命令请参见脚本输出)
    hf jobs uv run --flavor a100x4 ...
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

from datasets import load_dataset
from huggingface_hub import DatasetCard, get_token, login
from torch import cuda
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 启用 HF Transfer 以加快下载速度
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_gpu_availability() -> int:
    """检查 CUDA 是否可用并返回 GPU 数量。"""
    if not cuda.is_available():
        logger.error("CUDA 不可用。此脚本需要 GPU。")
        logger.error(
            "请在配备 NVIDIA GPU 的机器上运行,或使用带 GPU 规格的 HF Jobs。"
        )
        sys.exit(1)

    num_gpus = cuda.device_count()
    for i in range(num_gpus):
        gpu_name = cuda.get_device_name(i)
        gpu_memory = cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"GPU {i}: {gpu_name},内存 {gpu_memory:.1f} GB")

    return num_gpus


def create_dataset_card(
    source_dataset: str,
    model_id: str,
    messages_column: str,
    prompt_column: Optional[str],
    sampling_params: SamplingParams,
    tensor_parallel_size: int,
    num_examples: int,
    generation_time: str,
    num_skipped: int = 0,
    max_model_len_used: Optional[int] = None,
) -> str:
    """创建完整的数据集说明卡片,记录生成过程。"""
    filtering_section = ""
    if num_skipped > 0:
        skip_percentage = (num_skipped / num_examples) * 100
        processed = num_examples - num_skipped
        filtering_section = f"""

### 过滤统计

- **总样本数**: {num_examples:,}
- **已处理**: {processed:,} ({100 - skip_percentage:.1f}%)
- **已跳过(过长)**: {num_skipped:,} ({skip_percentage:.1f}%)
- **使用的最大模型长度**: {max_model_len_used:,} tokens

注意:超过最大模型长度的提示词已被跳过,响应为空。"""

    return f"""---
tags:
- generated
- vllm
- uv-script
---

# 生成的响应数据集

此数据集包含来自 [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) 的提示词的生成响应。

## 生成详情

- **源数据集**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **输入列**: `{prompt_column if prompt_column else messages_column}` ({"纯文本提示词" if prompt_column else "聊天消息"})
- **模型**: [{model_id}](https://huggingface.co/{model_id})
- **样本数量**: {num_examples:,}
- **生成日期**: {generation_time}{filtering_section}

### 采样参数

- **温度**: {sampling_params.temperature}
- **Top P**: {sampling_params.top_p}
- **Top K**: {sampling_params.top_k}
- **Min P**: {sampling_params.min_p}
- **最大 Tokens**: {sampling_params.max_tokens}
- **重复惩罚**: {sampling_params.repetition_penalty}

### 硬件配置

- **张量并行大小**: {tensor_parallel_size}
- **GPU 配置**: {tensor_parallel_size} 个 GPU

## 数据集结构

数据集包含源数据集的所有列,外加:
- `response`: 模型生成的响应

## 生成脚本

使用 [uv-scripts/vllm](https://huggingface.co/datasets/uv-scripts/vllm) 中的 vLLM 推理脚本生成。

要重现此生成过程:

```bash
uv run https://huggingface.co/datasets/uv-scripts/vllm/raw/main/generate-responses.py \\
    {source_dataset} \\
    <output-dataset> \\
    --model-id {model_id} \\
    {"--prompt-column " + prompt_column if prompt_column else "--messages-column " + messages_column} \\
    --temperature {sampling_params.temperature} \\
    --top-p {sampling_params.top_p} \\
    --top-k {sampling_params.top_k} \\
    --max-tokens {sampling_params.max_tokens}{f" \\\\\\n    --max-model-len {max_model_len_used}" if max_model_len_used else ""}
```
"""


def main(
    src_dataset_hub_id: str,
    output_dataset_hub_id: str,
    model_id: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    messages_column: str = "messages",
    prompt_column: Optional[str] = None,
    output_column: str = "response",
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    max_tokens: int = 16384,
    repetition_penalty: float = 1.0,
    gpu_memory_utilization: float = 0.90,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: Optional[int] = None,
    skip_long_prompts: bool = True,
    max_samples: Optional[int] = None,
    hf_token: Optional[str] = None,
):
    """
    主生成流程。

    参数:
        src_dataset_hub_id: Hugging Face Hub 上的输入数据集
        output_dataset_hub_id: 在 Hugging Face Hub 上保存结果的位置
        model_id: 用于生成的 Hugging Face 模型 ID
        messages_column: 包含聊天消息的列名
        prompt_column: 包含纯文本提示词的列名(messages_column 的替代选项)
        output_column: 生成响应的列名
        temperature: 采样温度
        top_p: Top-p 采样参数
        top_k: Top-k 采样参数
        min_p: 最小概率阈值
        max_tokens: 要生成的最大 token 数
        repetition_penalty: 重复惩罚参数
        gpu_memory_utilization: GPU 内存利用率因子
        max_model_len: 最大模型上下文长度(None 使用模型默认值)
        tensor_parallel_size: 使用的 GPU 数量(None 为自动检测)
        skip_long_prompts: 跳过超过 max_model_len 的提示词而不是失败
        max_samples: 要处理的最大样本数(None 表示全部)
        hf_token: Hugging Face 认证令牌
    """
    generation_start_time = datetime.now().isoformat()

    # GPU 检查和配置
    num_gpus = check_gpu_availability()
    if tensor_parallel_size is None:
        tensor_parallel_size = num_gpus
        logger.info(
            f"自动检测到 {num_gpus} 个 GPU,使用 tensor_parallel_size={tensor_parallel_size}"
        )
    else:
        logger.info(f"使用指定的 tensor_parallel_size={tensor_parallel_size}")
        if tensor_parallel_size > num_gpus:
            logger.warning(
                f"请求 {tensor_parallel_size} 个 GPU,但只有 {num_gpus} 个可用"
            )

    # 认证 - 尝试多种方法
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN") or get_token()

    if not HF_TOKEN:
        logger.error("未找到 HuggingFace 令牌。请通过以下方式提供令牌:")
        logger.error("  1. --hf-token 参数")
        logger.error("  2. HF_TOKEN 环境变量")
        logger.error("  3. 运行 'huggingface-cli login' 或在 Python 中使用 login()")
        sys.exit(1)

    logger.info("找到 HuggingFace 令牌,正在认证...")
    login(token=HF_TOKEN)

    # 初始化 vLLM
    logger.info(f"正在加载模型: {model_id}")
    vllm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if max_model_len is not None:
        vllm_kwargs["max_model_len"] = max_model_len
        logger.info(f"使用 max_model_len={max_model_len}")

    llm = LLM(**vllm_kwargs)

    # 加载分词器以应用聊天模板
    logger.info("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 创建采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )

    # 加载数据集
    logger.info(f"正在加载数据集: {src_dataset_hub_id}")
    dataset = load_dataset(src_dataset_hub_id, split="train")

    # 如果指定了 max_samples,则应用
    if max_samples is not None and max_samples < len(dataset):
        logger.info(f"将数据集限制为 {max_samples} 个样本")
        dataset = dataset.select(range(max_samples))

    total_examples = len(dataset)
    logger.info(f"数据集已加载,包含 {total_examples:,} 个样本")

    # 确定使用哪一列并进行验证
    if prompt_column:
        # 使用提示词列模式
        if prompt_column not in dataset.column_names:
            logger.error(
                f"未找到列 '{prompt_column}'。可用列: {dataset.column_names}"
            )
            sys.exit(1)
        logger.info(f"使用提示词列模式,列名: '{prompt_column}'")
        use_messages = False
    else:
        # 使用消息列模式
        if messages_column not in dataset.column_names:
            logger.error(
                f"未找到列 '{messages_column}'。可用列: {dataset.column_names}"
            )
            sys.exit(1)
        logger.info(f"使用消息列模式,列名: '{messages_column}'")
        use_messages = True

    # 获取用于过滤的有效最大长度
    if max_model_len is not None:
        effective_max_len = max_model_len
    else:
        # 获取模型的默认最大长度
        effective_max_len = llm.llm_engine.model_config.max_model_len
    logger.info(f"使用的有效最大模型长度: {effective_max_len}")

    # 处理消息并应用聊天模板
    logger.info("正在准备提示词...")
    all_prompts = []
    valid_prompts = []
    valid_indices = []
    skipped_info = []

    for i, example in enumerate(tqdm(dataset, desc="正在处理提示词")):
        if use_messages:
            # 消息模式: 使用现有的聊天消息
            messages = example[messages_column]
            # 应用聊天模板
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # 提示词模式: 将纯文本转换为消息格式
            user_prompt = example[prompt_column]
            messages = [{"role": "user", "content": user_prompt}]
            # 应用聊天模板
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        all_prompts.append(prompt)

        # 如果启用了过滤,则计算 token 数
        if skip_long_prompts:
            tokens = tokenizer.encode(prompt)
            if len(tokens) <= effective_max_len:
                valid_prompts.append(prompt)
                valid_indices.append(i)
            else:
                skipped_info.append((i, len(tokens)))
        else:
            valid_prompts.append(prompt)
            valid_indices.append(i)

    # 记录过滤结果
    if skip_long_prompts and skipped_info:
        logger.warning(
            f"跳过了 {len(skipped_info)} 个超过 max_model_len ({effective_max_len} tokens) 的提示词"
        )
        logger.info("跳过的提示词详情(前 10 个):")
        for idx, (prompt_idx, token_count) in enumerate(skipped_info[:10]):
            logger.info(
                f"  - 样本 {prompt_idx}: {token_count} tokens (超出 {token_count - effective_max_len})"
            )
        if len(skipped_info) > 10:
            logger.info(f"  ... 还有 {len(skipped_info) - 10} 个")

        skip_percentage = (len(skipped_info) / total_examples) * 100
        if skip_percentage > 10:
            logger.warning(f"警告: {skip_percentage:.1f}% 的提示词被跳过!")

    if not valid_prompts:
        logger.error("过滤后没有有效的提示词需要处理!")
        sys.exit(1)

    # 生成响应 - vLLM 内部处理批处理
    logger.info(f"开始为 {len(valid_prompts):,} 个有效提示词生成响应...")
    logger.info("vLLM 将自动处理批处理和调度")

    outputs = llm.generate(valid_prompts, sampling_params)

    # 提取生成的文本并创建完整的响应列表
    logger.info("正在提取生成的响应...")
    responses = [""] * total_examples  # 用空字符串初始化

    for idx, output in enumerate(outputs):
        original_idx = valid_indices[idx]
        response = output.outputs[0].text.strip()
        responses[original_idx] = response

    # 将响应添加到数据集
    logger.info("正在将响应添加到数据集...")
    dataset = dataset.add_column(output_column, responses)

    # 创建数据集说明卡片
    logger.info("正在创建数据集说明卡片...")
    card_content = create_dataset_card(
        source_dataset=src_dataset_hub_id,
        model_id=model_id,
        messages_column=messages_column,
        prompt_column=prompt_column,
        sampling_params=sampling_params,
        tensor_parallel_size=tensor_parallel_size,
        num_examples=total_examples,
        generation_time=generation_start_time,
        num_skipped=len(skipped_info) if skip_long_prompts else 0,
        max_model_len_used=effective_max_len if skip_long_prompts else None,
    )

    # 将数据集推送到 Hub
    logger.info(f"正在将数据集推送到: {output_dataset_hub_id}")
    dataset.push_to_hub(output_dataset_hub_id, token=HF_TOKEN)

    # 推送数据集说明卡片
    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset_hub_id, token=HF_TOKEN)

    logger.info("✅ 生成完成!")
    logger.info(
        f"数据集地址: https://huggingface.co/datasets/{output_dataset_hub_id}"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="使用 vLLM 为数据集提示词生成响应",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  # 使用默认 Qwen 模型的基本用法
  uv run generate-responses.py input-dataset output-dataset

  # 使用自定义模型和参数
  uv run generate-responses.py input-dataset output-dataset \\
    --model-id meta-llama/Llama-3.1-8B-Instruct \\
    --temperature 0.9 \\
    --max-tokens 2048

  # 强制指定 GPU 配置
  uv run generate-responses.py input-dataset output-dataset \\
    --tensor-parallel-size 2 \\
    --gpu-memory-utilization 0.95

  # 使用环境变量提供令牌
  HF_TOKEN=hf_xxx uv run generate-responses.py input-dataset output-dataset
            """,
        )

        parser.add_argument(
            "src_dataset_hub_id",
            help="Hugging Face Hub 上的输入数据集(例如: username/dataset-name)",
        )
        parser.add_argument(
            "output_dataset_hub_id", help="Hugging Face Hub 上的输出数据集名称"
        )
        parser.add_argument(
            "--model-id",
            type=str,
            default="Qwen/Qwen3-30B-A3B-Instruct-2507",
            help="用于生成的模型(默认: Qwen3-30B-A3B-Instruct-2507)",
        )
        parser.add_argument(
            "--messages-column",
            type=str,
            default="messages",
            help="包含聊天消息的列(默认: messages)",
        )
        parser.add_argument(
            "--prompt-column",
            type=str,
            help="包含纯文本提示词的列(--messages-column 的替代选项)",
        )
        parser.add_argument(
            "--output-column",
            type=str,
            default="response",
            help="生成响应的列名(默认: response)",
        )
        parser.add_argument(
            "--max-samples",
            type=int,
            help="要处理的最大样本数(默认: 全部)",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.7,
            help="采样温度(默认: 0.7)",
        )
        parser.add_argument(
            "--top-p",
            type=float,
            default=0.8,
            help="Top-p 采样参数(默认: 0.8)",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=20,
            help="Top-k 采样参数(默认: 20)",
        )
        parser.add_argument(
            "--min-p",
            type=float,
            default=0.0,
            help="最小概率阈值(默认: 0.0)",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=16384,
            help="要生成的最大 token 数(默认: 16384)",
        )
        parser.add_argument(
            "--repetition-penalty",
            type=float,
            default=1.0,
            help="重复惩罚(默认: 1.0)",
        )
        parser.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=0.90,
            help="GPU 内存利用率因子(默认: 0.90)",
        )
        parser.add_argument(
            "--max-model-len",
            type=int,
            help="最大模型上下文长度(默认: 模型默认值)",
        )
        parser.add_argument(
            "--tensor-parallel-size",
            type=int,
            help="使用的 GPU 数量(默认: 自动检测)",
        )
        parser.add_argument(
            "--hf-token",
            type=str,
            help="Hugging Face 令牌(也可以使用 HF_TOKEN 环境变量)",
        )
        parser.add_argument(
            "--skip-long-prompts",
            action="store_true",
            default=True,
            help="跳过超过 max_model_len 的提示词而不是失败(默认: True)",
        )
        parser.add_argument(
            "--no-skip-long-prompts",
            dest="skip_long_prompts",
            action="store_false",
            help="对超过 max_model_len 的提示词失败",
        )

        args = parser.parse_args()

        main(
            src_dataset_hub_id=args.src_dataset_hub_id,
            output_dataset_hub_id=args.output_dataset_hub_id,
            model_id=args.model_id,
            messages_column=args.messages_column,
            prompt_column=args.prompt_column,
            output_column=args.output_column,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            skip_long_prompts=args.skip_long_prompts,
            max_samples=args.max_samples,
            hf_token=args.hf_token,
        )
    else:
        # 在不带参数运行时显示 HF Jobs 示例
        print("""
vLLM 响应生成脚本
==============================

此脚本需要参数。有关使用信息:
    uv run generate-responses.py --help

多 GPU 的 HF Jobs 命令示例:
    # 如果您已使用 huggingface-cli 登录,令牌将自动检测
    hf jobs uv run \\
        --flavor l4x4 \\
        https://huggingface.co/datasets/uv-scripts/vllm/raw/main/generate-responses.py \\
        username/input-dataset \\
        username/output-dataset \\
        --messages-column messages \\
        --model-id Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --temperature 0.7 \\
        --max-tokens 16384
        """)
