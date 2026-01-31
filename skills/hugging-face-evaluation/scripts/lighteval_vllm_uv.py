# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "lighteval[accelerate,vllm]>=0.6.0",
#     "torch>=2.0.0",
#     "transformers>=4.40.0",
#     "accelerate>=0.30.0",
#     "vllm>=0.4.0",
# ]
# ///

"""
通过 `hf jobs uv run` 使用 vLLM 后端运行 lighteval 评估的入口脚本。

此脚本使用 vLLM 在自定义 HuggingFace 模型上进行高效的 GPU 推理来运行评估。
它与推理提供商脚本分开，直接在硬件上评估模型。

用法（独立运行）：
    python lighteval_vllm_uv.py --model "meta-llama/Llama-3.2-1B" --tasks "leaderboard|mmlu|5"

用法（通过 HF Jobs）：
    hf jobs uv run lighteval_vllm_uv.py \
        --flavor a10g-small \
        --secret HF_TOKEN=$HF_TOKEN \
        -- --model "meta-llama/Llama-3.2-1B" --tasks "leaderboard|mmlu|5"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional


def setup_environment() -> None:
    """配置 HuggingFace 身份验证的环境变量。"""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HF_HUB_TOKEN", hf_token)


def run_lighteval_vllm(
    model_id: str,
    tasks: str,
    output_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    use_chat_template: bool = False,
    system_prompt: Optional[str] = None,
) -> None:
    """
    使用 vLLM 后端运行 lighteval 以进行高效的 GPU 推理。

    参数：
        model_id: HuggingFace 模型 ID（例如，"meta-llama/Llama-3.2-1B"）
        tasks: 任务规范（例如，"leaderboard|mmlu|5" 或 "lighteval|hellaswag|0"）
        output_dir: 评估结果的目录
        max_samples: 限制每个任务的样本数量
        batch_size: 评估的批处理大小
        tensor_parallel_size: 用于张量并行的 GPU 数量
        gpu_memory_utilization: 要使用的 GPU 内存分数（0.0-1.0）
        dtype: 模型权重的数据类型（auto、float16、bfloat16）
        trust_remote_code: 允许执行来自模型仓库的远程代码
        use_chat_template: 为对话模型应用聊天模板
        system_prompt: 聊天模型的系统提示
    """
    setup_environment()

    # 构建 lighteval vllm 命令
    cmd = [
        "lighteval",
        "vllm",
        model_id,
        tasks,
        "--batch-size", str(batch_size),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
    ]

    if output_dir:
        cmd.extend(["--output-dir", output_dir])

    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if use_chat_template:
        cmd.append("--use-chat-template")

    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    print(f"运行中: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print("评估完成。")
    except subprocess.CalledProcessError as exc:
        print(f"评估失败，退出代码为 {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)


def run_lighteval_accelerate(
    model_id: str,
    tasks: str,
    output_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    use_chat_template: bool = False,
    system_prompt: Optional[str] = None,
) -> None:
    """
    使用 accelerate 后端运行 lighteval 以进行多 GPU 分布式推理。

    当 vLLM 不可用或对于 vLLM 不支持的模型时使用此后端。

    参数：
        model_id: HuggingFace 模型 ID
        tasks: 任务规范
        output_dir: 评估结果的目录
        max_samples: 限制每个任务的样本数量
        batch_size: 评估的批处理大小
        dtype: 模型权重的数据类型
        trust_remote_code: 允许执行远程代码
        use_chat_template: 应用聊天模板
        system_prompt: 聊天模型的系统提示
    """
    setup_environment()

    # 构建 lighteval accelerate 命令
    cmd = [
        "lighteval",
        "accelerate",
        model_id,
        tasks,
        "--batch-size", str(batch_size),
        "--dtype", dtype,
    ]

    if output_dir:
        cmd.extend(["--output-dir", output_dir])

    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if use_chat_template:
        cmd.append("--use-chat-template")

    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    print(f"运行中: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print("评估完成。")
    except subprocess.CalledProcessError as exc:
        print(f"评估失败，退出代码为 {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在自定义 HuggingFace 模型上使用 vLLM 或 accelerate 后端运行 lighteval 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用 vLLM 运行 MMLU 评估
  python lighteval_vllm_uv.py --model meta-llama/Llama-3.2-1B --tasks "leaderboard|mmlu|5"

  # 使用 accelerate 后端而不是 vLLM 运行
  python lighteval_vllm_uv.py --model meta-llama/Llama-3.2-1B --tasks "leaderboard|mmlu|5" --backend accelerate

  # 为指令调优模型使用聊天模板运行
  python lighteval_vllm_uv.py --model meta-llama/Llama-3.2-1B-Instruct --tasks "leaderboard|mmlu|5" --use-chat-template

  # 运行有限样本进行测试
  python lighteval_vllm_uv.py --model meta-llama/Llama-3.2-1B --tasks "leaderboard|mmlu|5" --max-samples 10

任务格式：
  任务使用格式："suite|task|num_fewshot"
  - leaderboard|mmlu|5（5-shot 的 MMLU）
  - lighteval|hellaswag|0（零样本的 HellaSwag）
  - leaderboard|gsm8k|5（5-shot 的 GSM8K）
  - 多个任务："leaderboard|mmlu|5,leaderboard|gsm8k|5"
        """,
    )

    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace 模型 ID（例如，meta-llama/Llama-3.2-1B）",
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="任务规范（例如，'leaderboard|mmlu|5'）",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "accelerate"],
        default="vllm",
        help="要使用的推理后端（默认：vllm）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="评估结果的目录",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="限制每个任务的样本数量（对测试有用）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="评估的批处理大小（默认：1）",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="用于张量并行的 GPU 数量（仅 vLLM，默认：1）",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="要使用的 GPU 内存分数（仅 vLLM，默认：0.8）",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="模型权重的数据类型（默认：auto）",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="允许执行来自模型仓库的远程代码",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="为指令调优/聊天模型应用聊天模板",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="聊天模型的系统提示",
    )

    args = parser.parse_args()

    if args.backend == "vllm":
        run_lighteval_vllm(
            model_id=args.model,
            tasks=args.tasks,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            use_chat_template=args.use_chat_template,
            system_prompt=args.system_prompt,
        )
    else:
        run_lighteval_accelerate(
            model_id=args.model,
            tasks=args.tasks,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            dtype=args.dtype if args.dtype != "auto" else "bfloat16",
            trust_remote_code=args.trust_remote_code,
            use_chat_template=args.use_chat_template,
            system_prompt=args.system_prompt,
        )


if __name__ == "__main__":
    main()
