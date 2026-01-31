# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface-hub>=0.26.0",
#     "python-dotenv>=1.2.1",
# ]
# ///

"""
使用 `hf jobs uv run` CLI 提交基于 vLLM 的评估任务。

此包装器构造适当的命令，以在具有 GPU 硬件的 Hugging Face Jobs 上执行 vLLM 评估脚本
（lighteval 或 inspect-ai）。

与 run_eval_job.py（使用推理提供商/API）不同，此脚本使用 vLLM 或 HuggingFace Transformers
直接在任务的 GPU 上运行模型。

用法：
    python run_vllm_eval_job.py \
        --model meta-llama/Llama-3.2-1B \
        --task mmlu \
        --framework lighteval \
        --hardware a10g-small
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import get_token
from dotenv import load_dotenv

load_dotenv()

# 不同评估框架的脚本路径
SCRIPT_DIR = Path(__file__).parent.resolve()
LIGHTEVAL_SCRIPT = SCRIPT_DIR / "lighteval_vllm_uv.py"
INSPECT_SCRIPT = SCRIPT_DIR / "inspect_vllm_uv.py"

# 不同模型大小的硬件推荐
HARDWARE_RECOMMENDATIONS = {
    "small": "t4-small",       # < 3B 参数
    "medium": "a10g-small",    # 3B - 13B 参数
    "large": "a10g-large",     # 13B - 34B 参数
    "xlarge": "a100-large",    # 34B+ 参数
}


def estimate_hardware(model_id: str) -> str:
    """
    根据模型 ID 命名约定估计适当的硬件。
    
    返回硬件类型推荐。
    """
    model_lower = model_id.lower()
    
    # 检查模型名称中的显式大小指示器
    if any(x in model_lower for x in ["70b", "72b", "65b"]):
        return "a100-large"
    elif any(x in model_lower for x in ["34b", "33b", "32b", "30b"]):
        return "a10g-large"
    elif any(x in model_lower for x in ["13b", "14b", "7b", "8b"]):
        return "a10g-small"
    elif any(x in model_lower for x in ["3b", "2b", "1b", "0.5b", "small", "mini"]):
        return "t4-small"
    
    # 默认使用中等硬件
    return "a10g-small"


def create_lighteval_job(
    model_id: str,
    tasks: str,
    hardware: str,
    hf_token: Optional[str] = None,
    max_samples: Optional[int] = None,
    backend: str = "vllm",
    batch_size: int = 1,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = False,
    use_chat_template: bool = False,
) -> None:
    """
    在 HuggingFace Jobs 上提交 lighteval 评估任务。
    """
    token = hf_token or os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise ValueError("需要 HF_TOKEN。请在环境中设置或作为参数传递。")

    if not LIGHTEVAL_SCRIPT.exists():
        raise FileNotFoundError(f"脚本未找到：{LIGHTEVAL_SCRIPT}")

    print(f"为 {model_id} 准备 lighteval 任务")
    print(f"  任务：{tasks}")
    print(f"  后端：{backend}")
    print(f"  硬件：{hardware}")

    cmd = [
        "hf", "jobs", "uv", "run",
        str(LIGHTEVAL_SCRIPT),
        "--flavor", hardware,
        "--secrets", f"HF_TOKEN={token}",
        "--",
        "--model", model_id,
        "--tasks", tasks,
        "--backend", backend,
        "--batch-size", str(batch_size),
        "--tensor-parallel-size", str(tensor_parallel_size),
    ]

    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if use_chat_template:
        cmd.append("--use-chat-template")

    print(f"\n执行：{' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print("hf jobs 命令失败", file=sys.stderr)
        raise


def create_inspect_job(
    model_id: str,
    task: str,
    hardware: str,
    hf_token: Optional[str] = None,
    limit: Optional[int] = None,
    backend: str = "vllm",
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = False,
) -> None:
    """
    在 HuggingFace Jobs 上提交 inspect-ai 评估任务。
    """
    token = hf_token or os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise ValueError("需要 HF_TOKEN。请在环境中设置或作为参数传递。")

    if not INSPECT_SCRIPT.exists():
        raise FileNotFoundError(f"脚本未找到：{INSPECT_SCRIPT}")

    print(f"为 {model_id} 准备 inspect-ai 任务")
    print(f"  任务：{task}")
    print(f"  后端：{backend}")
    print(f"  硬件：{hardware}")

    cmd = [
        "hf", "jobs", "uv", "run",
        str(INSPECT_SCRIPT),
        "--flavor", hardware,
        "--secrets", f"HF_TOKEN={token}",
        "--",
        "--model", model_id,
        "--task", task,
        "--backend", backend,
        "--tensor-parallel-size", str(tensor_parallel_size),
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    print(f"\n执行：{' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print("hf jobs 命令失败", file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="向 HuggingFace Jobs 提交基于 vLLM 的评估任务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 在 A10G GPU 上使用 vLLM 运行 lighteval
  python run_vllm_eval_job.py \
      --model meta-llama/Llama-3.2-1B \
      --task "leaderboard|mmlu|5" \
      --framework lighteval \
      --hardware a10g-small

  # 在更大模型上使用多 GPU 运行 inspect-ai
  python run_vllm_eval_job.py \
      --model meta-llama/Llama-3.2-70B \
      --task mmlu \
      --framework inspect \
      --hardware a100-large \
      --tensor-parallel-size 4

  # 根据模型大小自动检测硬件
  python run_vllm_eval_job.py \
      --model meta-llama/Llama-3.2-1B \
      --task mmlu \
      --framework inspect

  # 使用 HF Transformers 后端运行（而不是 vLLM）
  python run_vllm_eval_job.py \
      --model microsoft/phi-2 \
      --task mmlu \
      --framework inspect \
      --backend hf

硬件类型：
  - t4-small: T4 GPU，适合 < 3B 的模型
  - a10g-small: A10G GPU，适合 3B-13B 的模型
  - a10g-large: A10G GPU，适合 13B-34B 的模型
  - a100-large: A100 GPU，适合 34B+ 的模型

框架：
  - lighteval: HuggingFace 的 lighteval 库
  - inspect: UK AI Safety 的 inspect-ai 库

任务格式：
  - lighteval: "suite|task|num_fewshot"（例如，"leaderboard|mmlu|5"）
  - inspect: 任务名称（例如，"mmlu"、"gsm8k"）
        """,
    )

    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace 模型 ID（例如，meta-llama/Llama-3.2-1B）",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="评估任务（格式取决于框架）",
    )
    parser.add_argument(
        "--framework",
        choices=["lighteval", "inspect"],
        default="lighteval",
        help="要使用的评估框架（默认：lighteval）",
    )
    parser.add_argument(
        "--hardware",
        default=None,
        help="硬件类型（如果未指定则自动检测）",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "hf", "accelerate"],
        default="vllm",
        help="模型后端（默认：vllm）",
    )
    parser.add_argument(
        "--limit",
        "--max-samples",
        type=int,
        default=None,
        dest="limit",
        help="限制要评估的样本数量",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="评估的批处理大小（仅 lighteval）",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="用于张量并行的 GPU 数量",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="允许执行来自模型仓库的远程代码",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="应用聊天模板（仅 lighteval）",
    )

    args = parser.parse_args()

    # 如果未指定则自动检测硬件
    hardware = args.hardware or estimate_hardware(args.model)
    print(f"使用硬件：{hardware}")

    # 在框架之间映射后端名称
    backend = args.backend
    if args.framework == "lighteval" and backend == "hf":
        backend = "accelerate"  # lighteval 使用 "accelerate" 作为 HF 后端

    if args.framework == "lighteval":
        create_lighteval_job(
            model_id=args.model,
            tasks=args.task,
            hardware=hardware,
            max_samples=args.limit,
            backend=backend,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
            use_chat_template=args.use_chat_template,
        )
    else:
        create_inspect_job(
            model_id=args.model,
            task=args.task,
            hardware=hardware,
            limit=args.limit,
            backend=backend if backend != "accelerate" else "hf",
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
        )


if __name__ == "__main__":
    main()
