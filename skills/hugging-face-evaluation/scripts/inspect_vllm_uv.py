# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "inspect-ai>=0.3.0",
#     "inspect-evals",
#     "vllm>=0.4.0",
#     "torch>=2.0.0",
#     "transformers>=4.40.0",
# ]
# ///

"""
通过 vLLM 或 HuggingFace Transformers 后端运行 inspect-ai 评估的入口脚本。

此脚本使用本地 GPU 推理对自定义 HuggingFace 模型运行评估，
与推理提供商脚本（使用外部 API）分开。

用法（独立运行）：
    python inspect_vllm_uv.py --model "meta-llama/Llama-3.2-1B" --task "mmlu"

用法（通过 HF Jobs）：
    hf jobs uv run inspect_vllm_uv.py \
        --flavor a10g-small \
        --secret HF_TOKEN=$HF_TOKEN \
        -- --model "meta-llama/Llama-3.2-1B" --task "mmlu"

模型后端：
    - vllm: 使用 vLLM 进行快速推理（推荐用于大型模型）
    - hf: HuggingFace Transformers 后端（更广泛的模型兼容性）
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


def run_inspect_vllm(
    model_id: str,
    task: str,
    limit: Optional[int] = None,
    max_connections: int = 4,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    log_level: str = "info",
) -> None:
    """
    使用 vLLM 后端运行 inspect-ai 评估。

    参数：
        model_id: HuggingFace 模型 ID
        task: 要执行的 inspect-ai 任务（例如，"mmlu"、"gsm8k"）
        limit: 限制要评估的样本数量
        max_connections: 最大并发连接数
        temperature: 采样温度
        tensor_parallel_size: 用于张量并行的 GPU 数量
        gpu_memory_utilization: GPU 内存分数
        dtype: 数据类型（auto、float16、bfloat16）
        trust_remote_code: 允许远程代码执行
        log_level: 日志级别
    """
    setup_environment()

    model_spec = f"vllm/{model_id}"
    cmd = [
        "inspect",
        "eval",
        task,
        "--model",
        model_spec,
        "--log-level",
        log_level,
        "--max-connections",
        str(max_connections),
    ]

    # 与 HF 推理提供商不同，vLLM 支持 temperature=0
    cmd.extend(["--temperature", str(temperature)])

    # 旧版本的 inspect-ai CLI 不支持 --model-args；依赖默认值
    # 让 vLLM 为小型模型选择合理的设置。
    if tensor_parallel_size != 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if gpu_memory_utilization != 0.8:
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    if dtype != "auto":
        cmd.extend(["--dtype", dtype])
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"运行中: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print("评估完成。")
    except subprocess.CalledProcessError as exc:
        print(f"评估失败，退出代码为 {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)


def run_inspect_hf(
    model_id: str,
    task: str,
    limit: Optional[int] = None,
    max_connections: int = 1,
    temperature: float = 0.001,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    log_level: str = "info",
) -> None:
    """
    使用 HuggingFace Transformers 后端运行 inspect-ai 评估。

    当 vLLM 不支持模型架构时使用此函数。

    参数：
        model_id: HuggingFace 模型 ID
        task: 要执行的 inspect-ai 任务
        limit: 限制样本数量
        max_connections: 最大并发连接数（为内存考虑，请保持较低）
        temperature: 采样温度
        device: 要使用的设备（auto、cuda、cpu）
        dtype: 数据类型
        trust_remote_code: 允许远程代码执行
        log_level: 日志级别
    """
    setup_environment()

    model_spec = f"hf/{model_id}"

    cmd = [
        "inspect",
        "eval",
        task,
        "--model",
        model_spec,
        "--log-level",
        log_level,
        "--max-connections",
        str(max_connections),
        "--temperature",
        str(temperature),
    ]

    if device != "auto":
        cmd.extend(["--device", device])
    if dtype != "auto":
        cmd.extend(["--dtype", dtype])
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"运行中: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print("评估完成。")
    except subprocess.CalledProcessError as exc:
        print(f"评估失败，退出代码为 {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 vLLM 或 HuggingFace Transformers 在自定义模型上运行 inspect-ai 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用 vLLM 后端运行 MMLU
  python inspect_vllm_uv.py --model meta-llama/Llama-3.2-1B --task mmlu

  # 使用 HuggingFace Transformers 后端运行
  python inspect_vllm_uv.py --model meta-llama/Llama-3.2-1B --task mmlu --backend hf

  # 运行有限样本进行测试
  python inspect_vllm_uv.py --model meta-llama/Llama-3.2-1B --task mmlu --limit 10

  # 使用张量并行在多个 GPU 上运行
  python inspect_vllm_uv.py --model meta-llama/Llama-3.2-70B --task mmlu --tensor-parallel-size 4

可用任务（来自 inspect-evals）：
  - mmlu: 大规模多任务语言理解
  - gsm8k: 小学数学
  - hellaswag: 常识推理
  - arc_challenge: AI2 推理挑战
  - truthfulqa: TruthfulQA 基准测试
  - winogrande: Winograd 模式挑战
  - humaneval: 代码生成（HumanEval）

通过 HF Jobs：
  hf jobs uv run inspect_vllm_uv.py \
      --flavor a10g-small \
      --secret HF_TOKEN=$HF_TOKEN \
      -- --model meta-llama/Llama-3.2-1B --task mmlu
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
        help="要执行的 inspect-ai 任务（例如，mmlu、gsm8k）",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "hf"],
        default="vllm",
        help="模型后端（默认：vllm）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制要评估的样本数量",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="最大并发连接数（默认：vllm 为 4，hf 为 1）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="采样温度（默认：vllm 为 0.0，hf 为 0.001）",
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
        "--device",
        default="auto",
        help="HF 后端的设备（auto、cuda、cpu）",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="允许执行来自模型仓库的远程代码",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="日志级别（默认：info）",
    )

    args = parser.parse_args()

    if args.backend == "vllm":
        run_inspect_vllm(
            model_id=args.model,
            task=args.task,
            limit=args.limit,
            max_connections=args.max_connections or 4,
            temperature=args.temperature if args.temperature is not None else 0.0,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            log_level=args.log_level,
        )
    else:
        run_inspect_hf(
            model_id=args.model,
            task=args.task,
            limit=args.limit,
            max_connections=args.max_connections or 1,
            temperature=args.temperature if args.temperature is not None else 0.001,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    main()
