# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface-hub>=0.26.0",
#     "python-dotenv>=1.2.1",
# ]
# ///

"""
使用 `hf jobs uv run` CLI 提交评估任务。

此包装器构造适当的命令，以在具有请求的硬件的 Hugging Face Jobs 上执行本地
`inspect_eval_uv.py` 脚本。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import get_token
from dotenv import load_dotenv

load_dotenv()


SCRIPT_PATH = Path(__file__).with_name("inspect_eval_uv.py").resolve()


def create_eval_job(
    model_id: str,
    task: str,
    hardware: str = "cpu-basic",
    hf_token: Optional[str] = None,
    limit: Optional[int] = None,
) -> None:
    """
    使用 Hugging Face Jobs CLI 提交评估任务。
    """
    token = hf_token or os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise ValueError("需要 HF_TOKEN。请在环境中设置或作为参数传递。")

    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"脚本未找到：{SCRIPT_PATH}")

    print(f"为 {model_id} 准备评估任务，任务为 {task}（硬件：{hardware}）")

    cmd = [
        "hf",
        "jobs",
        "uv",
        "run",
        str(SCRIPT_PATH),
        "--flavor",
        hardware,
        "--secrets",
        f"HF_TOKEN={token}",
        "--",
        "--model",
        model_id,
        "--task",
        task,
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    print("执行：", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print("hf jobs 命令失败", file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="在 Hugging Face Jobs 上运行 inspect-ai 评估")
    parser.add_argument("--model", required=True, help="模型 ID（例如 Qwen/Qwen3-0.6B）")
    parser.add_argument("--task", required=True, help="Inspect 任务（例如 mmlu、gsm8k）")
    parser.add_argument("--hardware", default="cpu-basic", help="硬件类型（例如 t4-small、a10g-small）")
    parser.add_argument("--limit", type=int, default=None, help="限制要评估的样本数量")

    args = parser.parse_args()

    create_eval_job(
        model_id=args.model,
        task=args.task,
        hardware=args.hardware,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
