# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "inspect-ai>=0.3.0",
#     "inspect-evals",
#     "openai",
# ]
# ///

"""
通过 `hf jobs uv run` 运行 inspect-ai 评估的入口脚本。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _inspect_evals_tasks_root() -> Optional[Path]:
    """如果可用，返回已安装的 inspect_evals 包路径。"""
    try:
        import inspect_evals

        return Path(inspect_evals.__file__).parent
    except Exception:
        return None


def _normalize_task(task: str) -> str:
    """通过保留任务名称，允许 lighteval 风格的 `suite|task|shots` 字符串。"""
    if "|" in task:
        parts = task.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return task


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect-ai 任务运行器")
    parser.add_argument("--model", required=True, help="Hugging Face Hub 上的模型 ID")
    parser.add_argument("--task", required=True, help="要执行的 inspect-ai 任务")
    parser.add_argument("--limit", type=int, default=None, help="限制要评估的样本数量")
    parser.add_argument(
        "--tasks-root",
        default=None,
        help="inspect 任务文件的可选路径。默认为已安装的 inspect_evals 包。",
    )
    parser.add_argument(
        "--sandbox",
        default="local",
        help="要使用的沙箱后端（默认：HF jobs 无 Docker 时使用 local）。",
    )
    args = parser.parse_args()

    # 确保下游库可以读取作为密钥传递的令牌
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HF_HUB_TOKEN", hf_token)

    task = _normalize_task(args.task)
    tasks_root = Path(args.tasks_root) if args.tasks_root else _inspect_evals_tasks_root()
    if tasks_root and not tasks_root.exists():
        tasks_root = None

    cmd = [
        "inspect",
        "eval",
        task,
        "--model",
        f"hf-inference-providers/{args.model}",
        "--log-level",
        "info",
        # 减少批处理大小以避免 OOM 错误（默认值为 32）
        "--max-connections",
        "1",
        # 设置小的正温度值（HF 不允许 temperature=0）
        "--temperature",
        "0.001",
    ]

    if args.sandbox:
        cmd.extend(["--sandbox", args.sandbox])

    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    try:
        subprocess.run(cmd, check=True, cwd=tasks_root)
        print("评估完成。")
    except subprocess.CalledProcessError as exc:
        location = f" (cwd={tasks_root})" if tasks_root else ""
        print(f"评估失败，退出代码为 {exc.returncode}{location}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

