# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "inspect-ai>=0.3.0",
# ]
# ///

"""
Entry point script for running inspect-ai evaluations via `hf jobs uv run`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect-ai job runner")
    parser.add_argument("--model", required=True, help="Model ID on Hugging Face Hub")
    parser.add_argument("--task", required=True, help="inspect-ai task to execute")
    args = parser.parse_args()

    # Ensure downstream libraries can read the token passed as a secret
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HF_HUB_TOKEN", hf_token)

    cmd = [
        "inspect",
        "eval",
        args.task,
        "--model",
        f"hf/{args.model}",
        "--log-level",
        "info",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("Evaluation complete.")
    except subprocess.CalledProcessError as exc:
        print(f"Evaluation failed with exit code {exc.returncode}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

