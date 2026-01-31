#!/usr/bin/env python3
"""
评估排行榜 - 用于显示模型评估分数的 Gradio 应用。

从 hf-skills/evals-leaderboard 数据集读取排行榜数据。
需要单独运行 collect_evals.py 来更新数据集。

使用方法:
    python app.py
"""

from __future__ import annotations

import json

import gradio as gr
import requests

TABLE_HEADERS = [
    "Model",
    "Benchmark",
    "Score",
    "Source",
]

TABLE_DATATYPES = [
    "markdown",
    "text",
    "number",
    "markdown",
]


DATASET_REPO = "hf-skills/evals-leaderboard"
LEADERBOARD_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/data/leaderboard.jsonl"
METADATA_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/data/metadata.json"


def format_model_link(model_id: str) -> str:
    """将模型 ID 格式化为可点击链接。"""
    return f"[{model_id}](https://huggingface.co/{model_id})"


def format_source_link(source_type: str, contributor: str, source_url: str) -> str:
    """将来源格式化为可点击链接。"""
    return f"{source_type} by [{contributor}]({source_url})"


def fetch_leaderboard() -> tuple[list[dict], dict]:
    """从 HF 数据集获取排行榜数据。"""
    # 获取排行榜 JSONL
    resp = requests.get(LEADERBOARD_URL, timeout=30)
    resp.raise_for_status()
    leaderboard = [json.loads(line) for line in resp.text.strip().split("\n") if line]

    # 获取元数据
    resp = requests.get(METADATA_URL, timeout=30)
    resp.raise_for_status()
    metadata = resp.json()

    return leaderboard, metadata


def refresh_handler() -> tuple[str, list[list]]:
    """从数据集刷新排行榜数据。"""
    try:
        leaderboard, metadata = fetch_leaderboard()

        # 构建表格行
        rows = []
        for entry in leaderboard:
            rows.append(
                [
                    format_model_link(entry["model_id"]),
                    entry["benchmark"],
                    entry["score"],
                    format_source_link(
                        entry["source_type"],
                        entry["contributor"],
                        entry["source_url"],
                    ),
                ]
            )

        status = "\n".join(
            [
                f"**数据来源:** [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO})",
                f"**最后更新:** {metadata.get('generated_at', '未知')}",
                f"**有分数的模型:** {metadata.get('models_with_scores', '未知')}",
                f"**总条目数:** {metadata.get('total_entries', len(leaderboard))}",
            ]
        )

        return status, rows

    except Exception as e:
        return f"❌ 加载排行榜失败: {e}", []


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 📊 HF 评估排行榜
        
        显示从 model-index 元数据或其拉取请求中获取的 MMLU、BigCodeBench 和 ARC MC 分数，
        适用于热门的文本生成模型。
        """
    )

    status_box = gr.Markdown("加载排行榜中...")

    leaderboard_table = gr.Dataframe(
        headers=TABLE_HEADERS,
        datatype=TABLE_DATATYPES,
        interactive=False,
        wrap=True,
    )

    demo.load(
        refresh_handler,
        outputs=[status_box, leaderboard_table],
    )

    gr.Markdown(
        f"""
        ---
        
        **链接:**
        - [数据集: {DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO})
        - [GitHub 仓库](https://github.com/huggingface/skills)
        """
    )


if __name__ == "__main__":
    demo.launch()
