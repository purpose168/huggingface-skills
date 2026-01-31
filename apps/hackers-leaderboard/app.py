#!/usr/bin/env python3
"""
黑客排行榜 - 用于显示 hf-skills 组织参与度的 Gradio 应用。

从 hf-skills/hackers-leaderboard 数据集读取排行榜数据。
需要单独运行 collect_points.py 来更新数据集。

使用方法:
    python app.py
"""

from __future__ import annotations

import json

import gradio as gr
import requests

TABLE_HEADERS = [
    "Rank",
    "Username",
    "Points",
    "💬 Discussions",
]

TABLE_DATATYPES = [
    "number",
    "markdown",
    "number",
]


DATASET_REPO = "hf-skills/hackers-leaderboard"
LEADERBOARD_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/data/leaderboard.jsonl"
METADATA_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/data/metadata.json"


def format_username(username: str) -> str:
    """将用户名格式化为可点击链接。"""
    return f"[{username}](https://huggingface.co/{username})"


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
        for i, entry in enumerate(leaderboard, 1):
            rows.append(
                [
                    i,
                    format_username(entry["username"]),
                    entry["prs_opened"],
                ]
            )

        status = "\n".join(
            [
                f"**数据来源:** [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO})",
                f"**最后更新:** {metadata.get('generated_at', '未知')}",
                f"**参与者:** {metadata.get('total_participants', len(leaderboard))}",
                f"**总积分:** {metadata.get('total_points', sum(e['total_points'] for e in leaderboard))}",
            ]
        )

        return status, rows

    except Exception as e:
        return f"❌ 加载排行榜失败: {e}", []


with gr.Blocks() as demo:
    gr.HTML(
        """
        <div class="subtitle">
            <img src="https://github.com/huggingface/skills/raw/main/assets/banner.png" alt="人类最后的黑客马拉松 (2025)" width="100%">
        </div>
        <div class="leaderboard-title"><h1>🏆 人类最后的黑客马拉松排行榜</h1></div>
        """
    )

    leaderboard_table = gr.Dataframe(
        headers=TABLE_HEADERS,
        datatype=TABLE_DATATYPES,
        interactive=False,
        wrap=True,
    )

    status_box = gr.Markdown("点击刷新以加载排行榜...")
    
    demo.load(
        refresh_handler,
        outputs=[status_box, leaderboard_table],
    )

    gr.Markdown(
        """
        ---
        
        **链接:**
        - [加入 hf-skills](https://huggingface.co/organizations/hf-skills/share/KrqrmBxkETjvevFbfkXeezcyMbgMjjMaOp)
        - [任务说明](https://github.com/huggingface/skills/tree/main/apps/quests)
        - [GitHub 仓库](https://github.com/huggingface/skills)
        """
    )

if __name__ == "__main__":
    demo.launch()
