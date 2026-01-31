# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "huggingface-hub>=1.1.4",
#     "python-dotenv>=1.2.1",
#     "pyyaml>=6.0.3",
#     "requests>=2.32.5",
# ]
# ///

"""
将 Artificial Analysis 评估结果添加到 Hugging Face 模型卡片。

注意：这是一个独立的参考脚本。对于集成功能
（README 提取、验证等），请使用：
    ../scripts/evaluation_manager.py import-aa [选项]

独立使用：
AA_API_KEY="<您的-api-密钥>" HF_TOKEN="<您的-huggingface-令牌>" \
python artificial_analysis_to_hub.py \
--creator-slug <artificial-analysis-creator-slug> \
--model-name <artificial-analysis-model-name> \
--repo-id <huggingface-repo-id>

集成使用（推荐）：
python ../scripts/evaluation_manager.py import-aa \
--creator-slug <creator-slug> \
--model-name <model-name> \
--repo-id <repo-id> \
[--create-pr]
"""

import argparse
import os

import requests
import dotenv
from huggingface_hub import ModelCard

dotenv.load_dotenv()

API_KEY = os.getenv("AA_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
HEADERS = {"x-api-key": API_KEY}

if not API_KEY:
    raise ValueError("AA_API_KEY 未设置")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN 未设置")


def get_model_evaluations_data(creator_slug, model_name):
    response = requests.get(URL, headers=HEADERS)
    response_data = response.json()["data"]
    for model in response_data:
        if (
            model["model_creator"]["slug"] == creator_slug
            and model["slug"] == model_name
        ):
            return model
    raise ValueError(f"未找到模型 {model_name}")


def aa_evaluations_to_model_index(
    model,
    dataset_name="Artificial Analysis Benchmarks",
    dataset_type="artificial_analysis",
    task_type="evaluation",
):
    if not model:
        raise ValueError("需要模型数据")

    model_name = model.get("name", model.get("slug", "unknown-model"))
    evaluations = model.get("evaluations", {})

    metrics = []
    for key, value in evaluations.items():
        metrics.append(
            {
                "name": key.replace("_", " ").title(),
                "type": key,
                "value": value,
            }
        )

    model_index = [
        {
            "name": model_name,
            "results": [
                {
                    "task": {"type": task_type},
                    "dataset": {"name": dataset_name, "type": dataset_type},
                    "metrics": metrics,
                    "source": {
                        "name": "Artificial Analysis API",
                        "url": "https://artificialanalysis.ai",
                    },
                }
            ],
        }
    ]

    return model_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator-slug", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    args = parser.parse_args()

    aa_evaluations_data = get_model_evaluations_data(
        creator_slug=args.creator_slug, model_name=args.model_name
    )

    model_index = aa_evaluations_to_model_index(model=aa_evaluations_data)

    card = ModelCard.load(args.repo_id)
    card.data["model-index"] = model_index

    commit_message = (
        f"为 {args.model_name} 添加 Artificial Analysis 评估结果"
    )
    commit_description = (
        f"此提交将 {args.model_name} 模型的 Artificial Analysis 评估结果添加到此仓库。 "
        "要查看分数，请访问 [Artificial Analysis](https://artificialanalysis.ai) 网站。"
    )

    card.push_to_hub(
        args.repo_id,
        token=HF_TOKEN,
        commit_message=commit_message,
        commit_description=commit_description,
        create_pr=True,
    )


if __name__ == "__main__":
    main()
