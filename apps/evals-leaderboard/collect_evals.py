#!/usr/bin/env python3
"""
从热门模型的 model-index 元数据中收集评估分数。

扫描 Hub 上的热门文本生成模型，并从其 model-index 元数据或开放的拉取请求中提取基准测试分数。

结果将保存到评估排行榜的数据集。

使用方法:
    python collect_evals.py [--push-to-hub]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

API_BASE = "https://huggingface.co/api"
PIPELINE_FILTER = "text-generation"
TRENDING_LIMIT = 50
TRENDING_FETCH_LIMIT = 100
PR_SCAN_LIMIT = 40
USER_AGENT = "skills-evals-leaderboard/0.3"


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def _coerce_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("%"):
            candidate = candidate[:-1]
        try:
            return float(candidate)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    label: str
    aliases: tuple[str, ...]

    def matches(self, fields: List[str]) -> bool:
        for alias in self.aliases:
            alias_norm = _normalize(alias)
            if not alias_norm:
                continue
            for field in fields:
                if alias_norm in field:
                    return True
        return False


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "mmlu": BenchmarkSpec(
        key="mmlu",
        label="MMLU",
        aliases=("mmlu", "massive multitask language understanding"),
    ),
    "bigcodebench": BenchmarkSpec(
        key="bigcodebench",
        label="BigCodeBench",
        aliases=("bigcodebench", "big code bench"),
    ),
    "arc_mc": BenchmarkSpec(
        key="arc_mc",
        label="ARC MC",
        aliases=(
            "arc mc",
            "arc-challenge",
            "arc challenge",
            "arc multiple choice",
            "arc c",
        ),
    ),
}


class EvalsCollector:
    """从 model-index 元数据中收集评估分数。"""

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.logs: List[str] = []
        self.results: List[Dict[str, Any]] = []

    def log(self, message: str) -> None:
        """添加日志消息。"""
        print(message)
        self.logs.append(message)

    def collect_all(self) -> List[Dict[str, Any]]:
        """从热门模型中收集评估分数。"""
        self.log("🔍 获取热门文本生成模型...")
        trending = self._fetch_trending_models()

        for entry in trending:
            repo_id = entry.get("modelId") or entry.get("id")
            if not repo_id:
                continue
            scores = self._collect_scores(repo_id)
            if scores["scores"]:
                self.results.extend(self._format_scores(repo_id, scores["scores"]))

        self.log(f"✅ 已收集 {len(self.results)} 个评估条目")
        return self.results

    def _fetch_trending_models(self) -> List[Dict[str, Any]]:
        params = {"sort": "trendingScore", "limit": TRENDING_FETCH_LIMIT}
        response = self.session.get(
            f"{API_BASE}/models",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError("意外的热门模型响应。")
        filtered = [
            model
            for model in data
            if (model.get("pipeline_tag") == PIPELINE_FILTER or PIPELINE_FILTER in (model.get("tags") or []))
        ]
        if not filtered:
            self.log("⚠️ 热门模型中没有文本生成模型。")
            return []
        limited = filtered[:TRENDING_LIMIT]
        self.log(f"📊 找到 {len(limited)} 个热门文本生成模型")
        return limited

    def _collect_scores(self, repo_id: str) -> Dict[str, Any]:
        owner = repo_id.split("/")[0]
        card_meta = self._read_model_card(repo_id)
        model_index = card_meta.get("model-index")
        if model_index:
            self.log(f"✅ {repo_id}: 找到模型卡片元数据。")
            scores = self._extract_scores(
                repo_id=repo_id,
                model_index=model_index,
                contributor=owner,
                source_type="model-card",
                source_url=f"https://huggingface.co/{repo_id}",
                revision="main",
            )
            if scores:
                return {"model_id": repo_id, "scores": scores}

        prs = self._fetch_pull_requests(repo_id)
        for pr in prs:
            revision = f"refs/pr/{pr['num']}"
            pr_meta = self._read_model_card(repo_id, revision=revision)
            pr_index = pr_meta.get("model-index")
            if not pr_index:
                continue
            author_info = pr.get("author", {}) or {}
            contributor = author_info.get("name") or author_info.get("fullname") or "unknown-author"
            discussion_path = f"{repo_id}/discussions/{pr['num']}"
            source_url = f"https://huggingface.co/{discussion_path}"
            scores = self._extract_scores(
                repo_id=repo_id,
                model_index=pr_index,
                contributor=contributor,
                source_type="pull-request",
                source_url=source_url,
                revision=revision,
            )
            if scores:
                note = f"📝 {repo_id}: PR #{pr['num']} by {contributor}。"
                self.log(note)
                return {"model_id": repo_id, "scores": scores}

        self.log(f"⚠️ {repo_id}: 未找到目标基准测试。")
        return {"model_id": repo_id, "scores": {}}

    def _read_model_card(
        self,
        repo_id: str,
        revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="model",
                revision=revision,
                token=self.token,
            )
        except HfHubHTTPError as err:
            ctx = f"{repo_id} ({revision or 'main'})"
            self.log(f"🚫 {ctx}: README 下载失败 ({err})。")
            return {}
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        return self._parse_front_matter(text)

    @staticmethod
    def _parse_front_matter(content: str) -> Dict[str, Any]:
        content = content.lstrip("\ufeff")
        if not content.startswith("---"):
            return {}
        lines = content.splitlines()
        end_idx = None
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_idx = idx
                break
        if end_idx is None:
            return {}
        front_matter = "\n".join(lines[1:end_idx])
        try:
            data = yaml.safe_load(front_matter) or {}
            return data if isinstance(data, dict) else {}
        except yaml.YAMLError:
            return {}

    def _fetch_pull_requests(self, repo_id: str) -> List[Dict[str, Any]]:
        url = f"{API_BASE}/models/{repo_id}/discussions"
        try:
            response = self.session.get(
                url,
                params={"limit": PR_SCAN_LIMIT},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as err:
            self.log(f"🚫 {repo_id}: PR 列表请求失败 ({err})。")
            return []

        payload = response.json()
        discussions = payload.get("discussions", [])
        prs = [disc for disc in discussions if disc.get("isPullRequest")]
        prs.sort(key=lambda item: item.get("createdAt", ""), reverse=True)
        if prs:
            self.log(f"📬 {repo_id}: 扫描 {len(prs)} 个拉取请求。")
        return prs

    def _extract_scores(
        self,
        repo_id: str,
        model_index: Any,
        contributor: str,
        source_type: str,
        source_url: str,
        revision: str,
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(model_index, list):
            return {}
        scores: Dict[str, Dict[str, Any]] = {}
        for entry in model_index:
            if not isinstance(entry, dict):
                continue
            model_name = entry.get("name") or repo_id.split("/")[-1]
            for result in entry.get("results", []):
                dataset_info = result.get("dataset") or {}
                dataset_name = dataset_info.get("name")
                dataset_type = dataset_info.get("type")
                task_info = result.get("task") or {}
                task_type = task_info.get("type")
                for metric in result.get("metrics", []):
                    benchmark_key = self._match_benchmark(
                        dataset_name,
                        dataset_type,
                        metric,
                    )
                    if not benchmark_key:
                        continue
                    raw_value = metric.get("value")
                    value = _coerce_score(raw_value)
                    if value is None:
                        continue
                    unit = metric.get("unit") or ""
                    is_pct = isinstance(raw_value, str) and raw_value.strip().endswith("%")
                    if not unit and is_pct:
                        unit = "%"
                    metric_name = metric.get("name") or metric.get("type") or ""
                    payload = {
                        "model": repo_id,
                        "model_name": model_name,
                        "benchmark_key": benchmark_key,
                        "benchmark_label": BENCHMARKS[benchmark_key].label,
                        "value": value,
                        "unit": unit,
                        "dataset": dataset_name or dataset_type or "",
                        "task_type": task_type or "",
                        "metric_name": metric_name,
                        "contributor": contributor,
                        "source_type": source_type,
                        "source_url": source_url,
                        "revision": revision,
                    }
                    existing = scores.get(benchmark_key)
                    if not existing or value > existing["value"]:
                        scores[benchmark_key] = payload
        return scores

    def _match_benchmark(
        self,
        dataset_name: Optional[str],
        dataset_type: Optional[str],
        metric: Dict[str, Any],
    ) -> Optional[str]:
        fields = [
            _normalize(dataset_name),
            _normalize(dataset_type),
            _normalize(metric.get("name")),
            _normalize(metric.get("type")),
        ]
        fields = [field for field in fields if field]
        for key, spec in BENCHMARKS.items():
            if spec.matches(fields):
                return key
        return None

    def _format_scores(self, model_id: str, scores: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将分数格式化为数据集中的扁平记录。"""
        rows = []
        for benchmark_key, payload in scores.items():
            rows.append(
                {
                    "model_id": model_id,
                    "benchmark": payload["benchmark_label"],
                    "benchmark_key": benchmark_key,
                    "score": round(payload["value"], 2),
                    "source_type": payload["source_type"],
                    "source_url": payload["source_url"],
                    "contributor": payload["contributor"],
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        return rows

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """获取按分数降序排序的结果。"""
        return sorted(self.results, key=lambda x: x["score"], reverse=True)

    def save_json(self, filepath: str) -> None:
        """将排行榜保存到JSON文件。"""
        leaderboard = self.get_leaderboard()
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(leaderboard),
            "benchmarks": list(BENCHMARKS.keys()),
            "leaderboard": leaderboard,
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        self.log(f"💾 已保存排行榜到 {filepath}")

    def push_to_hub(self, repo_id: str = "hf-skills/evals-leaderboard") -> None:
        """将排行榜数据推送到HF数据集。"""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            self.log("❌ huggingface_hub 未安装。运行：pip install huggingface_hub")
            return

        api = HfApi(token=self.token)
        leaderboard = self.get_leaderboard()

        # 创建JSONL格式数据集
        jsonl_content = "\n".join(json.dumps(row) for row in leaderboard)

        # 创建元数据文件
        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(leaderboard),
            "models_with_scores": len(set(r["model_id"] for r in leaderboard)),
            "benchmarks": list(BENCHMARKS.keys()),
        }

        try:
            # 如果仓库不存在则创建
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            self.log(f"📁 确保数据集仓库存在：{repo_id}")

            # 上传排行榜数据
            api.upload_file(
                path_or_fileobj=jsonl_content.encode(),
                path_in_repo="data/leaderboard.jsonl",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Update leaderboard - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
            )

            # 上传元数据
            api.upload_file(
                path_or_fileobj=json.dumps(metadata, indent=2).encode(),
                path_in_repo="data/metadata.json",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Update metadata - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
            )

            self.log(f"🚀 已将排行榜推送到 {repo_id}")
        except Exception as e:
            self.log(f"❌ 推送到hub失败：{e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="从 model-index 元数据收集评估分数")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="将结果推送到 HF 数据集",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="leaderboard.json",
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="hf-skills/evals-leaderboard",
        help="用于推送的 HF 数据集仓库 ID",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️ 未找到 HF_TOKEN。某些请求可能会受到速率限制。")

    collector = EvalsCollector(token=token)
    collector.collect_all()

    # 打印排行榜摘要
    print("\n" + "=" * 60)
    print("📊 评估排行榜")
    print("=" * 60)

    leaderboard = collector.get_leaderboard()
    for entry in leaderboard[:20]:
        print(f"{entry['model_id']:40} | {entry['benchmark']:12} | {entry['score']:6.2f}")

    if len(leaderboard) > 20:
        print(f"   ... 还有 {len(leaderboard) - 20} 个条目")

    print("=" * 60)
    print(f"总条目数: {len(leaderboard)}")
    print(f"有分数的模型: {len(set(r['model_id'] for r in leaderboard))}")

    # 保存到本地
    collector.save_json(args.output)

    # 如果请求，推送到 hub
    if args.push_to_hub:
        collector.push_to_hub(args.repo_id)


if __name__ == "__main__":
    main()
