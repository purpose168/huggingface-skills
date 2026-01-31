#!/usr/bin/env python3
"""
收集 hf-skills 组织的参与积分。

跟踪所有仓库（模型、数据集、空间）的用户活动并计数：
- 每个开启的讨论 1 分
- 每条发表的评论 1 分
- 每个开启的 PR 1 分
- 每个拥有/创建的仓库 1 分

结果将保存到黑客排行榜的数据集。

使用方法:
    HF_TOKEN=$HF_TOKEN python collect_points.py [--push-to-hub]
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

API_BASE = "https://huggingface.co/api"
ORG_NAME = "hf-skills"
USER_AGENT = "hf-skills-leaderboard/1.0"
DISCUSSION_LIMIT = 100  # Max discussions to fetch per repo
TRENDING_LIMIT = 50  # Number of trending repos to scan for external PRs


@dataclass
class UserStats:
    """跟踪单个用户的参与统计数据。"""

    username: str
    is_org_member: bool = True
    discussions_opened: int = 0
    comments_made: int = 0
    prs_opened: int = 0
    repos_owned: int = 0
    activities: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_points(self) -> int:
        return self.discussions_opened + self.comments_made + self.prs_opened + self.repos_owned

    def to_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "is_org_member": self.is_org_member,
            "total_points": self.total_points,
            "discussions_opened": self.discussions_opened,
            "comments_made": self.comments_made,
            "prs_opened": self.prs_opened,
            "repos_owned": self.repos_owned,
        }


class PointsCollector:
    """从 hf-skills 组织收集参与积分。"""

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.user_stats: dict[str, UserStats] = {}
        self.logs: list[str] = []

    def log(self, message: str) -> None:
        """添加日志消息。"""
        print(message)
        self.logs.append(message)

    def _fetch_org_members(self) -> list[str]:
        """获取组织的所有成员。"""
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)
            members = list(api.list_organization_members(ORG_NAME))
            usernames = [m.username for m in members if m.username]
            self.log(f"👥 找到 {len(usernames)} 个组织成员")
            return usernames
        except Exception as e:
            self.log(f"⚠️ 获取组织成员失败: {e}")
            # 备用方案：直接尝试API
            try:
                url = f"{API_BASE}/organizations/{ORG_NAME}/members"
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                members = response.json()
                usernames = [m.get("user") or m.get("username") or m.get("name") for m in members]
                usernames = [u for u in usernames if u]
                self.log(f"👥 找到 {len(usernames)} 个组织成员 (通过API)")
                return usernames
            except Exception as e2:
                self.log(f"⚠️ 备用方案也失败: {e2}")
                return []

    def collect_all(self) -> dict[str, UserStats]:
        """从组织中的所有仓库收集积分。"""
        self.log(f"🔍 扫描组织: {ORG_NAME}")

        # 首先，获取所有组织成员并初始化他们的统计数据
        members = self._fetch_org_members()
        for username in members:
            self.user_stats[username] = UserStats(username=username)

        # 从所有仓库类型收集
        models = self._list_repos("models")
        datasets = self._list_repos("datasets")
        spaces = self._list_repos("spaces")

        all_repos = [
            *[(r, "model") for r in models],
            *[(r, "dataset") for r in datasets],
            *[(r, "space") for r in spaces],
        ]

        self.log(f"📦 找到 {len(models)} 个模型, {len(datasets)} 个数据集, {len(spaces)} 个空间")

        for repo_info, repo_type in all_repos:
            repo_id = repo_info.get("id") or repo_info.get("modelId")
            if not repo_id:
                continue

            # 为仓库所有者加分
            owner = repo_info.get("author") or repo_id.split("/")[0]
            if owner and owner != ORG_NAME:
                self._add_point(owner, "repos_owned", repo_id, "repo_created")

            # 扫描讨论
            self._scan_discussions(repo_id, repo_type)

        return dict(self.user_stats)

    def scan_external_repos(self, repo_types: list[str] | None = None) -> None:
        """扫描整个 Hub 上的热门仓库，查找组织成员的 PR。

        参数:
            repo_types: 要扫描的仓库类型列表。选项："models"、"datasets"、"spaces"。
                       如果为 None，则扫描所有类型。
        """
        org_members = set(self.user_stats.keys())
        if not org_members:
            self.log("⚠️ 未加载组织成员。请先运行 collect_all()。")
            return

        if repo_types is None:
            repo_types = ["models", "datasets", "spaces"]

        self.log(f"🌐 扫描热门仓库，查找 {len(org_members)} 个组织成员的 PR...")
        self.log(f"📂 仓库类型: {', '.join(repo_types)}")

        for repo_type in repo_types:
            trending = self._fetch_trending(repo_type)
            self.log(f"📈 扫描 {len(trending)} 个热门 {repo_type}...")

            for repo_info in trending:
                repo_id = repo_info.get("id") or repo_info.get("modelId")
                if not repo_id:
                    continue

                # 跳过组织仓库（已扫描）
                if repo_id.startswith(f"{ORG_NAME}/"):
                    continue

                # 使用作者过滤器扫描每个组织成员的 PR/讨论
                self._scan_repo_for_members(repo_id, repo_type, org_members)

    def _fetch_trending(self, repo_type: str) -> list[dict[str, Any]]:
        """获取给定类型的热门仓库。"""
        endpoint = f"{API_BASE}/{repo_type}"
        params = {"sort": "trendingScore", "limit": TRENDING_LIMIT}

        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.log(f"⚠️ 获取热门 {repo_type} 失败: {e}")
            return []

    def _scan_repo_for_members(self, repo_id: str, repo_type: str, org_members: set[str]) -> None:
        """使用作者过滤器扫描仓库的讨论，查找组织成员的活动。"""
        # 从 repo_id 解析命名空间和仓库名
        parts = repo_id.split("/")
        if len(parts) != 2:
            return
        namespace, repo = parts

        for member in org_members:
            # 使用作者过滤器进行高效查询
            self._fetch_member_discussions(
                repo_type=repo_type,
                namespace=namespace,
                repo=repo,
                author=member,
                discussion_type="pull_request",
            )
            self._fetch_member_discussions(
                repo_type=repo_type,
                namespace=namespace,
                repo=repo,
                author=member,
                discussion_type="discussion",
            )

    def _fetch_member_discussions(
        self,
        repo_type: str,
        namespace: str,
        repo: str,
        author: str,
        discussion_type: str = "all",
    ) -> None:
        """获取特定作者在仓库中的讨论。

        使用: GET /api/{repoType}/{namespace}/{repo}/discussions?author={author}&type={type}
        """
        url = f"{API_BASE}/{repo_type}/{namespace}/{repo}/discussions"
        params = {
            "author": author,
            "type": discussion_type,
            "status": "all",
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            return

        discussions = data.get("discussions", [])
        repo_id = f"{namespace}/{repo}"

        for discussion in discussions:
            is_pr = discussion.get("isPullRequest", False)
            disc_num = discussion.get("num")

            if is_pr:
                self._add_point(author, "prs_opened", repo_id, "external_pr", disc_num)
                self.log(f"🔀 找到 {author} 在 {repo_id} 上的 PR")
            else:
                self._add_point(author, "discussions_opened", repo_id, "external_discussion", disc_num)
                self.log(f"💬 找到 {author} 在 {repo_id} 上的讨论")

            # 计算讨论中的评论数
            num_comments = discussion.get("numComments", 0)
            if num_comments > 0:
                self._fetch_discussion_comments(repo_type, namespace, repo, disc_num, author)

    def _fetch_discussion_comments(
        self,
        repo_type: str,
        namespace: str,
        repo: str,
        disc_num: int,
        target_author: str,
    ) -> None:
        """获取讨论的评论并计算目标作者的评论数。"""
        url = f"{API_BASE}/{repo_type}/{namespace}/{repo}/discussions/{disc_num}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            return

        repo_id = f"{namespace}/{repo}"
        events = data.get("events", [])
        for event in events:
            if event.get("type") == "comment":
                author_info = event.get("author", {}) or {}
                author = author_info.get("name") or author_info.get("fullname")
                if author == target_author:
                    self._add_point(author, "comments_made", repo_id, "external_comment", disc_num)

    def _list_repos(self, repo_type: str) -> list[dict[str, Any]]:
        """列出组织中给定类型的所有仓库。"""
        endpoint = f"{API_BASE}/{repo_type}"
        params = {"author": ORG_NAME, "limit": 1000}

        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.log(f"⚠️ 列出 {repo_type} 失败: {e}")
            return []

    def _scan_discussions(self, repo_id: str, repo_type: str) -> None:
        """扫描仓库的所有讨论并计算参与度。"""
        # 映射仓库类型到 API 路径
        type_map = {"model": "models", "dataset": "datasets", "space": "spaces"}
        api_type = type_map.get(repo_type, "models")

        url = f"{API_BASE}/{api_type}/{repo_id}/discussions"

        try:
            response = self.session.get(url, params={"limit": DISCUSSION_LIMIT}, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            self.log(f"⚠️ 获取 {repo_id} 的讨论失败: {e}")
            return

        discussions = data.get("discussions", [])
        if not discussions:
            return

        self.log(f"💬 {repo_id}: 找到 {len(discussions)} 个讨论")

        for discussion in discussions:
            self._process_discussion(repo_id, api_type, discussion)

    def _process_discussion(self, repo_id: str, api_type: str, discussion: dict[str, Any]) -> None:
        """处理单个讨论及其评论。"""
        author_info = discussion.get("author", {}) or {}
        author = author_info.get("name") or author_info.get("fullname")
        is_pr = discussion.get("isPullRequest", False)
        disc_num = discussion.get("num")

        if author and author != ORG_NAME:
            activity_type = "pr_opened" if is_pr else "discussion_opened"
            point_type = "prs_opened" if is_pr else "discussions_opened"
            self._add_point(author, point_type, repo_id, activity_type, disc_num)

        # 获取讨论详情以获取评论
        if disc_num:
            self._fetch_comments(repo_id, api_type, disc_num)

    def _fetch_comments(self, repo_id: str, api_type: str, disc_num: int) -> None:
        """获取并计算讨论的评论数。"""
        url = f"{API_BASE}/{api_type}/{repo_id}/discussions/{disc_num}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            # 静默跳过失败的评论获取
            return

        events = data.get("events", [])
        for event in events:
            event_type = event.get("type")
            # 计算评论（不是初始帖子、状态变更等）
            if event_type == "comment":
                author_info = event.get("author", {}) or {}
                author = author_info.get("name") or author_info.get("fullname")
                if author and author != ORG_NAME:
                    self._add_point(author, "comments_made", repo_id, "comment", disc_num)

    def _add_point(
        self,
        username: str,
        point_type: str,
        repo_id: str,
        activity_type: str,
        disc_num: int | None = None,
    ) -> None:
        """为用户的统计数据添加积分。"""
        if not username:
            return

        # 为不在组织中的用户（外部贡献者）初始化统计数据
        if username not in self.user_stats:
            self.user_stats[username] = UserStats(username=username, is_org_member=False)

        stats = self.user_stats[username]
        current = getattr(stats, point_type, 0)
        setattr(stats, point_type, current + 1)

        stats.activities.append(
            {
                "type": activity_type,
                "repo_id": repo_id,
                "discussion_num": disc_num,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """获取按总积分排序的排行榜。"""
        leaderboard = [stats.to_dict() for stats in self.user_stats.values()]
        leaderboard.sort(key=lambda x: x["total_points"], reverse=True)
        return leaderboard

    def save_json(self, filepath: str) -> None:
        """将排行榜保存到JSON文件。"""
        leaderboard = self.get_leaderboard()
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "organization": ORG_NAME,
            "total_participants": len(leaderboard),
            "leaderboard": leaderboard,
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        self.log(f"💾 已保存排行榜到 {filepath}")

    def push_to_hub(self, repo_id: str = "hf-skills/hackers-leaderboard") -> None:
        """将排行榜数据推送到HF数据集。"""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            self.log("❌ huggingface_hub 未安装。运行：pip install huggingface_hub")
            return

        api = HfApi()
        leaderboard = self.get_leaderboard()

        # 创建JSONL格式数据集
        jsonl_content = "\n".join(json.dumps(row) for row in leaderboard)

        # 同时创建元数据文件
        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "organization": ORG_NAME,
            "total_participants": len(leaderboard),
            "total_points": sum(row["total_points"] for row in leaderboard),
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
    parser = argparse.ArgumentParser(description="从 hf-skills 组织收集参与积分")
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
        default="hf-skills/hackers-leaderboard",
        help="用于推送的 HF 数据集仓库 ID",
    )
    parser.add_argument(
        "--scan-external",
        action="store_true",
        help="同时扫描热门仓库以获取组织成员的 PR/讨论",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        nargs="+",
        choices=["models", "datasets", "spaces"],
        default=None,
        help="要扫描的仓库类型（用于 --scan-external）。默认：所有类型",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️ 未找到 HF_TOKEN。某些请求可能会受到速率限制。")

    collector = PointsCollector(token=token)
    collector.collect_all()

    # 可选：扫描外部仓库以获取成员活动
    if args.scan_external:
        collector.scan_external_repos(repo_types=args.repo_type)

    # 打印排行榜
    print("\n" + "=" * 50)
    print("🏆 黑客排行榜")
    print("=" * 50)

    leaderboard = collector.get_leaderboard()
    for i, entry in enumerate(leaderboard[:20], 1):
        print(
            f"{i:2}. {entry['username']:20} - {entry['total_points']:4} 分 "
            f"(💬{entry['discussions_opened']} 📝{entry['comments_made']} "
            f"🔀{entry['prs_opened']} 📦{entry['repos_owned']})"
        )

    if len(leaderboard) > 20:
        print(f"   ... 还有 {len(leaderboard) - 20} 个参与者")

    print("=" * 50)
    print(f"总参与者数: {len(leaderboard)}")
    print(f"已颁发总积分: {sum(e['total_points'] for e in leaderboard)}")

    # 保存到本地
    collector.save_json(args.output)

    # 如果请求，推送到 hub
    if args.push_to_hub:
        collector.push_to_hub(args.repo_id)


if __name__ == "__main__":
    main()
