#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub",
#     "pyyaml",
#     "requests",
#     "python-dotenv",
# ]
# ///
"""
Hugging Face Hub 论文管理器
管理论文索引、链接、作者身份和文章创建。
"""

import argparse
import os
import sys
import re
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from huggingface_hub import HfApi, hf_hub_download, HfFolder
    import yaml
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    print(f"错误: 缺少必需的依赖项: {e}")
    print("安装命令: uv add huggingface_hub pyyaml requests python-dotenv")
    sys.exit(1)

# 加载环境变量
load_dotenv()


class PaperManager:
    """管理 Hugging Face Hub 上的论文发布操作。"""

    def __init__(self, hf_token: Optional[str] = None):
        """
        使用 HF 令牌初始化论文管理器。

        参数:
            hf_token: Hugging Face API 令牌,如果未提供则从环境变量获取
        """
        # 获取 Hugging Face 令牌,优先级: 参数 > 环境变量 > HfFolder
        self.token = hf_token or os.getenv("HF_TOKEN") or HfFolder.get_token()
        if not self.token:
            print("警告: 未找到 HF_TOKEN。某些操作将失败。")
        # 初始化 Hugging Face API 客户端
        self.api = HfApi(token=self.token)

    def index_paper(self, arxiv_id: str) -> Dict[str, Any]:
        """
        从 arXiv 在 Hugging Face 上索引论文。

        参数:
            arxiv_id: arXiv 标识符(例如 "2301.12345")

        返回:
            dict: 状态信息,包含索引状态和 URL
        """
        # 清理 arXiv ID,移除前缀和多余字符
        arxiv_id = self._clean_arxiv_id(arxiv_id)

        print(f"正在 Hugging Face 上索引论文 {arxiv_id}...")

        # 构建论文 URL
        paper_url = f"https://huggingface.co/papers/{arxiv_id}"

        try:
            # 检查论文是否已存在
            response = requests.get(paper_url, timeout=10)
            if response.status_code == 200:
                print(f"✓ 论文已在 {paper_url} 索引")
                return {"status": "exists", "url": paper_url}
            else:
                # 论文未索引,提示用户访问 URL 触发索引
                print(f"论文未索引。访问 {paper_url} 以触发索引。")
                print("首次访问 URL 时,论文将自动索引。")
                return {"status": "not_indexed", "url": paper_url, "action": "visit_url"}
        except requests.RequestException as e:
            print(f"检查论文状态时出错: {e}")
            return {"status": "error", "message": str(e)}

    def check_paper(self, arxiv_id: str) -> Dict[str, Any]:
        """
        检查论文是否存在于 Hugging Face 上。

        参数:
            arxiv_id: arXiv 标识符

        返回:
            dict: 论文状态和元数据
        """
        # 清理 arXiv ID
        arxiv_id = self._clean_arxiv_id(arxiv_id)
        # 构建 Hugging Face 论文 URL
        paper_url = f"https://huggingface.co/papers/{arxiv_id}"

        try:
            # 发送请求检查论文是否存在
            response = requests.get(paper_url, timeout=10)
            if response.status_code == 200:
                # 论文存在,返回详细信息
                return {
                    "exists": True,
                    "url": paper_url,
                    "arxiv_id": arxiv_id,
                    "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}"
                }
            else:
                # 论文不存在,返回索引 URL
                return {
                    "exists": False,
                    "arxiv_id": arxiv_id,
                    "index_url": paper_url,
                    "message": f"访问 {paper_url} 以索引此论文"
                }
        except requests.RequestException as e:
            # 请求异常,返回错误信息
            return {"exists": False, "error": str(e)}

    def link_paper_to_repo(
        self,
        repo_id: str,
        arxiv_id: str,
        repo_type: str = "model",
        citation: Optional[str] = None,
        create_pr: bool = False
    ) -> Dict[str, Any]:
        """
        将论文链接到模型/数据集/空间仓库。

        参数:
            repo_id: 仓库标识符(例如 "username/repo-name")
            arxiv_id: arXiv 标识符
            repo_type: 仓库类型("model"、"dataset" 或 "space")
            citation: 可选的完整引用文本
            create_pr: 创建 PR 而不是直接提交

        返回:
            dict: 操作状态和结果信息
        """
        # 清理 arXiv ID
        arxiv_id = self._clean_arxiv_id(arxiv_id)

        print(f"正在将论文 {arxiv_id} 链接到 {repo_type} {repo_id}...")

        try:
            # 从仓库下载当前 README 文件
            readme_path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type=repo_type,
                token=self.token
            )

            # 读取 README 内容
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析或创建 YAML 前置元数据,并添加论文引用
            updated_content = self._add_paper_to_readme(content, arxiv_id, citation)

            # 上传更新后的 README
            commit_message = f"添加论文引用: arXiv:{arxiv_id}"

            if create_pr:
                # 创建 PR(基础版本中未实现)
                print("PR 创建功能尚未实现。直接提交。")

            # 上传文件到仓库
            self.api.upload_file(
                path_or_fileobj=updated_content.encode('utf-8'),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
                token=self.token
            )

            # 构建论文和仓库的 URL
            paper_url = f"https://huggingface.co/papers/{arxiv_id}"
            repo_url = f"https://huggingface.co/{repo_id}"

            print(f"✓ 成功将论文链接到仓库")
            print(f"  论文: {paper_url}")
            print(f"  仓库: {repo_url}")

            return {
                "status": "success",
                "paper_url": paper_url,
                "repo_url": repo_url,
                "arxiv_id": arxiv_id
            }

        except Exception as e:
            print(f"链接论文时出错: {e}")
            return {"status": "error", "message": str(e)}

    def _add_paper_to_readme(
        self,
        content: str,
        arxiv_id: str,
        citation: Optional[str] = None
    ) -> str:
        """
        将论文引用添加到 README 内容中。

        参数:
            content: 当前 README 内容
            arxiv_id: arXiv 标识符
            citation: 可选的引用文本

        返回:
            str: 更新后的 README 内容
        """
        # 构建 arXiv 和 Hugging Face 论文 URL
        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
        hf_paper_url = f"https://huggingface.co/papers/{arxiv_id}"

        # 检查是否存在 YAML 前置元数据
        yaml_pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(yaml_pattern, content, re.DOTALL)

        if match:
            # YAML 存在,检查论文是否已被引用
            if arxiv_id in content:
                print(f"论文 {arxiv_id} 已在 README 中引用")
                return content

            # 添加到现有内容(YAML 之后)
            yaml_end = match.end()
            before = content[:yaml_end]
            after = content[yaml_end:]
        else:
            # 没有 YAML,添加最小前置元数据
            yaml_content = "---\n---\n\n"
            before = yaml_content
            after = content

        # 添加论文引用部分
        paper_section = f"\n## 论文\n\n"
        # 根据内容判断是模型还是工作
        paper_section += f"此{'模型' if 'model' in content.lower() else '工作'}基于以下研究中呈现的内容:\n\n"
        paper_section += f"**[在 arXiv 上查看]({arxiv_url})** | "
        paper_section += f"**[在 Hugging Face 上查看]({hf_paper_url})**\n\n"

        # 如果提供了引用文本,添加引用部分
        if citation:
            paper_section += f"### 引用\n\n```bibtex\n{citation}\n```\n\n"

        # 在 YAML 之后、主要内容之前插入
        updated_content = before + paper_section + after

        return updated_content

    def create_research_article(
        self,
        template: str,
        title: str,
        output: str,
        authors: Optional[str] = None,
        abstract: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        从模板创建研究文章。

        参数:
            template: 模板名称("standard"、"modern"、"arxiv"、"ml-report")
            title: 论文标题
            output: 输出文件名
            authors: 逗号分隔的作者姓名
            abstract: 摘要文本

        返回:
            dict: 创建状态和结果信息
        """
        print(f"正在使用 '{template}' 模板创建研究文章...")

        # 加载模板文件
        template_dir = Path(__file__).parent.parent / "templates"
        template_file = template_dir / f"{template}.md"

        # 检查模板文件是否存在
        if not template_file.exists():
            return {
                "status": "error",
                "message": f"模板 '{template}' 在 {template_file} 未找到"
            }

        # 读取模板内容
        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()

        # 替换占位符
        content = template_content.replace("{{TITLE}}", title)
        content = content.replace("{{DATE}}", datetime.now().strftime("%Y-%m-%d"))

        # 替换作者信息
        if authors:
            content = content.replace("{{AUTHORS}}", authors)
        else:
            content = content.replace("{{AUTHORS}}", "您的姓名")

        # 替换摘要
        if abstract:
            content = content.replace("{{ABSTRACT}}", abstract)
        else:
            content = content.replace("{{ABSTRACT}}", "摘要待写...")

        # 写入输出文件
        with open(output, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✓ 研究文章已创建于 {output}")

        return {
            "status": "success",
            "output": output,
            "template": template
        }

    def get_arxiv_info(self, arxiv_id: str) -> Dict[str, Any]:
        """
        从 arXiv API 获取论文信息。

        参数:
            arxiv_id: arXiv 标识符

        返回:
            dict: 论文元数据,包括标题、作者、摘要等
        """
        # 清理 arXiv ID
        arxiv_id = self._clean_arxiv_id(arxiv_id)
        # 构建 arXiv API URL
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

        try:
            # 发送请求获取论文信息
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()

            # 解析 XML 响应(简化版)
            content = response.text

            # 使用正则表达式提取基本信息(更好的做法是使用 XML 解析器)
            title_match = re.search(r'<title>(.*?)</title>', content, re.DOTALL)
            authors_matches = re.findall(r'<name>(.*?)</name>', content)
            summary_match = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL)

            # 返回论文元数据
            return {
                "arxiv_id": arxiv_id,
                "title": title_match.group(1).strip() if title_match else None,
                "authors": authors_matches[1:] if len(authors_matches) > 1 else [],  # 跳过第一个(feed 标题)
                "abstract": summary_match.group(1).strip() if summary_match else None,
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            }
        except Exception as e:
            # 返回错误信息
            return {"error": str(e)}

    def generate_citation(
        self,
        arxiv_id: str,
        format: str = "bibtex"
    ) -> str:
        """
        为论文生成引用。

        参数:
            arxiv_id: arXiv 标识符
            format: 引用格式("bibtex"、"apa"、"mla")

        返回:
            str: 格式化的引用文本
        """
        # 获取论文信息
        info = self.get_arxiv_info(arxiv_id)

        # 检查是否出错
        if "error" in info:
            return f"获取论文信息时出错: {info['error']}"

        # 根据格式生成引用
        if format == "bibtex":
            # 生成 BibTeX 引用
            key = f"arxiv{arxiv_id.replace('.', '_')}"
            authors = " and ".join(info.get("authors", ["未知"]))
            title = info.get("title", "无标题")
            # 从 ID 提取年份(简化版)
            year = arxiv_id.split(".")[0][:2]
            year = f"20{year}" if int(year) < 50 else f"19{year}"

            # 构建 BibTeX 格式的引用
            citation = f"""@article{{{key},
  title={{{title}}},
  author={{{authors}}},
  journal={{arXiv preprint arXiv:{arxiv_id}}},
  year={{{year}}}
}}"""
            return citation

        # 其他格式尚未实现
        return f"格式 '{format}' 尚未实现"

    @staticmethod
    def _clean_arxiv_id(arxiv_id: str) -> str:
        """
        清理和标准化 arXiv ID。

        参数:
            arxiv_id: 原始 arXiv ID,可能包含前缀或 URL

        返回:
            str: 清理后的标准 arXiv ID
        """
        # 移除常见前缀和空白字符
        arxiv_id = arxiv_id.strip()
        # 移除 arxiv: 或 arXiv: 前缀
        arxiv_id = re.sub(r'^(arxiv:|arXiv:)', '', arxiv_id, flags=re.IGNORECASE)
        # 移除 URL 前缀
        arxiv_id = re.sub(r'https?://arxiv\.org/(abs|pdf)/', '', arxiv_id)
        # 移除 .pdf 后缀
        arxiv_id = arxiv_id.replace('.pdf', '')
        return arxiv_id


def main():
    """主命令行入口点。"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="Hugging Face Hub 论文管理器",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 添加子命令
    subparsers = parser.add_subparsers(dest="command", help="要执行的命令")

    # index 命令 - 索引论文
    index_parser = subparsers.add_parser("index", help="从 arXiv 索引论文")
    index_parser.add_argument("--arxiv-id", required=True, help="arXiv 论文 ID")

    # check 命令 - 检查论文是否存在
    check_parser = subparsers.add_parser("check", help="检查论文是否存在")
    check_parser.add_argument("--arxiv-id", required=True, help="arXiv 论文 ID")

    # link 命令 - 将论文链接到仓库
    link_parser = subparsers.add_parser("link", help="将论文链接到仓库")
    link_parser.add_argument("--repo-id", required=True, help="仓库 ID")
    link_parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    link_parser.add_argument("--arxiv-id", help="单个 arXiv ID")
    link_parser.add_argument("--arxiv-ids", help="逗号分隔的 arXiv ID 列表")
    link_parser.add_argument("--citation", help="完整引用文本")
    link_parser.add_argument("--create-pr", action="store_true", help="创建 PR 而不是直接提交")

    # create 命令 - 创建研究文章
    create_parser = subparsers.add_parser("create", help="创建研究文章")
    create_parser.add_argument("--template", required=True, help="模板名称")
    create_parser.add_argument("--title", required=True, help="论文标题")
    create_parser.add_argument("--output", required=True, help="输出文件名")
    create_parser.add_argument("--authors", help="逗号分隔的作者列表")
    create_parser.add_argument("--abstract", help="摘要文本")

    # info 命令 - 获取论文信息
    info_parser = subparsers.add_parser("info", help="获取论文信息")
    info_parser.add_argument("--arxiv-id", required=True, help="arXiv 论文 ID")
    info_parser.add_argument("--format", default="json", choices=["json", "text"])

    # citation 命令 - 生成引用
    citation_parser = subparsers.add_parser("citation", help="生成引用")
    citation_parser.add_argument("--arxiv-id", required=True, help="arXiv 论文 ID")
    citation_parser.add_argument("--format", default="bibtex", choices=["bibtex", "apa", "mla"])

    # search 命令 - 搜索论文
    search_parser = subparsers.add_parser("search", help="搜索论文")
    search_parser.add_argument("--query", required=True, help="搜索查询")

    # 解析命令行参数
    args = parser.parse_args()

    # 如果没有指定命令,显示帮助信息并退出
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 初始化论文管理器
    manager = PaperManager()

    # 根据命令执行相应操作
    if args.command == "index":
        # 索引论文
        result = manager.index_paper(args.arxiv_id)
        print(json.dumps(result, indent=2))

    elif args.command == "check":
        # 检查论文
        result = manager.check_paper(args.arxiv_id)
        print(json.dumps(result, indent=2))

    elif args.command == "link":
        # 收集所有 arXiv ID
        arxiv_ids = []
        if args.arxiv_id:
            arxiv_ids.append(args.arxiv_id)
        if args.arxiv_ids:
            arxiv_ids.extend([id.strip() for id in args.arxiv_ids.split(",")])

        # 检查是否提供了 arXiv ID
        if not arxiv_ids:
            print("错误: 必须提供 --arxiv-id 或 --arxiv-ids")
            sys.exit(1)

        # 为每个 arXiv ID 创建链接
        for arxiv_id in arxiv_ids:
            result = manager.link_paper_to_repo(
                repo_id=args.repo_id,
                arxiv_id=arxiv_id,
                repo_type=args.repo_type,
                citation=args.citation,
                create_pr=args.create_pr
            )
            print(json.dumps(result, indent=2))

    elif args.command == "create":
        # 创建研究文章
        result = manager.create_research_article(
            template=args.template,
            title=args.title,
            output=args.output,
            authors=args.authors,
            abstract=args.abstract
        )
        print(json.dumps(result, indent=2))

    elif args.command == "info":
        # 获取论文信息
        result = manager.get_arxiv_info(args.arxiv_id)
        if args.format == "json":
            # 以 JSON 格式输出
            print(json.dumps(result, indent=2))
        else:
            # 以文本格式输出
            if "error" in result:
                print(f"错误: {result['error']}")
            else:
                print(f"标题: {result.get('title')}")
                print(f"作者: {', '.join(result.get('authors', []))}")
                print(f"arXiv URL: {result.get('arxiv_url')}")
                print(f"\n摘要:\n{result.get('abstract')}")

    elif args.command == "citation":
        # 生成引用
        citation = manager.generate_citation(args.arxiv_id, args.format)
        print(citation)

    elif args.command == "search":
        # 搜索论文(功能开发中)
        print(f"正在搜索: {args.query}")
        print("搜索功能即将推出!")
        print(f"访问: https://huggingface.co/papers?search={args.query}")


if __name__ == "__main__":
    main()
