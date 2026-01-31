# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars>=1.31.0",
#     "huggingface-hub",
#     "datasets",
#     "ascii-graph",
# ]
# ///
"""
使用 Polars 流式分析 CommonCrawl 数据集中教育质量的时间趋势。

回答:"网络是否变得越来越有教育性?"

演示 Polars 与 HF Hub 的集成 - 处理 5000 万+文档而无需下载 300GB+ 数据。

使用示例:
    # 分析英文 PDF (默认)
    uv run finepdfs-stats.py

    # 分析所有 70+ 语言
    uv run finepdfs-stats.py --all-languages

    # 快速测试
    uv run finepdfs-stats.py --limit 10000 --show-plan

    # 保存结果到 HF Hub
    uv run finepdfs-stats.py --output-repo username/finepdfs-temporal-stats

    # 在 HF Jobs 上运行
    hf jobs uv run \\
        -s HF_TOKEN \\
        -e HF_XET_HIGH_PERFORMANCE=1 \\
        https://huggingface.co/datasets/uv-scripts/dataset-stats/raw/main/finepdfs-stats.py \\
        -- --output-repo username/stats
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import polars as pl
from ascii_graph import Pyasciigraph
from datasets import Dataset
from huggingface_hub import HfApi, create_repo, list_repo_tree, login

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# finepdfs-edu 的常见语言+文字代码
COMMON_LANGUAGES = {
    "eng_Latn": "英语 (拉丁文字)",
    "fra_Latn": "法语 (拉丁文字)",
    "deu_Latn": "德语 (拉丁文字)",
    "spa_Latn": "西班牙语 (拉丁文字)",
    "por_Latn": "葡萄牙语 (拉丁文字)",
    "ita_Latn": "意大利语 (拉丁文字)",
    "nld_Latn": "荷兰语 (拉丁文字)",
    "pol_Latn": "波兰语 (拉丁文字)",
    "rus_Cyrl": "俄语 (西里尔文字)",
    "zho_Hans": "中文 (简体)",
    "zho_Hant": "中文 (繁体)",
    "jpn_Jpan": "日语",
    "kor_Hang": "韩语",
    "ara_Arab": "阿拉伯语",
    "hin_Deva": "印地语 (天城文)",
}


def list_available_languages(dataset_id: str) -> list[str]:
    """列出数据集中可用的语言子集。"""
    try:
        tree = list_repo_tree(dataset_id, path_in_repo="data", repo_type="dataset")
        languages = [
            item.path.replace("data/", "")
            for item in tree
            if item.path.startswith("data/")
            and "/" not in item.path.replace("data/", "")
        ]
        return sorted(languages)
    except Exception as e:
        logger.warning(f"无法列出语言: {e}")
        return list(COMMON_LANGUAGES.keys())


def compute_temporal_stats(df: pl.LazyFrame, output_path: Path) -> pl.DataFrame:
    """单次扫描:按数据dump分组计算统计数据用于时间分析。"""
    query = df.group_by("dump").agg(
        pl.len().alias("doc_count"),
        pl.col("token_count").sum().alias("total_tokens"),
        pl.col("fw_edu_scores").list.mean().mean().alias("avg_edu_score"),
        (pl.col("fw_edu_scores").list.mean() >= 3).sum().alias("high_edu_count"),
    )
    query.sink_parquet(output_path, engine="streaming")
    return pl.read_parquet(output_path)


def compute_global_stats(temporal: pl.DataFrame) -> pl.DataFrame:
    """从时间分解中计算全局统计。"""
    total = temporal["doc_count"].sum()
    return pl.DataFrame(
        {
            "total_docs": [total],
            "total_tokens": [temporal["total_tokens"].sum()],
            "avg_edu_score": [
                (temporal["avg_edu_score"] * temporal["doc_count"]).sum() / total
            ],
            "high_edu_rate": [temporal["high_edu_count"].sum() / total],
            "num_dumps": [len(temporal)],
        }
    )


def format_temporal_stats(temporal: pl.DataFrame) -> pl.DataFrame:
    """格式化时间统计数据,包含 high_edu_rate,按时间排序。"""
    return (
        temporal.with_columns(
            (pl.col("high_edu_count") / pl.col("doc_count")).alias("high_edu_rate")
        )
        .select(["dump", "doc_count", "avg_edu_score", "high_edu_rate"])
        .sort(
            "dump"
        )  # 时间顺序 (CC-MAIN-2017-xx 在 CC-MAIN-2024-xx 之前)
    )


def create_ascii_charts(temporal_stats: pl.DataFrame) -> str:
    """创建 ASCII 条形图显示时间趋势。"""
    # 从 dump 名称中提取年份 (CC-MAIN-2024-42 -> 2024)
    # 按年份分组并计算平均值以获得更清晰的显示
    yearly = (
        temporal_stats.with_columns(
            pl.col("dump").str.extract(r"CC-MAIN-(\d{4})", 1).alias("year")
        )
        .group_by("year")
        .agg(
            pl.col("doc_count").sum(),
            pl.col("avg_edu_score").mean(),
            pl.col("high_edu_rate").mean(),
        )
        .sort("year")
    )

    lines = []

    # 高教育内容比率图表 (差异更明显)
    data_rate = [
        (row["year"], row["high_edu_rate"] * 100)
        for row in yearly.iter_rows(named=True)
    ]
    graph = Pyasciigraph(line_length=60, float_format="{0:.1f}%")
    lines.extend(graph.graph("高教育内容 (edu >= 3)", data_rate))

    lines.append("")

    # 平均教育分数图表
    data_score = [
        (row["year"], row["avg_edu_score"]) for row in yearly.iter_rows(named=True)
    ]
    graph2 = Pyasciigraph(line_length=60, float_format="{0:.2f}")
    lines.extend(graph2.graph("平均教育分数", data_score))

    return "\n".join(lines)


def create_readme(
    args,
    global_stats: pl.DataFrame,
    temporal_stats: pl.DataFrame,
    scan_time: float,
    ascii_charts: str,
) -> str:
    """为统计数据集创建 README 内容。"""
    stats = global_stats.to_dicts()[0]
    total_docs = stats.get("total_docs", 0)
    docs_per_sec = total_docs / scan_time if scan_time > 0 else 0

    # 获取首尾年份平均值用于趋势分析 (比单个 dump 更有代表性)
    yearly = (
        temporal_stats.with_columns(
            pl.col("dump").str.extract(r"CC-MAIN-(\d{4})", 1).alias("year")
        )
        .group_by("year")
        .agg(
            pl.col("doc_count").sum(),
            pl.col("avg_edu_score").mean(),
            pl.col("high_edu_rate").mean(),
        )
        .sort("year")
    )
    first_year = yearly.head(1).to_dicts()[0]
    last_year = yearly.tail(1).to_dicts()[0]

    scope = (
        "所有语言"
        if args.all_languages
        else COMMON_LANGUAGES.get(args.lang, args.lang)
    )

    return f"""---
tags:
  - uv-script
  - 统计数据
  - polars
  - finepdfs-edu
  - 时间分析
license: odc-by
configs:
  - config_name: global_stats
    data_files: global_stats/train-*.parquet
  - config_name: temporal_stats
    data_files: temporal_stats/train-*.parquet
default_viewer_config: temporal_stats
---

# 网络是否变得越来越有教育性?

对 **{scope}** 在 {stats.get("num_dumps", 0)} 个 CommonCrawl 数据 dump 中的教育质量进行时间分析。

## 趋势

```
{ascii_charts}
```

## 主要发现

| 年份 | 平均教育分数 | 高教育比率 |
|------|-------------|-----------|
| {first_year["year"]} | {first_year["avg_edu_score"]:.2f} | {first_year["high_edu_rate"] * 100:.1f}% |
| {last_year["year"]} | {last_year["avg_edu_score"]:.2f} | {last_year["high_edu_rate"] * 100:.1f}% |

## 性能

- **{total_docs:,} 个文档** 在 **{scan_time:.0f} 秒**内处理完成
- 使用 Polars 流式处理达到 **{docs_per_sec:,.0f} 文档/秒**
- 单次扫描,无需完整下载数据集

## 摘要

| 指标 | 值 |
|------|-----|
| 范围 | {scope} |
| 文档总数 | {total_docs:,} |
| Token 总数 | {stats.get("total_tokens", 0):,} |
| 平均教育分数 | {stats.get("avg_edu_score", 0):.3f} |
| 高教育比率 | {stats.get("high_edu_rate", 0) * 100:.1f}% |
| CommonCrawl Dump 数 | {stats.get("num_dumps", 0)} |

## 文件

- `global_stats` - 总体摘要
- `temporal_stats` - 每个 dump 的分解 (按时间顺序排序)

## 复现

```bash
uv run https://huggingface.co/datasets/uv-scripts/dataset-stats/raw/main/finepdfs-stats.py \\
    {"--all-languages" if args.all_languages else f"--lang {args.lang}"} --output-repo your-username/stats
```

## 来源

- **数据集**: [{args.source_dataset}](https://huggingface.co/datasets/{args.source_dataset})
- **脚本**: [uv-scripts/dataset-stats](https://huggingface.co/datasets/uv-scripts/dataset-stats)
"""


def main():
    parser = argparse.ArgumentParser(
        description="分析 CommonCrawl 数据 dump 中的教育质量趋势",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--source-dataset",
        type=str,
        default="HuggingFaceFW/finepdfs-edu",
        help="源数据集 (默认: HuggingFaceFW/finepdfs-edu)",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="eng_Latn",
        help="语言+文字代码 (默认: eng_Latn)",
    )

    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="分析所有语言 (70+) 而不是单一语言",
    )

    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="显示 Polars 查询计划 (演示优化)",
    )

    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="列出可用语言并退出",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="限制前 N 行 (用于测试)",
    )

    parser.add_argument(
        "--output-repo",
        type=str,
        help="用于上传结果的 HuggingFace 数据集仓库",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./stats_output",
        help="输出文件的本地目录",
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace API 令牌 (或设置 HF_TOKEN 环境变量)",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="使输出数据集私有",
    )

    args = parser.parse_args()

    # 检查高性能模式
    if os.environ.get("HF_XET_HIGH_PERFORMANCE"):
        logger.info("高性能模式已启用 (HF_XET_HIGH_PERFORMANCE=1)")

    # 列出语言模式
    if args.list_languages:
        print(f"{args.source_dataset} 可用的语言+文字代码:\n")
        print("常用语言:")
        for code, name in COMMON_LANGUAGES.items():
            print(f"  {code:12} - {name}")
        print("\n从 HF Hub 获取完整列表...")
        all_langs = list_available_languages(args.source_dataset)
        print(f"\n所有可用的语言 (共 {len(all_langs)} 种):")
        for lang in all_langs[:30]:  # 显示前 30 种
            name = COMMON_LANGUAGES.get(lang, "")
            print(f"  {lang:12} {name}")
        if len(all_langs) > 30:
            print(f"  ... 还有 {len(all_langs) - 30} 种")
        sys.exit(0)

    # 构建 parquet 路径
    if args.all_languages:
        source_path = f"hf://datasets/{args.source_dataset}/data/*/train/*.parquet"
        scope_desc = "所有语言"
    else:
        source_path = (
            f"hf://datasets/{args.source_dataset}/data/{args.lang}/train/*.parquet"
        )
        scope_desc = f"{args.lang} ({COMMON_LANGUAGES.get(args.lang, 'unknown')})"

    logger.info(f"扫描路径: {source_path}")
    logger.info(f"范围: {scope_desc}")

    # 创建惰性框架 - 此时不加载任何数据!
    logger.info("正在创建惰性查询计划...")
    df = pl.scan_parquet(source_path)

    # 如果指定了 limit 则应用
    if args.limit:
        logger.info(f"限制为前 {args.limit:,} 行")
        df = df.head(args.limit)

    # 如果请求则显示查询计划
    if args.show_plan:
        # 构建示例查询以显示计划
        sample_query = df.select(
            pl.len(),
            pl.col("token_count").sum(),
            pl.col("language").n_unique(),
        )
        print("\n查询计划 (展示 Polars 优化):")
        print("=" * 60)
        print(sample_query.explain())
        print("=" * 60)
        print("\n注意: Polars 使用投影下推 - 只读取需要的列!")
        print("永远不会加载 'text' 列,这使得处理非常快。\n")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 单次扫描:计算时间统计
    logger.info("正在计算时间统计 (单次扫描)...")
    start = time.perf_counter()
    temporal_path = output_dir / "temporal_stats.parquet"
    temporal_raw = compute_temporal_stats(df, temporal_path)
    scan_time = time.perf_counter() - start
    logger.info(f"扫描完成,耗时 {scan_time:.2f}s - {len(temporal_raw)} 个 dump")

    # 计算统计
    global_stats = compute_global_stats(temporal_raw)
    temporal_stats = format_temporal_stats(temporal_raw)

    # 保存
    global_stats.write_parquet(output_dir / "global_stats.parquet")
    temporal_stats.write_parquet(output_dir / "temporal_stats.parquet")

    # 打印结果
    total_docs = global_stats["total_docs"][0]
    docs_per_sec = total_docs / scan_time if scan_time > 0 else 0

    print("\n" + "=" * 70)
    print("网络是否变得越来越有教育性?")
    print("=" * 70)

    print(f"\n范围: {scope_desc}")
    print(f"数据集: {args.source_dataset}")

    print("\n" + "-" * 70)
    print("全局统计")
    print("-" * 70)
    print(global_stats)

    print("\n" + "-" * 70)
    print(f"时间趋势 ({len(temporal_stats)} 个 CommonCrawl dump)")
    print("-" * 70)
    # 显示前 5 个和后 5 个
    if len(temporal_stats) > 10:
        print最早的 dump:")
        print(temporal_stats.head(5))
        print("\n...")
        print("\n最新的 dump:")
        print(temporal_stats.tail(5))
    else:
        print(temporal_stats)

    # 创建 ASCII 图表
    ascii_charts = create_ascii_charts(temporal_stats)
    print("\n" + "-" * 70)
    print("趋势可视化")
    print("-" * 70)
    print(ascii_charts)

    print("\n" + "-" * 70)
    print("性能")
    print("-" * 70)
    print(f"扫描时间: {scan_time:.2f}s")
    print(f"文档数: {total_docs:,}")
    print(f"吞吐量: {docs_per_sec:,.0f} 文档/秒")

    logger.info(f"结果已保存至: {output_dir}")

    # 如果请求则上传到 HF Hub
    if args.output_repo:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        api = HfApi(token=hf_token)

        logger.info(f"创建/更新数据集仓库: {args.output_repo}")
        create_repo(
            args.output_repo,
            repo_type="dataset",
            private=args.private,
            token=hf_token,
            exist_ok=True,
        )

        # 上传每个配置
        configs = [
            ("global_stats", global_stats),
            ("temporal_stats", temporal_stats),
        ]

        for config_name, stats_df in configs:
            logger.info(f"正在上传 {config_name}...")
            ds = Dataset.from_polars(stats_df)
            ds.push_to_hub(
                args.output_repo,
                config_name=config_name,
                token=hf_token,
                private=args.private,
            )
            time.sleep(1)  # 避免 409 冲突

        # 上传 README
        readme_content = create_readme(
            args, global_stats, temporal_stats, scan_time, ascii_charts
        )
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=args.output_repo,
            repo_type="dataset",
            token=hf_token,
        )

        dataset_url = f"https://huggingface.co/datasets/{args.output_repo}"
        logger.info(f"数据集已上传: {dataset_url}")
        print(f"\n结果已上传至: {dataset_url}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("网络是否变得越来越有教育性?")
        print("=" * 40)
        print("\n使用 Polars 流式分析 CommonCrawl 数据 dump 中的教育质量趋势")
        print("无需下载数据!\n")
        print("示例命令:\n")
        print("# 快速测试:")
        print("uv run finepdfs-stats.py --limit 10000\n")
        print("# 分析英文 PDF:")
        print("uv run finepdfs-stats.py\n")
        print("# 分析所有 70+ 语言:")
        print("uv run finepdfs-stats.py --all-languages\n")
        print("# 显示查询计划 (查看 Polars 优化):")
        print("uv run finepdfs-stats.py --show-plan --limit 1000\n")
        print("# 保存结果到 HF Hub:")
        print("uv run finepdfs-stats.py --output-repo username/temporal-stats\n")
        print("# 在 HF Jobs 上运行:")
        print("hf jobs uv run \\")
        print("    -s HF_TOKEN \\")
        print("    -e HF_XET_HIGH_PERFORMANCE=1 \\")
        print(
            "    https://huggingface.co/datasets/uv-scripts/dataset-stats/raw/main/finepdfs-stats.py \\"
        )
        print("    -- --output-repo username/stats")
        sys.exit(0)

    main()
