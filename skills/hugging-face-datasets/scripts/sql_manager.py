#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "duckdb>=1.0.0",
#   "huggingface_hub>=0.20.0",
#   "datasets>=2.14.0",
#   "pandas>=2.0.0",
# ]
# ///
"""
Hugging Face 数据集 SQL 管理器

使用 DuckDB 的 SQL 接口查询、转换和推送 Hugging Face 数据集。
支持 hf:// 协议进行直接数据集访问、数据处理和将结果推送回 Hub。

版本: 1.0.0

使用方法:
    # 查询数据集
    uv run sql_manager.py query --dataset "cais/mmlu" --sql "SELECT * FROM data LIMIT 10"
    
    # 查询并推送到新数据集
    uv run sql_manager.py query --dataset "cais/mmlu" --sql "SELECT * FROM data WHERE subject='nutrition'" \
        --push-to "username/nutrition-subset"
    
    # 描述数据集架构
    uv run sql_manager.py describe --dataset "cais/mmlu"
    
    # 列出可用的分割/配置
    uv run sql_manager.py info --dataset "cais/mmlu"
    
    # 获取随机样本
    uv run sql_manager.py sample --dataset "cais/mmlu" --n 5
    
    # 导出到 parquet
    uv run sql_manager.py export --dataset "cais/mmlu" --output "data.parquet"
"""


import os
import json
import argparse
from typing import Optional, List, Dict, Any, Union

import duckdb
from huggingface_hub import HfApi


# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")


class HFDatasetSQL:
    """
    使用 DuckDB SQL 查询 Hugging Face 数据集。

    示例:
        >>> sql = HFDatasetSQL()
        >>> results = sql.query("cais/mmlu", "SELECT * FROM data LIMIT 5")
        >>> schema = sql.describe("cais/mmlu")
        >>> sql.query_and_push("cais/mmlu", "SELECT * FROM data WHERE subject='nutrition'", "user/nutrition-qa")
    """

    def __init__(self, token: Optional[str] = None):
        """使用可选的 HF token 初始化 SQL 管理器。"""
        self.token = token or HF_TOKEN
        self.conn = duckdb.connect()
        self._setup_connection()

    def _setup_connection(self):
        """配置 DuckDB 连接以访问 HF。"""
        # 如果可用，设置 HF token（用于私有数据集）
        if self.token:
            self.conn.execute(f"CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN '{self.token}');")

    def _build_hf_path(
        self, dataset_id: str, split: str = "*", config: Optional[str] = None, revision: str = "~parquet"
    ) -> str:
        """
        构建数据集的 hf:// 路径。

        参数:
            dataset_id: 数据集 ID（例如，"cais/mmlu"）
            split: 分割名称或 "*" 表示所有分割
            config: 可选的配置/子集名称
            revision: 修订版本，默认为 ~parquet 以获取自动转换的 parquet

        返回:
            hf:// 路径字符串
        """
        if config:
            return f"hf://datasets/{dataset_id}@{revision}/{config}/{split}/*.parquet"
        else:
            return f"hf://datasets/{dataset_id}@{revision}/default/{split}/*.parquet"

    def _build_hf_path_flexible(
        self,
        dataset_id: str,
        split: Optional[str] = None,
        config: Optional[str] = None,
    ) -> str:
        """
        构建带有通配符的灵活 hf:// 路径以进行发现。

        参数:
            dataset_id: 数据集 ID
            split: 可选的特定分割
            config: 可选的配置名称

        返回:
            带有适当通配符的 hf:// 路径
        """
        base = f"hf://datasets/{dataset_id}@~parquet"

        if config and split:
            return f"{base}/{config}/{split}/*.parquet"
        elif config:
            return f"{base}/{config}/*/*.parquet"
        elif split:
            return f"{base}/*/{split}/*.parquet"
        else:
            return f"{base}/*/*/*.parquet"

    def query(
        self,
        dataset_id: str,
        sql: str,
        split: str = "train",
        config: Optional[str] = None,
        limit: Optional[int] = None,
        output_format: str = "dict",
    ) -> Union[List[Dict], Any]:
        """
        在 Hugging Face 数据集上执行 SQL 查询。

        参数:
            dataset_id: 数据集 ID（例如，"cais/mmlu"，"ibm/duorc"）
            sql: SQL 查询。使用 'data' 作为表名（将被替换为实际路径）
            split: 数据集分割（train、test、validation 或 * 表示所有）
            config: 可选的数据集配置/子集
            limit: 可选的限制覆盖
            output_format: 输出格式 - "dict"、"df"（pandas）、"arrow"、"raw"

        返回:
            指定格式的查询结果

        示例:
            >>> sql.query("cais/mmlu", "SELECT * FROM data WHERE subject='nutrition' LIMIT 10")
            >>> sql.query("cais/mmlu", "SELECT subject, COUNT(*) as cnt FROM data GROUP BY subject")
        """

        # Build the HF path
        hf_path = self._build_hf_path(dataset_id, split=split, config=config)

        # Replace 'data' placeholder with actual path
        # Handle various SQL patterns
        processed_sql = sql.replace("FROM data", f"FROM '{hf_path}'")
        processed_sql = processed_sql.replace("from data", f"FROM '{hf_path}'")
        processed_sql = processed_sql.replace("JOIN data", f"JOIN '{hf_path}'")
        processed_sql = processed_sql.replace("join data", f"JOIN '{hf_path}'")

        # If user provides raw path, use as-is
        if "hf://" in sql:
            processed_sql = sql

        # Apply limit if specified and not already in query
        if limit and "LIMIT" not in processed_sql.upper():
            processed_sql += f" LIMIT {limit}"

        try:
            result = self.conn.execute(processed_sql)

            if output_format == "df":
                return result.fetchdf()
            elif output_format == "arrow":
                return result.fetch_arrow_table()
            elif output_format == "raw":
                return result.fetchall()
            else:  # dict
                columns = [desc[0] for desc in result.description]
                rows = result.fetchall()
                return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            print(f"❌ Query error: {e}")
            print(f"   SQL: {processed_sql[:200]}...")
            raise

    def query_raw(self, sql: str, output_format: str = "dict") -> Union[List[Dict], Any]:
        """
        执行原始 SQL 查询，不进行路径替换。

        适用于已经包含完整 hf:// 路径的查询或多数据集查询。

        参数:
            sql: 完整的 SQL 查询
            output_format: 输出格式

        返回:
            查询结果
        """

        result = self.conn.execute(sql)

        if output_format == "df":
            return result.fetchdf()
        elif output_format == "arrow":
            return result.fetch_arrow_table()
        elif output_format == "raw":
            return result.fetchall()
        else:
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]

    def describe(self, dataset_id: str, split: str = "train", config: Optional[str] = None) -> List[Dict[str, str]]:
        """
        获取数据集的架构/结构。

        参数:
            dataset_id: 数据集 ID
            split: 数据集分割
            config: 可选配置

        返回:
            带有名称、类型、可空信息的列定义列表
        """

        hf_path = self._build_hf_path(dataset_id, split=split, config=config)

        sql = f"DESCRIBE SELECT * FROM '{hf_path}' LIMIT 1"
        result = self.conn.execute(sql)

        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    def sample(
        self,
        dataset_id: str,
        n: int = 10,
        split: str = "train",
        config: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """
        从数据集获取随机样本。

        参数:
            dataset_id: 数据集 ID
            n: 样本数量
            split: 数据集分割
            config: 可选配置
            seed: 用于可重现性的随机种子

        返回:
            采样行的列表
        """

        hf_path = self._build_hf_path(dataset_id, split=split, config=config)

        if seed is not None:
            sql = f"SELECT * FROM '{hf_path}' USING SAMPLE {n} (RESERVOIR, {seed})"
        else:
            sql = f"SELECT * FROM '{hf_path}' USING SAMPLE {n}"

        return self.query_raw(sql)

    def count(
        self, dataset_id: str, split: str = "train", config: Optional[str] = None, where: Optional[str] = None
    ) -> int:
        """
        计算数据集中的行数，可选带过滤器。

        参数:
            dataset_id: 数据集 ID
            split: 数据集分割
            config: 可选配置
            where: 可选的 WHERE 子句（不含 WHERE 关键字）

        返回:
            行数
        """

        hf_path = self._build_hf_path(dataset_id, split=split, config=config)

        sql = f"SELECT COUNT(*) FROM '{hf_path}'"
        if where:
            sql += f" WHERE {where}"

        result = self.conn.execute(sql).fetchone()
        return result[0] if result else 0

    def unique_values(
        self, dataset_id: str, column: str, split: str = "train", config: Optional[str] = None, limit: int = 100
    ) -> List[Any]:
        """
        获取列中的唯一值。

        参数:
            dataset_id: 数据集 ID
            column: 列名
            split: 数据集分割
            config: 可选配置
            limit: 要返回的最大唯一值数量

        返回:
            唯一值列表
        """

        hf_path = self._build_hf_path(dataset_id, split=split, config=config)

        sql = f"SELECT DISTINCT {column} FROM '{hf_path}' LIMIT {limit}"
        result = self.conn.execute(sql).fetchall()

        return [row[0] for row in result]

    def histogram(
        self, dataset_id: str, column: str, split: str = "train", config: Optional[str] = None, bins: int = 10
    ) -> List[Dict]:
        """
        获取列的值分布/直方图。

        参数:
            dataset_id: 数据集 ID
            column: 列名
            split: 数据集分割
            config: 可选配置
            bins: 数值列的箱数

        返回:
            分布数据
        """

        hf_path = self._build_hf_path(dataset_id, split=split, config=config)

        sql = f"""
        SELECT 
            {column},
            COUNT(*) as count
        FROM '{hf_path}'
        GROUP BY {column}
        ORDER BY count DESC
        LIMIT {bins}
        """

        return self.query_raw(sql)

    def filter_and_transform(
        self,
        dataset_id: str,
        select: str = "*",
        where: Optional[str] = None,
        group_by: Optional[str] = None,
        order_by: Optional[str] = None,
        split: str = "train",
        config: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """
        使用 SQL 子句过滤和转换数据集。

        参数:
            dataset_id: 数据集 ID
            select: SELECT 子句（列、表达式、聚合）
            where: WHERE 子句（过滤条件）
            group_by: GROUP BY 子句
            order_by: ORDER BY 子句
            split: 数据集分割
            config: 可选配置
            limit: 行限制

        返回:
            转换后的数据

        示例:
            >>> sql.filter_and_transform(
            ...     "cais/mmlu",
            ...     select="subject, COUNT(*) as cnt",
            ...     group_by="subject",
            ...     order_by="cnt DESC",
            ...     limit=10
            ... )
        """

        hf_path = self._build_hf_path(dataset_id, split=split, config=config)

        sql_parts = [f"SELECT {select}", f"FROM '{hf_path}'"]

        if where:
            sql_parts.append(f"WHERE {where}")
        if group_by:
            sql_parts.append(f"GROUP BY {group_by}")
        if order_by:
            sql_parts.append(f"ORDER BY {order_by}")
        if limit:
            sql_parts.append(f"LIMIT {limit}")

        sql = " ".join(sql_parts)
        return self.query_raw(sql)

    def join_datasets(
        self,
        left_dataset: str,
        right_dataset: str,
        on: str,
        select: str = "*",
        join_type: str = "INNER",
        left_split: str = "train",
        right_split: str = "train",
        left_config: Optional[str] = None,
        right_config: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """
        连接两个数据集。

        参数:
            left_dataset: 左侧数据集 ID
            right_dataset: 右侧数据集 ID
            on: 连接条件（例如，"left.id = right.id"）
            select: SELECT 子句
            join_type: 连接类型（INNER、LEFT、RIGHT、FULL）
            left_split: 左侧数据集的分割
            right_split: 右侧数据集的分割
            left_config: 左侧数据集的配置
            right_config: 右侧数据集的配置
            limit: 行限制

        返回:
            连接后的数据
        """

        left_path = self._build_hf_path(left_dataset, split=left_split, config=left_config)
        right_path = self._build_hf_path(right_dataset, split=right_split, config=right_config)

        sql = f"""
        SELECT {select}
        FROM '{left_path}' AS left_table
        {join_type} JOIN '{right_path}' AS right_table
        ON {on}
        """

        if limit:
            sql += f" LIMIT {limit}"

        return self.query_raw(sql)

    def export_to_parquet(
        self,
        dataset_id: str,
        output_path: str,
        sql: Optional[str] = None,
        split: str = "train",
        config: Optional[str] = None,
    ) -> str:
        """
        将查询结果导出到本地 Parquet 文件。

        参数:
            dataset_id: 源数据集 ID
            output_path: 输出 Parquet 文件的本地路径
            sql: 可选的 SQL 查询（如果未提供则使用 SELECT *）
            split: 数据集分割
            config: 可选配置

        返回:
            创建文件的路径
        """

        hf_path = self._build_hf_path(dataset_id, split=split, config=config)

        if sql:
            # Process the query
            processed_sql = sql.replace("FROM data", f"FROM '{hf_path}'")
            processed_sql = processed_sql.replace("from data", f"FROM '{hf_path}'")
        else:
            processed_sql = f"SELECT * FROM '{hf_path}'"

        export_sql = f"COPY ({processed_sql}) TO '{output_path}' (FORMAT PARQUET)"
        self.conn.execute(export_sql)

        print(f"✅ Exported to {output_path}")
        return output_path

    def export_to_jsonl(
        self,
        dataset_id: str,
        output_path: str,
        sql: Optional[str] = None,
        split: str = "train",
        config: Optional[str] = None,
    ) -> str:
        """
        将查询结果导出到 JSONL 格式。

        参数:
            dataset_id: 源数据集 ID
            output_path: 输出 JSONL 文件的本地路径
            sql: 可选的 SQL 查询
            split: 数据集分割
            config: 可选配置

        返回:
            创建文件的路径
        """

        results = self.query(dataset_id, sql or "SELECT * FROM data", split=split, config=config)

        with open(output_path, "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

        print(f"✅ Exported {len(results)} rows to {output_path}")
        return output_path

    def push_to_hub(
        self,
        dataset_id: str,
        target_repo: str,
        sql: Optional[str] = None,
        split: str = "train",
        config: Optional[str] = None,
        target_split: str = "train",
        private: bool = True,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        查询数据集并将结果推送到新的 Hub 仓库。

        参数:
            dataset_id: 源数据集 ID
            target_repo: 目标仓库 ID（例如，"username/new-dataset"）
            sql: 转换数据的 SQL 查询（可选，默认为 SELECT *）
            split: 源分割
            config: 源配置
            target_split: 目标分割名称
            private: 是否创建私有仓库
            commit_message: 提交消息

        返回:
            创建的数据集的 URL
        """

        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets library required for push_to_hub. Install with: pip install datasets")

        # Execute query
        results = self.query(dataset_id, sql or "SELECT * FROM data", split=split, config=config)

        if not results:
            print("❌ No results to push")
            return ""

        # Convert to HF Dataset
        ds = Dataset.from_list(results)

        # Push to Hub
        ds.push_to_hub(
            target_repo,
            split=target_split,
            private=private,
            commit_message=commit_message or f"Created from {dataset_id} via SQL query",
            token=self.token,
        )

        url = f"https://huggingface.co/datasets/{target_repo}"
        print(f"✅ Pushed {len(results)} rows to {url}")
        return url

    def create_view(self, name: str, dataset_id: str, split: str = "train", config: Optional[str] = None):
        """
        创建 DuckDB 视图以简化查询。

        参数:
            name: 视图名称
            dataset_id: 数据集 ID
            split: 数据集分割
            config: 可选配置
        """

        hf_path = self._build_hf_path(dataset_id, split=split, config=config)
        self.conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM '{hf_path}'")
        print(f"✅ Created view '{name}' for {dataset_id}")

    def info(self, dataset_id: str) -> Dict[str, Any]:
        """
        获取关于数据集的信息，包括可用的配置和分割。

        参数:
            dataset_id: 数据集 ID

        返回:
            数据集信息
        """

        api = HfApi(token=self.token)

        try:
            info = api.dataset_info(dataset_id)

            result = {
                "id": info.id,
                "author": info.author,
                "private": info.private,
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags,
                "created_at": str(info.created_at) if info.created_at else None,
                "last_modified": str(info.last_modified) if info.last_modified else None,
            }

            # Try to get config/split info from card data
            if info.card_data:
                result["configs"] = getattr(info.card_data, "configs", None)

            return result

        except Exception as e:
            print(f"❌ Failed to get info: {e}")
            return {}

    def close(self):
        """关闭数据库连接。"""

        self.conn.close()


def main():
    """命令行入口点。"""

    parser = argparse.ArgumentParser(
        description="使用 SQL 查询 Hugging Face 数据集",

        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query dataset with SQL
  python sql_manager.py query --dataset "cais/mmlu" --sql "SELECT * FROM data WHERE subject='nutrition' LIMIT 10"
  
  # Get random sample
  python sql_manager.py sample --dataset "cais/mmlu" --n 5
  
  # Describe schema
  python sql_manager.py describe --dataset "cais/mmlu"
  
  # Get value counts
  python sql_manager.py histogram --dataset "cais/mmlu" --column "subject"
  
  # Filter and transform
  python sql_manager.py transform --dataset "cais/mmlu" \\
    --select "subject, COUNT(*) as cnt" \\
    --group-by "subject" \\
    --order-by "cnt DESC"
  
  # Query and push to Hub
  python sql_manager.py query --dataset "cais/mmlu" \\
    --sql "SELECT * FROM data WHERE subject='nutrition'" \\
    --push-to "username/nutrition-subset"
  
  # Export to Parquet
  python sql_manager.py export --dataset "cais/mmlu" \\
    --sql "SELECT * FROM data WHERE subject='nutrition'" \\
    --output "nutrition.parquet"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 通用参数
    def add_common_args(p):
        p.add_argument("--dataset", "-d", required=True, help="数据集 ID（例如，cais/mmlu）")
        p.add_argument("--split", "-s", default="train", help="数据集分割（默认：train）")
        p.add_argument("--config", "-c", help="数据集配置/子集")


    # 查询命令
    query_parser = subparsers.add_parser("query", help="在数据集上执行 SQL 查询")
    add_common_args(query_parser)
    query_parser.add_argument("--sql", required=True, help="SQL 查询（使用 'data' 作为表名）")
    query_parser.add_argument("--limit", "-l", type=int, help="限制结果")
    query_parser.add_argument("--format", choices=["json", "table", "csv"], default="json", help="输出格式")
    query_parser.add_argument("--push-to", help="将结果推送到此 Hub 仓库")
    query_parser.add_argument("--private", action="store_true", help="使推送的仓库变为私有")


    # 样本命令
    sample_parser = subparsers.add_parser("sample", help="从数据集获取随机样本")
    add_common_args(sample_parser)
    sample_parser.add_argument("--n", type=int, default=10, help="样本数量")
    sample_parser.add_argument("--seed", type=int, help="随机种子")


    # 描述命令
    describe_parser = subparsers.add_parser("describe", help="获取数据集架构")
    add_common_args(describe_parser)


    # 计数命令
    count_parser = subparsers.add_parser("count", help="计算数据集中的行数")
    add_common_args(count_parser)
    count_parser.add_argument("--where", "-w", help="用于过滤的 WHERE 子句")


    # 直方图命令
    histogram_parser = subparsers.add_parser("histogram", help="获取值分布")
    add_common_args(histogram_parser)
    histogram_parser.add_argument("--column", required=True, help="列名")
    histogram_parser.add_argument("--bins", type=int, default=20, help="箱数")


    # 唯一值命令
    unique_parser = subparsers.add_parser("unique", help="获取列中的唯一值")
    add_common_args(unique_parser)
    unique_parser.add_argument("--column", required=True, help="列名")
    unique_parser.add_argument("--limit", "-l", type=int, default=100, help="最大值数量")


    # 转换命令
    transform_parser = subparsers.add_parser("transform", help="过滤和转换数据集")
    add_common_args(transform_parser)
    transform_parser.add_argument("--select", default="*", help="SELECT 子句")
    transform_parser.add_argument("--where", "-w", help="WHERE 子句")
    transform_parser.add_argument("--group-by", help="GROUP BY 子句")
    transform_parser.add_argument("--order-by", help="ORDER BY 子句")
    transform_parser.add_argument("--limit", "-l", type=int, help="LIMIT")
    transform_parser.add_argument("--push-to", help="将结果推送到 Hub 仓库")


    # 导出命令
    export_parser = subparsers.add_parser("export", help="将查询结果导出到文件")
    add_common_args(export_parser)
    export_parser.add_argument("--sql", help="SQL 查询（默认为 SELECT *）")
    export_parser.add_argument("--output", "-o", required=True, help="输出文件路径")
    export_parser.add_argument("--format", choices=["parquet", "jsonl"], default="parquet", help="输出格式")


    # 信息命令
    info_parser = subparsers.add_parser("info", help="获取数据集信息")
    info_parser.add_argument("--dataset", "-d", required=True, help="数据集 ID")


    # 原始 SQL 命令
    raw_parser = subparsers.add_parser("raw", help="执行带有完整 hf:// 路径的原始 SQL")
    raw_parser.add_argument("--sql", required=True, help="完整的 SQL 查询")
    raw_parser.add_argument("--format", choices=["json", "table", "csv"], default="json", help="输出格式")


    args = parser.parse_args()

    # Initialize SQL manager
    sql = HFDatasetSQL()

    try:
        if args.command == "query":
            results = sql.query(args.dataset, args.sql, split=args.split, config=args.config, limit=args.limit)

            if getattr(args, "push_to", None):
                sql.push_to_hub(
                    args.dataset, args.push_to, sql=args.sql, split=args.split, config=args.config, private=args.private
                )
            else:
                _print_results(results, args.format)

        elif args.command == "sample":
            results = sql.sample(args.dataset, n=args.n, split=args.split, config=args.config, seed=args.seed)
            _print_results(results, "json")

        elif args.command == "describe":
            schema = sql.describe(args.dataset, split=args.split, config=args.config)
            _print_results(schema, "table")

        elif args.command == "count":
            count = sql.count(args.dataset, split=args.split, config=args.config, where=args.where)
            print(f"Count: {count:,}")

        elif args.command == "histogram":
            results = sql.histogram(args.dataset, args.column, split=args.split, config=args.config, bins=args.bins)
            _print_results(results, "table")

        elif args.command == "unique":
            values = sql.unique_values(
                args.dataset, args.column, split=args.split, config=args.config, limit=args.limit
            )
            for v in values:
                print(v)

        elif args.command == "transform":
            results = sql.filter_and_transform(
                args.dataset,
                select=args.select,
                where=args.where,
                group_by=args.group_by,
                order_by=args.order_by,
                split=args.split,
                config=args.config,
                limit=args.limit,
            )

            if getattr(args, "push_to", None):
                # Build SQL for push
                query_sql = f"SELECT {args.select} FROM data"
                if args.where:
                    query_sql += f" WHERE {args.where}"
                if args.group_by:
                    query_sql += f" GROUP BY {args.group_by}"
                if args.order_by:
                    query_sql += f" ORDER BY {args.order_by}"
                if args.limit:
                    query_sql += f" LIMIT {args.limit}"

                sql.push_to_hub(args.dataset, args.push_to, sql=query_sql, split=args.split, config=args.config)
            else:
                _print_results(results, "json")

        elif args.command == "export":
            if args.format == "parquet":
                sql.export_to_parquet(args.dataset, args.output, sql=args.sql, split=args.split, config=args.config)
            else:
                sql.export_to_jsonl(args.dataset, args.output, sql=args.sql, split=args.split, config=args.config)

        elif args.command == "info":
            info = sql.info(args.dataset)
            _print_results([info], "json")

        elif args.command == "raw":
            results = sql.query_raw(args.sql)
            _print_results(results, args.format)

    finally:
        sql.close()


def _print_results(results: List[Dict], format: str):
    """以指定格式打印结果。"""

    if not results:
        print("无结果")
        return


    if format == "json":
        print(json.dumps(results, indent=2, default=str))

    elif format == "csv":
        if results:
            keys = results[0].keys()
            print(",".join(str(k) for k in keys))
            for row in results:
                print(",".join(str(row.get(k, "")) for k in keys))

    elif format == "table":
        if results:
            keys = list(results[0].keys())
            # Calculate column widths
            widths = {k: max(len(str(k)), max(len(str(r.get(k, ""))) for r in results)) for k in keys}

            # Header
            header = " | ".join(str(k).ljust(widths[k]) for k in keys)
            print(header)
            print("-" * len(header))

            # Rows
            for row in results:
                print(" | ".join(str(row.get(k, "")).ljust(widths[k]) for k in keys))


if __name__ == "__main__":
    main()
