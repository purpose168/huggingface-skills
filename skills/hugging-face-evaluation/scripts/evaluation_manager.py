# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "huggingface-hub>=1.1.4",
#     "markdown-it-py>=3.0.0",
#     "python-dotenv>=1.2.1",
#     "pyyaml>=6.0.3",
#     "requests>=2.32.5",
# ]
# ///

"""
管理 Hugging Face 模型卡片中的评估结果。

本脚本提供两种方法：
1. 从模型 README 文件中提取评估表格
2. 从 Artificial Analysis API 导入评估分数

两种方法都会更新模型卡片中的 model-index 元数据。
"""


import argparse
import os
import re
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple


def load_env() -> None:
    """如果 python-dotenv 可用，则加载 .env 文件；在没有它的情况下保持帮助可用。"""
    try:
        import dotenv  # type: ignore
    except ModuleNotFoundError:
        return
    dotenv.load_dotenv()



def require_markdown_it():
    try:
        from markdown_it import MarkdownIt  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "表格解析需要 markdown-it-py。 "
            "使用 `uv add markdown-it-py` 或 `pip install markdown-it-py` 安装。"
        ) from exc
    return MarkdownIt



def require_model_card():
    try:
        from huggingface_hub import ModelCard  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "模型卡片操作需要 huggingface-hub。 "
            "使用 `uv add huggingface_hub` 或 `pip install huggingface-hub` 安装。"
        ) from exc
    return ModelCard



def require_requests():
    try:
        import requests  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Artificial Analysis 导入需要 requests。 "
            "使用 `uv add requests` 或 `pip install requests` 安装。"
        ) from exc
    return requests



def require_yaml():
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "YAML 输出需要 PyYAML。 "
            "使用 `uv add pyyaml` 或 `pip install pyyaml` 安装。"
        ) from exc
    return yaml



# ============================================================================
# Method 1: Extract Evaluations from README
# ============================================================================


def extract_tables_from_markdown(markdown_content: str) -> List[str]:
    """从内容中提取所有 markdown 表格。"""
    # 匹配 markdown 表格的模式
    table_pattern = r"(\|[^\n]+\|(?:\r?\n\|[^\n]+\|)+)"
    tables = re.findall(table_pattern, markdown_content)
    return tables



def parse_markdown_table(table_str: str) -> Tuple[List[str], List[List[str]]]:
    """
    将 markdown 表格字符串解析为表头和行数据。

    返回:
        (表头, 数据行) 的元组
    """
    lines = [line.strip() for line in table_str.strip().split("\n")]

    # 删除分隔线（带破折号的行）
    lines = [line for line in lines if not re.match(r"^\|[\s\-:]+\|$", line)]

    if len(lines) < 2:
        return [], []

    # 解析表头
    header = [cell.strip() for cell in lines[0].split("|")[1:-1]]

    # 解析数据行
    data_rows = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if cells:
            data_rows.append(cells)

    return header, data_rows



def is_evaluation_table(header: List[str], rows: List[List[str]]) -> bool:
    """判断表格是否包含评估结果。"""
    if not header or not rows:
        return False

    # 检查第一列是否看起来像基准测试名称
    benchmark_keywords = [
        "benchmark", "task", "dataset", "eval", "test", "metric",
        "mmlu", "humaneval", "gsm", "hellaswag", "arc", "winogrande",
        "truthfulqa", "boolq", "piqa", "siqa"
    ]

    first_col = header[0].lower()
    has_benchmark_header = any(keyword in first_col for keyword in benchmark_keywords)

    # 检查表格中是否有数值
    has_numeric_values = False
    for row in rows:
        for cell in row:
            try:
                float(cell.replace("%", "").replace(",", ""))
                has_numeric_values = True
                break
            except ValueError:
                continue
        if has_numeric_values:
            break

    return has_benchmark_header or has_numeric_values



def normalize_model_name(name: str) -> tuple[set[str], str]:
    """
    规范化模型名称以进行匹配。

    参数:
        name: 要规范化的模型名称

    返回:
        (token_set, normalized_string) 的元组
    """
    # 移除 markdown 格式
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', name)  # 移除 markdown 链接
    cleaned = re.sub(r'\*\*([^\*]+)\*\*', r'\1', cleaned)  # 移除粗体
    cleaned = cleaned.strip()

    # 规范化并分词
    normalized = cleaned.lower().replace("-", " ").replace("_", " ")
    tokens = set(normalized.split())

    return tokens, normalized



def find_main_model_column(header: List[str], model_name: str) -> Optional[int]:
    """
    识别与主模型对应的列索引。

    只有当与模型名称存在精确的规范化匹配时才返回列。
    这样可以防止从训练检查点或类似模型中提取分数。

    参数:
        header: 表格列标题
        model_name: 来自 repo_id 的模型名称（例如，"OLMo-3-32B-Think"）

    返回:
        主模型的列索引，如果未找到精确匹配则返回 None
    """
    if not header or not model_name:
        return None

    # 规范化模型名称并提取令牌
    model_tokens, _ = normalize_model_name(model_name)

    # 只查找精确匹配
    for i, col_name in enumerate(header):
        if not col_name:
            continue

        # 跳过第一列（基准测试名称）
        if i == 0:
            continue

        col_tokens, _ = normalize_model_name(col_name)

        # 检查精确令牌匹配
        if model_tokens == col_tokens:
            return i

    # 未找到精确匹配
    return None



def find_main_model_row(
    rows: List[List[str]], model_name: str
) -> tuple[Optional[int], List[str]]:
    """
    在转置表格中识别与主模型对应的行索引。

    在转置表格中，每一行代表一个不同的模型，第一列包含模型名称。

    参数:
        rows: 表格数据行
        model_name: 来自 repo_id 的模型名称（例如，"OLMo-3-32B"）

    返回:
        (row_index, available_models) 的元组
        - row_index: 主模型的索引，如果未找到精确匹配则返回 None
        - available_models: 表格中找到的所有模型名称列表
    """
    if not rows or not model_name:
        return None, []

    model_tokens, _ = normalize_model_name(model_name)
    available_models = []

    for i, row in enumerate(rows):
        if not row or not row[0]:
            continue

        row_name = row[0].strip()

        # 跳过分隔符/标题行
        if not row_name or row_name.startswith('---'):
            continue

        row_tokens, _ = normalize_model_name(row_name)

        # 收集所有非空模型名称
        if row_tokens:
            available_models.append(row_name)

        # 检查精确令牌匹配
        if model_tokens == row_tokens:
            return i, available_models

    return None, available_models



def is_transposed_table(header: List[str], rows: List[List[str]]) -> bool:
    """
    判断表格是否是转置的（模型作为行，基准测试作为列）。

    表格被认为是转置的条件：
    - 第一列包含模型类名称（而非基准测试名称）
    - 大多数其他列包含数值
    - 标题行包含基准测试类名称

    参数:
        header: 表格列标题
        rows: 表格数据行

    返回:
        如果表格看起来是转置的，则返回 True，否则返回 False
    """
    if not header or not rows or len(header) < 3:
        return False

    # 检查第一列标题是否暗示模型名称
    first_col = header[0].lower()
    model_indicators = ["model", "system", "llm", "name"]
    has_model_header = any(indicator in first_col for indicator in model_indicators)

    # 检查其余标题是否看起来像基准测试
    benchmark_keywords = [
        "mmlu", "humaneval", "gsm", "hellaswag", "arc", "winogrande",
        "eval", "score", "benchmark", "test", "math", "code", "mbpp",
        "truthfulqa", "boolq", "piqa", "siqa", "drop", "squad"
    ]

    benchmark_header_count = 0
    for col_name in header[1:]:
        col_lower = col_name.lower()
        if any(keyword in col_lower for keyword in benchmark_keywords):
            benchmark_header_count += 1

    has_benchmark_headers = benchmark_header_count >= 2

    # 检查数据行是否在大多数列中包含数值（除第一列外）
    numeric_count = 0
    total_cells = 0

    for row in rows[:5]:  # 检查前 5 行
        for cell in row[1:]:  # 跳过第一列
            total_cells += 1
            try:
                float(cell.replace("%", "").replace(",", "").strip())
                numeric_count += 1
            except (ValueError, AttributeError):
                continue

    has_numeric_data = total_cells > 0 and (numeric_count / total_cells) > 0.5

    return (has_model_header or has_benchmark_headers) and has_numeric_data



def extract_metrics_from_table(
    header: List[str],
    rows: List[List[str]],
    table_format: str = "auto",
    model_name: Optional[str] = None,
    model_column_index: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    从解析的表格数据中提取指标。

    参数:
        header: 表格列标题
        rows: 表格数据行
        table_format: "rows"（基准测试作为行）、"columns"（基准测试作为列）、
                     "transposed"（模型作为行，基准测试作为列）或 "auto"
        model_name: 可选的模型名称，用于识别正确的列/行

    返回:
        带有名称、类型和值的指标字典列表
    """

    metrics = []

    if table_format == "auto":
        # 首先检查是否是转置表格（模型作为行）
        if is_transposed_table(header, rows):
            table_format = "transposed"
        else:
            # 检查第一列标题是否为空/通用（表示基准测试在行中）
            first_header = header[0].lower().strip() if header else ""
            is_first_col_benchmarks = not first_header or first_header in ["", "benchmark", "task", "dataset", "metric", "eval"]

            if is_first_col_benchmarks:
                table_format = "rows"
            else:
                # 启发式：如果第一行大部分是数值，基准测试在列中
                try:
                    numeric_count = sum(
                        1 for cell in rows[0] if cell and
                        re.match(r"^\d+\.?\d*%?$", cell.replace(",", "").strip())
                    )
                    table_format = "columns" if numeric_count > len(rows[0]) / 2 else "rows"
                except (IndexError, ValueError):
                    table_format = "rows"

    if table_format == "rows":
        # 基准测试在行中，分数在列中
        # 如果提供了 model_name，尝试识别主模型列
        target_column = model_column_index
        if target_column is None and model_name:
            target_column = find_main_model_column(header, model_name)

        for row in rows:
            if not row:
                continue

            benchmark_name = row[0].strip()
            if not benchmark_name:
                continue

            # 如果我们识别了特定列，使用它；否则使用第一个数值
            if target_column is not None and target_column < len(row):
                try:
                    value_str = row[target_column].replace("%", "").replace(",", "").strip()
                    if value_str:
                        value = float(value_str)
                        metrics.append({
                            "name": benchmark_name,
                            "type": benchmark_name.lower().replace(" ", "_"),
                            "value": value
                        })
                except (ValueError, IndexError):
                    pass
            else:
                # 从剩余列中提取数值（原始行为）
                for i, cell in enumerate(row[1:], start=1):
                    try:
                        # 移除常见后缀并转换为浮点数
                        value_str = cell.replace("%", "").replace(",", "").strip()
                        if not value_str:
                            continue

                        value = float(value_str)

                        # 确定指标名称
                        metric_name = benchmark_name
                        if len(header) > i and header[i].lower() not in ["score", "value", "result"]:
                            metric_name = f"{benchmark_name} ({header[i]})"

                        metrics.append({
                            "name": metric_name,
                            "type": benchmark_name.lower().replace(" ", "_"),
                            "value": value
                        })
                        break  # 每行只取第一个数值
                    except (ValueError, IndexError):
                        continue

    elif table_format == "transposed":
        # 模型在行中（第一列），基准测试在列中（标题）
        # 找到与目标模型匹配的行
        if not model_name:
            print("警告：转置表格格式需要 model_name")
            return metrics

        target_row_idx, available_models = find_main_model_row(rows, model_name)

        if target_row_idx is None:
            print(f"\n⚠ 无法在转置表格中找到模型 '{model_name}'")
            if available_models:
                print("\n表格中的可用模型：")
                for i, model in enumerate(available_models, 1):
                    print(f"  {i}. {model}")
                print("\n请从上面的列表中选择正确的模型名称。")
                print("您可以使用 --model-name-override 标志指定：")
                print(f'  --model-name-override "{available_models[0]}"')
            return metrics

        target_row = rows[target_row_idx]

        # 从每列提取指标（跳过第一列，因为是模型名称）
        for i in range(1, len(header)):
            benchmark_name = header[i].strip()
            if not benchmark_name or i >= len(target_row):
                continue

            try:
                value_str = target_row[i].replace("%", "").replace(",", "").strip()
                if not value_str:
                    continue

                value = float(value_str)

                metrics.append({
                    "name": benchmark_name,
                    "type": benchmark_name.lower().replace(" ", "_").replace("-", "_"),
                    "value": value
                })
            except (ValueError, AttributeError):
                continue

    else:  # table_format == "columns"
        # 基准测试在列中
        if not rows:
            return metrics

        # 使用第一数据行的值
        data_row = rows[0]

        for i, benchmark_name in enumerate(header):
            if not benchmark_name or i >= len(data_row):
                continue

            try:
                value_str = data_row[i].replace("%", "").replace(",", "").strip()
                if not value_str:
                    continue

                value = float(value_str)

                metrics.append({
                    "name": benchmark_name,
                    "type": benchmark_name.lower().replace(" ", "_"),
                    "value": value
                })
            except ValueError:
                continue

    return metrics



def extract_evaluations_from_readme(
    repo_id: str,
    task_type: str = "text-generation",
    dataset_name: str = "Benchmarks",
    dataset_type: str = "benchmark",
    model_name_override: Optional[str] = None,
    table_index: Optional[int] = None,
    model_column_index: Optional[int] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    从模型的 README 中提取评估结果。

    参数:
        repo_id: Hugging Face 模型仓库 ID
        task_type: model-index 的任务类型（例如，"text-generation"）
        dataset_name: 基准测试数据集的名称
        dataset_type: 数据集的类型标识符
        model_name_override: 用于匹配的覆盖模型名称（比较表格的列标题）
        table_index: 来自 inspect-tables 输出的 1-索引表格编号

    返回:
        model-index 格式的结果，如果未找到评估则返回 None
    """

    try:
        load_env()
        ModelCard = require_model_card()
        hf_token = os.getenv("HF_TOKEN")
        card = ModelCard.load(repo_id, token=hf_token)
        readme_content = card.content

        if not readme_content:
            print(f"未找到 {repo_id} 的 README 内容")
            return None

        # 从 repo_id 提取模型名称或使用覆盖
        if model_name_override:
            model_name = model_name_override
            print(f"使用模型名称覆盖: '{model_name}'")
        else:
            model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        # 使用 markdown-it 解析器进行准确的表格提取
        all_tables = extract_tables_with_parser(readme_content)

        if not all_tables:
            print(f"在 {repo_id} 的 README 中未找到表格")
            return None

        # 如果指定了 table_index，使用该特定表格
        if table_index is not None:
            if table_index < 1 or table_index > len(all_tables):
                print(f"无效的表格索引 {table_index}。找到 {len(all_tables)} 个表格。")
                print("运行 inspect-tables 查看可用表格。")
                return None
            tables_to_process = [all_tables[table_index - 1]]
        else:
            # 仅过滤评估表格
            eval_tables = []
            for table in all_tables:
                header = table.get("headers", [])
                rows = table.get("rows", [])
                if is_evaluation_table(header, rows):
                    eval_tables.append(table)

            if len(eval_tables) > 1:
                print(f"\n⚠ 找到 {len(eval_tables)} 个评估表格。")
                print("首先运行 inspect-tables，然后使用 --table 选择一个：")
                print(f'  uv run scripts/evaluation_manager.py inspect-tables --repo-id "{repo_id}"')
                return None
            elif len(eval_tables) == 0:
                print(f"在 {repo_id} 的 README 中未找到评估表格")
                return None

            tables_to_process = eval_tables

        # 从选定的表格中提取指标
        all_metrics = []
        for table in tables_to_process:
            header = table.get("headers", [])
            rows = table.get("rows", [])
            metrics = extract_metrics_from_table(
                header,
                rows,
                model_name=model_name,
                model_column_index=model_column_index
            )
            all_metrics.extend(metrics)

        if not all_metrics:
            print(f"未从表格中提取指标")
            return None

        # 构建 model-index 结构
        display_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        results = [{
            "task": {"type": task_type},
            "dataset": {
                "name": dataset_name,
                "type": dataset_type
            },
            "metrics": all_metrics,
            "source": {
                "name": "Model README",
                "url": f"https://huggingface.co/{repo_id}"
            }
        }]

        return results

    except Exception as e:
        print(f"从 README 提取评估时出错: {e}")
        return None



# ============================================================================
# Table Inspection (using markdown-it-py for accurate parsing)
# ============================================================================


def extract_tables_with_parser(markdown_content: str) -> List[Dict[str, Any]]:
    """
    使用 markdown-it-py 解析器从 markdown 中提取表格。
    使用 GFM（GitHub Flavored Markdown），它包含表格支持。
    """
    MarkdownIt = require_markdown_it()
    # 禁用 linkify 以避免可选依赖错误；表格解析不需要它。
    md = MarkdownIt("gfm-like", {"linkify": False})
    tokens = md.parse(markdown_content)

    tables = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.type == "table_open":
            table_data = {"headers": [], "rows": []}
            current_row = []
            in_header = False

            i += 1
            while i < len(tokens) and tokens[i].type != "table_close":
                t = tokens[i]
                if t.type == "thead_open":
                    in_header = True
                elif t.type == "thead_close":
                    in_header = False
                elif t.type == "tr_open":
                    current_row = []
                elif t.type == "tr_close":
                    if in_header:
                        table_data["headers"] = current_row
                    else:
                        table_data["rows"].append(current_row)
                    current_row = []
                elif t.type == "inline":
                    current_row.append(t.content.strip())
                i += 1

            if table_data["headers"] or table_data["rows"]:
                tables.append(table_data)

        i += 1

    return tables



def detect_table_format(table: Dict[str, Any], repo_id: str) -> Dict[str, Any]:
    """分析表格以检测其格式并识别模型列。"""
    headers = table.get("headers", [])
    rows = table.get("rows", [])

    if not headers or not rows:
        return {"format": "unknown", "columns": headers, "model_columns": [], "row_count": 0, "sample_rows": []}

    first_header = headers[0].lower() if headers else ""
    is_first_col_benchmarks = not first_header or first_header in ["", "benchmark", "task", "dataset", "metric", "eval"]

    # 检查数值列
    numeric_columns = []
    for col_idx in range(1, len(headers)):
        numeric_count = 0
        for row in rows[:5]:
            if col_idx < len(row):
                try:
                    val = re.sub(r'\s*\([^)]*\)', '', row[col_idx])
                    float(val.replace("%", "").replace(",", "").strip())
                    numeric_count += 1
                except (ValueError, AttributeError):
                    pass
        if numeric_count > len(rows[:5]) / 2:
            numeric_columns.append(col_idx)

    # 确定格式
    if is_first_col_benchmarks and len(numeric_columns) > 1:
        format_type = "comparison"
    elif is_first_col_benchmarks and len(numeric_columns) == 1:
        format_type = "simple"
    elif len(numeric_columns) > len(headers) / 2:
        format_type = "transposed"
    else:
        format_type = "unknown"

    # 查找模型列
    model_columns = []
    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    model_tokens, _ = normalize_model_name(model_name)

    for idx, header in enumerate(headers):
        if idx == 0 and is_first_col_benchmarks:
            continue
        if header:
            header_tokens, _ = normalize_model_name(header)
            is_match = model_tokens == header_tokens
            is_partial = model_tokens.issubset(header_tokens) or header_tokens.issubset(model_tokens)
            model_columns.append({
                "index": idx,
                "header": header,
                "is_exact_match": is_match,
                "is_partial_match": is_partial and not is_match
            })

    return {
        "format": format_type,
        "columns": headers,
        "model_columns": model_columns,
        "row_count": len(rows),
        "sample_rows": [row[0] for row in rows[:5] if row]
    }



def inspect_tables(repo_id: str) -> None:
    """检查并显示模型 README 中的所有评估表格。"""

    try:
        load_env()
        ModelCard = require_model_card()
        hf_token = os.getenv("HF_TOKEN")
        card = ModelCard.load(repo_id, token=hf_token)
        readme_content = card.content

        if not readme_content:
            print(f"未找到 {repo_id} 的 README 内容")
            return

        tables = extract_tables_with_parser(readme_content)

        if not tables:
            print(f"在 {repo_id} 的 README 中未找到表格")
            return

        print(f"\n{'='*70}")
        print(f"在 README 中找到的表格: {repo_id}")
        print(f"{'='*70}")


        eval_table_count = 0
        for table in tables:
            analysis = detect_table_format(table, repo_id)

            if analysis["format"] == "unknown" and not analysis.get("sample_rows"):
                continue

            eval_table_count += 1
            print(f"\n## Table {eval_table_count}")
            print(f"   Format: {analysis['format']}")
            print(f"   Rows: {analysis['row_count']}")

            print(f"\n   Columns ({len(analysis['columns'])}):")
            for col_info in analysis.get("model_columns", []):
                idx = col_info["index"]
                header = col_info["header"]
                if col_info["is_exact_match"]:
                    print(f"      [{idx}] {header}  ✓ EXACT MATCH")
                elif col_info["is_partial_match"]:
                    print(f"      [{idx}] {header}  ~ partial match")
                else:
                    print(f"      [{idx}] {header}")

            if analysis.get("sample_rows"):
                print(f"\n   Sample rows (first column):")
                for row_val in analysis["sample_rows"][:5]:
                    print(f"      - {row_val}")

        if eval_table_count == 0:
            print("\nNo evaluation tables detected.")
        else:
            print("\nSuggested next step:")
            print(f'  uv run scripts/evaluation_manager.py extract-readme --repo-id "{repo_id}" --table <table-number> [--model-column-index <column-index>]')

        print(f"\n{'='*70}\n")

    except Exception as e:
        print(f"Error inspecting tables: {e}")


# ============================================================================
# Pull Request Management
# ============================================================================


def get_open_prs(repo_id: str) -> List[Dict[str, Any]]:
    """
    Fetch open pull requests for a Hugging Face model repository.

    Args:
        repo_id: Hugging Face model repository ID (e.g., "allenai/Olmo-3-32B-Think")

    Returns:
        List of open PR dictionaries with num, title, author, and createdAt
    """
    requests = require_requests()
    url = f"https://huggingface.co/api/models/{repo_id}/discussions"

    try:
        response = requests.get(url, timeout=30, allow_redirects=True)
        response.raise_for_status()

        data = response.json()
        discussions = data.get("discussions", [])

        open_prs = [
            {
                "num": d["num"],
                "title": d["title"],
                "author": d["author"]["name"],
                "createdAt": d.get("createdAt", "unknown"),
            }
            for d in discussions
            if d.get("status") == "open" and d.get("isPullRequest")
        ]

        return open_prs

    except requests.RequestException as e:
        print(f"Error fetching PRs from Hugging Face: {e}")
        return []


def list_open_prs(repo_id: str) -> None:
    """Display open pull requests for a model repository."""
    prs = get_open_prs(repo_id)

    print(f"\n{'='*70}")
    print(f"Open Pull Requests for: {repo_id}")
    print(f"{'='*70}")

    if not prs:
        print("\nNo open pull requests found.")
    else:
        print(f"\nFound {len(prs)} open PR(s):\n")
        for pr in prs:
            print(f"  PR #{pr['num']} - {pr['title']}")
            print(f"     Author: {pr['author']}")
            print(f"     Created: {pr['createdAt']}")
            print(f"     URL: https://huggingface.co/{repo_id}/discussions/{pr['num']}")
            print()

    print(f"{'='*70}\n")


# ============================================================================
# Method 2: Import from Artificial Analysis
# ============================================================================


def get_aa_model_data(creator_slug: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Fetch model evaluation data from Artificial Analysis API.

    Args:
        creator_slug: Creator identifier (e.g., "anthropic", "openai")
        model_name: Model slug/identifier

    Returns:
        Model data dictionary or None if not found
    """
    load_env()
    AA_API_KEY = os.getenv("AA_API_KEY")
    if not AA_API_KEY:
        raise ValueError("AA_API_KEY environment variable is not set")

    url = "https://artificialanalysis.ai/api/v2/data/llms/models"
    headers = {"x-api-key": AA_API_KEY}

    requests = require_requests()

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json().get("data", [])

        for model in data:
            creator = model.get("model_creator", {})
            if creator.get("slug") == creator_slug and model.get("slug") == model_name:
                return model

        print(f"Model {creator_slug}/{model_name} not found in Artificial Analysis")
        return None

    except requests.RequestException as e:
        print(f"Error fetching data from Artificial Analysis: {e}")
        return None


def aa_data_to_model_index(
    model_data: Dict[str, Any],
    dataset_name: str = "Artificial Analysis Benchmarks",
    dataset_type: str = "artificial_analysis",
    task_type: str = "evaluation"
) -> List[Dict[str, Any]]:
    """
    Convert Artificial Analysis model data to model-index format.

    Args:
        model_data: Raw model data from AA API
        dataset_name: Dataset name for model-index
        dataset_type: Dataset type identifier
        task_type: Task type for model-index

    Returns:
        Model-index formatted results
    """
    model_name = model_data.get("name", model_data.get("slug", "unknown-model"))
    evaluations = model_data.get("evaluations", {})

    if not evaluations:
        print(f"No evaluations found for model {model_name}")
        return []

    metrics = []
    for key, value in evaluations.items():
        if value is not None:
            metrics.append({
                "name": key.replace("_", " ").title(),
                "type": key,
                "value": value
            })

    results = [{
        "task": {"type": task_type},
        "dataset": {
            "name": dataset_name,
            "type": dataset_type
        },
        "metrics": metrics,
        "source": {
            "name": "Artificial Analysis API",
            "url": "https://artificialanalysis.ai"
        }
    }]

    return results


def import_aa_evaluations(
    creator_slug: str,
    model_name: str,
    repo_id: str
) -> Optional[List[Dict[str, Any]]]:
    """
    Import evaluation results from Artificial Analysis for a model.

    Args:
        creator_slug: Creator identifier in AA
        model_name: Model identifier in AA
        repo_id: Hugging Face repository ID to update

    Returns:
        Model-index formatted results or None if import fails
    """
    model_data = get_aa_model_data(creator_slug, model_name)

    if not model_data:
        return None

    results = aa_data_to_model_index(model_data)
    return results


# ============================================================================
# Model Card Update Functions
# ============================================================================


def update_model_card_with_evaluations(
    repo_id: str,
    results: List[Dict[str, Any]],
    create_pr: bool = False,
    commit_message: Optional[str] = None
) -> bool:
    """
    Update a model card with evaluation results.

    Args:
        repo_id: Hugging Face repository ID
        results: Model-index formatted results
        create_pr: Whether to create a PR instead of direct push
        commit_message: Custom commit message

    Returns:
        True if successful, False otherwise
    """
    try:
        load_env()
        ModelCard = require_model_card()
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set")

        # Load existing card
        card = ModelCard.load(repo_id, token=hf_token)

        # Get model name
        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        # Create or update model-index
        model_index = [{
            "name": model_name,
            "results": results
        }]

        # Merge with existing model-index if present
        if "model-index" in card.data:
            existing = card.data["model-index"]
            if isinstance(existing, list) and existing:
                # Keep existing name if present
                if "name" in existing[0]:
                    model_index[0]["name"] = existing[0]["name"]

                # Merge results
                existing_results = existing[0].get("results", [])
                model_index[0]["results"].extend(existing_results)

        card.data["model-index"] = model_index

        # Prepare commit message
        if not commit_message:
            commit_message = f"Add evaluation results to {model_name}"

        commit_description = (
            "This commit adds structured evaluation results to the model card. "
            "The results are formatted using the model-index specification and "
            "will be displayed in the model card's evaluation widget."
        )

        # Push update
        card.push_to_hub(
            repo_id,
            token=hf_token,
            commit_message=commit_message,
            commit_description=commit_description,
            create_pr=create_pr
        )

        action = "Pull request created" if create_pr else "Model card updated"
        print(f"✓ {action} successfully for {repo_id}")
        return True

    except Exception as e:
        print(f"Error updating model card: {e}")
        return False


def show_evaluations(repo_id: str) -> None:
    """Display current evaluations in a model card."""
    try:
        load_env()
        ModelCard = require_model_card()
        hf_token = os.getenv("HF_TOKEN")
        card = ModelCard.load(repo_id, token=hf_token)

        if "model-index" not in card.data:
            print(f"No model-index found in {repo_id}")
            return

        model_index = card.data["model-index"]

        print(f"\nEvaluations for {repo_id}:")
        print("=" * 60)

        for model_entry in model_index:
            model_name = model_entry.get("name", "Unknown")
            print(f"\nModel: {model_name}")

            results = model_entry.get("results", [])
            for i, result in enumerate(results, 1):
                print(f"\n  Result Set {i}:")

                task = result.get("task", {})
                print(f"    Task: {task.get('type', 'unknown')}")

                dataset = result.get("dataset", {})
                print(f"    Dataset: {dataset.get('name', 'unknown')}")

                metrics = result.get("metrics", [])
                print(f"    Metrics ({len(metrics)}):")
                for metric in metrics:
                    name = metric.get("name", "Unknown")
                    value = metric.get("value", "N/A")
                    print(f"      - {name}: {value}")

                source = result.get("source", {})
                if source:
                    print(f"    Source: {source.get('name', 'Unknown')}")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"Error showing evaluations: {e}")


def validate_model_index(repo_id: str) -> bool:
    """Validate model-index format in a model card."""
    try:
        load_env()
        ModelCard = require_model_card()
        hf_token = os.getenv("HF_TOKEN")
        card = ModelCard.load(repo_id, token=hf_token)

        if "model-index" not in card.data:
            print(f"✗ No model-index found in {repo_id}")
            return False

        model_index = card.data["model-index"]

        if not isinstance(model_index, list):
            print("✗ model-index must be a list")
            return False

        for i, entry in enumerate(model_index):
            if "name" not in entry:
                print(f"✗ Entry {i} missing 'name' field")
                return False

            if "results" not in entry:
                print(f"✗ Entry {i} missing 'results' field")
                return False

            for j, result in enumerate(entry["results"]):
                if "task" not in result:
                    print(f"✗ Result {j} in entry {i} missing 'task' field")
                    return False

                if "dataset" not in result:
                    print(f"✗ Result {j} in entry {i} missing 'dataset' field")
                    return False

                if "metrics" not in result:
                    print(f"✗ Result {j} in entry {i} missing 'metrics' field")
                    return False

        print(f"✓ Model-index format is valid for {repo_id}")
        return True

    except Exception as e:
        print(f"Error validating model-index: {e}")
        return False


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Manage evaluation results in Hugging Face model cards.\n\n"
            "Use standard Python or `uv run scripts/evaluation_manager.py ...` "
            "to auto-resolve dependencies from the PEP 723 header."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=dedent(
            """\
            Typical workflows:
              - Inspect tables first:
                  uv run scripts/evaluation_manager.py inspect-tables --repo-id <model>
              - Extract from README (prints YAML by default):
                  uv run scripts/evaluation_manager.py extract-readme --repo-id <model> --table N
              - Apply changes:
                  uv run scripts/evaluation_manager.py extract-readme --repo-id <model> --table N --apply
              - Import from Artificial Analysis:
                  AA_API_KEY=... uv run scripts/evaluation_manager.py import-aa --creator-slug org --model-name slug --repo-id <model>

            Tips:
              - YAML is printed by default; use --apply or --create-pr to write changes.
              - Set HF_TOKEN (and AA_API_KEY for import-aa); .env is loaded automatically if python-dotenv is installed.
              - When multiple tables exist, run inspect-tables then select with --table N.
              - To apply changes (push or PR), rerun extract-readme with --apply or --create-pr.
            """
        ),
    )
    parser.add_argument("--version", action="version", version="evaluation_manager 1.2.0")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Extract from README command
    extract_parser = subparsers.add_parser(
        "extract-readme",
        help="Extract evaluation tables from model README",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Parse README tables into model-index YAML. Default behavior prints YAML; use --apply/--create-pr to write changes.",
        epilog=dedent(
            """\
            Examples:
              uv run scripts/evaluation_manager.py extract-readme --repo-id username/model
              uv run scripts/evaluation_manager.py extract-readme --repo-id username/model --table 2 --model-column-index 3
              uv run scripts/evaluation_manager.py extract-readme --repo-id username/model --table 2 --model-name-override \"**Model 7B**\"  # exact header text
              uv run scripts/evaluation_manager.py extract-readme --repo-id username/model --table 2 --create-pr

            Apply changes:
              - Default: prints YAML to stdout (no writes).
              - Add --apply to push directly, or --create-pr to open a PR.
            Model selection:
              - Preferred: --model-column-index <header index shown by inspect-tables>
              - If using --model-name-override, copy the column header text exactly.
            """
        ),
    )
    extract_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")
    extract_parser.add_argument("--table", type=int, help="Table number (1-indexed, from inspect-tables output)")
    extract_parser.add_argument("--model-column-index", type=int, help="Preferred: column index from inspect-tables output (exact selection)")
    extract_parser.add_argument("--model-name-override", type=str, help="Exact column header/model name for comparison/transpose tables (when index is not used)")
    extract_parser.add_argument("--task-type", type=str, default="text-generation", help="Sets model-index task.type (e.g., text-generation, summarization)")
    extract_parser.add_argument("--dataset-name", type=str, default="Benchmarks", help="Dataset name")
    extract_parser.add_argument("--dataset-type", type=str, default="benchmark", help="Dataset type")
    extract_parser.add_argument("--create-pr", action="store_true", help="Create PR instead of direct push")
    extract_parser.add_argument("--apply", action="store_true", help="Apply changes (default is to print YAML only)")
    extract_parser.add_argument("--dry-run", action="store_true", help="Preview YAML without updating (default)")

    # Import from AA command
    aa_parser = subparsers.add_parser(
        "import-aa",
        help="Import evaluation scores from Artificial Analysis",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Fetch scores from Artificial Analysis API and write them into model-index.",
        epilog=dedent(
            """\
            Examples:
              AA_API_KEY=... uv run scripts/evaluation_manager.py import-aa --creator-slug anthropic --model-name claude-sonnet-4 --repo-id username/model
              uv run scripts/evaluation_manager.py import-aa --creator-slug openai --model-name gpt-4o --repo-id username/model --create-pr

            Requires: AA_API_KEY in env (or .env if python-dotenv installed).
            """
        ),
    )
    aa_parser.add_argument("--creator-slug", type=str, required=True, help="AA creator slug")
    aa_parser.add_argument("--model-name", type=str, required=True, help="AA model name")
    aa_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")
    aa_parser.add_argument("--create-pr", action="store_true", help="Create PR instead of direct push")

    # Show evaluations command
    show_parser = subparsers.add_parser(
        "show",
        help="Display current evaluations in model card",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Print model-index content from the model card (requires HF_TOKEN for private repos).",
    )
    show_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate model-index format",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Schema sanity check for model-index section of the card.",
    )
    validate_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")

    # Inspect tables command
    inspect_parser = subparsers.add_parser(
        "inspect-tables",
        help="Inspect tables in README → outputs suggested extract-readme command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. inspect-tables     → see table structure, columns, and table numbers
  2. extract-readme     → run with --table N (from step 1); YAML prints by default
  3. apply changes      → rerun extract-readme with --apply or --create-pr

Reminder:
  - Preferred: use --model-column-index <index>. If needed, use --model-name-override with the exact column header text.
"""
    )
    inspect_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")

    # Get PRs command
    prs_parser = subparsers.add_parser(
        "get-prs",
        help="List open pull requests for a model repository",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Check for existing open PRs before creating new ones to avoid duplicates.",
        epilog=dedent(
            """\
            Examples:
              uv run scripts/evaluation_manager.py get-prs --repo-id "allenai/Olmo-3-32B-Think"

            IMPORTANT: Always run this before using --create-pr to avoid duplicate PRs.
            """
        ),
    )
    prs_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        # Execute command
        if args.command == "extract-readme":
            results = extract_evaluations_from_readme(
                repo_id=args.repo_id,
                task_type=args.task_type,
                dataset_name=args.dataset_name,
                dataset_type=args.dataset_type,
                model_name_override=args.model_name_override,
                table_index=args.table,
                model_column_index=args.model_column_index
            )

            if not results:
                print("No evaluations extracted")
                return

            apply_changes = args.apply or args.create_pr

            # Default behavior: print YAML (dry-run)
            yaml = require_yaml()
            print("\nExtracted evaluations (YAML):")
            print(
                yaml.dump(
                    {"model-index": [{"name": args.repo_id.split('/')[-1], "results": results}]},
                    sort_keys=False
                )
            )

            if apply_changes:
                if args.model_name_override and args.model_column_index is not None:
                    print("Note: --model-column-index takes precedence over --model-name-override.")
                update_model_card_with_evaluations(
                    repo_id=args.repo_id,
                    results=results,
                    create_pr=args.create_pr,
                    commit_message="Extract evaluation results from README"
                )

        elif args.command == "import-aa":
            results = import_aa_evaluations(
                creator_slug=args.creator_slug,
                model_name=args.model_name,
                repo_id=args.repo_id
            )

            if not results:
                print("No evaluations imported")
                return

            update_model_card_with_evaluations(
                repo_id=args.repo_id,
                results=results,
                create_pr=args.create_pr,
                commit_message=f"Add Artificial Analysis evaluations for {args.model_name}"
            )

        elif args.command == "show":
            show_evaluations(args.repo_id)

        elif args.command == "validate":
            validate_model_index(args.repo_id)

        elif args.command == "inspect-tables":
            inspect_tables(args.repo_id)

        elif args.command == "get-prs":
            list_open_prs(args.repo_id)
    except ModuleNotFoundError as exc:
        # Surface dependency hints cleanly when user only needs help output
        print(exc)
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
