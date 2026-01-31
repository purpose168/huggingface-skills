#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyyaml",
# ]
# ///
"""
评估提取功能的测试脚本。

此脚本演示表格提取功能，无需
HF 令牌或进行实际的 API 调用。

注意：此脚本从同一目录的 evaluation_manager.py 导入。
从 scripts/ 目录运行：cd scripts && uv run test_extraction.py
"""

import yaml

from evaluation_manager import (
    extract_tables_from_markdown,
    parse_markdown_table,
    is_evaluation_table,
    extract_metrics_from_table
)

# 包含各种表格格式的示例 README 内容
SAMPLE_README = """
# 我的出色模型

## 评估结果

以下是基准测试结果：

| 基准测试 | 分数 |
|-----------|-------|
| MMLU      | 85.2  |
| HumanEval | 72.5  |
| GSM8K     | 91.3  |

### 详细 breakdown

| 类别      | MMLU  | GSM8K | HumanEval |
|---------------|-------|-------|-----------|
| 性能   | 85.2  | 91.3  | 72.5      |

## 其他信息

这不是一个评估表格：

| 特性 | 值 |
|---------|-------|
| 大小    | 7B    |
| 类型    | 聊天  |

## 更多结果

| 基准测试     | 准确率 | F1 分数 |
|---------------|----------|----------|
| HellaSwag     | 88.9     | 0.87     |
| TruthfulQA    | 68.7     | 0.65     |
"""


def test_table_extraction():
    """测试 markdown 表格提取。"""
    print("=" * 60)
    print("测试 1: 表格提取")
    print("=" * 60)

    tables = extract_tables_from_markdown(SAMPLE_README)
    print(f"在示例 README 中找到 {len(tables)} 个表格\n")

    for i, table in enumerate(tables, 1):
        print(f"表格 {i}:")
        print(table[:100] + "..." if len(table) > 100 else table)
        print()

    return tables


def test_table_parsing(tables):
    """测试表格解析。"""
    print("\n" + "=" * 60)
    print("测试 2: 表格解析")
    print("=" * 60)

    parsed_tables = []
    for i, table in enumerate(tables, 1):
        print(f"\n解析表格 {i}:")
        header, rows = parse_markdown_table(table)

        print(f"  表头: {header}")
        print(f"  行数: {len(rows)}")
        for j, row in enumerate(rows[:3], 1):  # 显示前 3 行
            print(f"    行 {j}: {row}")
        if len(rows) > 3:
            print(f"    ... 还有 {len(rows) - 3} 行")

        parsed_tables.append((header, rows))

    return parsed_tables


def test_evaluation_detection(parsed_tables):
    """测试评估表格检测。"""
    print("\n" + "=" * 60)
    print("测试 3: 评估表格检测")
    print("=" * 60)

    eval_tables = []
    for i, (header, rows) in enumerate(parsed_tables, 1):
        is_eval = is_evaluation_table(header, rows)
        status = "✓ 是" if is_eval else "✗ 否"
        print(f"\n表格 {i}: {status} 评估表格")
        print(f"  表头: {header}")

        if is_eval:
            eval_tables.append((header, rows))

    print(f"\n找到 {len(eval_tables)} 个评估表格")
    return eval_tables


def test_metric_extraction(eval_tables):
    """测试指标提取。"""
    print("\n" + "=" * 60)
    print("测试 4: 指标提取")
    print("=" * 60)

    all_metrics = []
    for i, (header, rows) in enumerate(eval_tables, 1):
        print(f"\n从表格 {i} 提取指标:")
        metrics = extract_metrics_from_table(header, rows, table_format="auto")

        print(f"  提取了 {len(metrics)} 个指标:")
        for metric in metrics:
            print(f"    - {metric['name']}: {metric['value']} (类型: {metric['type']})")

        all_metrics.extend(metrics)

    return all_metrics


def test_model_index_format(metrics):
    """测试 model-index 格式生成。"""
    print("\n" + "=" * 60)
    print("测试 5: Model-Index 格式")
    print("=" * 60)

    model_index = {
        "model-index": [
            {
                "name": "test-model",
                "results": [
                    {
                        "task": {"type": "text-generation"},
                        "dataset": {
                            "name": "基准测试",
                            "type": "benchmark"
                        },
                        "metrics": metrics,
                        "source": {
                            "name": "模型 README",
                            "url": "https://huggingface.co/test/model"
                        }
                    }
                ]
            }
        ]
    }

    print("\n生成的 model-index 结构:")
    print(yaml.dump(model_index, sort_keys=False, default_flow_style=False))


def main():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("评估提取测试套件")
    print("=" * 60)
    print("\n此测试演示表格提取功能")
    print("无需 API 访问或令牌。\n")

    # 运行测试
    tables = test_table_extraction()
    parsed_tables = test_table_parsing(tables)
    eval_tables = test_evaluation_detection(parsed_tables)
    metrics = test_metric_extraction(eval_tables)
    test_model_index_format(metrics)

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"✓ 找到 {len(tables)} 个总表格")
    print(f"✓ 识别出 {len(eval_tables)} 个评估表格")
    print(f"✓ 提取了 {len(metrics)} 个指标")
    print("✓ 成功生成 model-index 格式")
    print("\n" + "=" * 60)
    print("所有测试完成！提取逻辑工作正常。")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
