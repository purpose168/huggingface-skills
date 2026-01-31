#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.20.0",
# ]
# ///
"""
Hugging Face 数据集管理器

增强的数据集创建和管理工具，设计为与 HF MCP 服务器协同工作。提供数据集创建、配置和内容管理功能，为对话式 AI 训练数据进行了优化。

版本: 2.0.0

使用方法:
    uv run dataset_manager.py init --repo_id username/dataset-name
    uv run dataset_manager.py quick_setup --repo_id username/dataset-name --template chat
    uv run dataset_manager.py add_rows --repo_id username/dataset-name --rows_json '[{"messages": [...]}]'
    uv run dataset_manager.py stats --repo_id username/dataset-name
    uv run dataset_manager.py list_templates
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

# 配置
HF_TOKEN = os.environ.get("HF_TOKEN")
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def init_dataset(repo_id, token=None, private=True):
    """
    在 Hugging Face Hub 上初始化新的数据集仓库。
    """
    api = HfApi(token=token)
    try:
        create_repo(repo_id, repo_type="dataset", private=private, token=token)
        print(f"已创建数据集仓库: {repo_id}")
    except HfHubHTTPError as e:
        if "409" in str(e):
            print(f"仓库 {repo_id} 已存在。")
        else:
            raise e

    # 如果不存在，则创建带有元数据的基本 README.md
    readme_content = f"""---
license: mit
---

# {repo_id.split("/")[-1]}

此数据集是使用 Claude 数据集技能创建的。
"""
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="初始化数据集 README",
        )
    except Exception as e:
        print(f"注意: README 可能已存在或更新失败: {e}")


def define_config(repo_id, system_prompt=None, token=None):
    """
    定义数据集的配置，包括系统提示。
    这会将 config.json 文件保存到仓库。
    """
    api = HfApi(token=token)

    config_data = {"dataset_config": {"version": "1.0", "created_at": time.time()}}

    if system_prompt:
        config_data["system_prompt"] = system_prompt

    # 上传 config.json
    api.upload_file(
        path_or_fileobj=json.dumps(config_data, indent=2).encode("utf-8"),
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="更新数据集配置",
    )
    print(f"已更新 {repo_id} 的配置")


def load_dataset_template(template_name: str) -> Dict[str, Any]:
    """从模板目录加载数据集模板配置。"""
    template_path = EXAMPLES_DIR.parent / "templates" / f"{template_name}.json"
    if not template_path.exists():
        available_templates = [f.stem for f in (EXAMPLES_DIR.parent / "templates").glob("*.json")]
        print(f"❌ 模板 '{template_name}' 未找到。")
        print(f"可用模板: {', '.join(available_templates)}")
        return {}

    with open(template_path) as f:
        return json.load(f)


def validate_by_template(rows: List[Dict[str, Any]], template: Dict[str, Any]) -> bool:
    """根据模板架构验证数据。"""
    if not template:
        return False

    schema = template.get("validation_schema", {})
    required_fields = set(schema.get("required_fields", []))
    recommended_fields = set(schema.get("recommended_fields", []))
    field_types = schema.get("field_types", {})

    for i, row in enumerate(rows):
        # 检查必填字段
        if not all(field in row for field in required_fields):
            missing = required_fields - set(row.keys())
            print(f"行 {i}: 缺少必填字段: {missing}")
            return False

        # 验证字段类型
        for field, expected_type in field_types.items():
            if field in row:
                if not _validate_field_type(row[field], expected_type, f"行 {i}, 字段 '{field}'"):
                    return False

        # 模板特定验证
        if template["type"] == "chat":
            if not _validate_chat_format(row, i):
                return False
        elif template["type"] == "classification":
            if not _validate_classification_format(row, i):
                return False
        elif template["type"] == "tabular":
            if not _validate_tabular_format(row, i):
                return False

        # 警告缺少推荐字段
        missing_recommended = recommended_fields - set(row.keys())
        if missing_recommended:
            print(f"行 {i}: 建议包含: {missing_recommended}")

    print(f"✓ 已验证 {len(rows)} 个示例，用于 {template['type']} 数据集")
    return True


def _validate_field_type(value: Any, expected_type: str, context: str) -> bool:
    """验证单个字段类型。"""
    if expected_type.startswith("enum:"):
        valid_values = expected_type[5:].split(",")
        if value not in valid_values:
            print(f"{context}: 无效值 '{value}'。必须是以下之一: {valid_values}")
            return False
    elif expected_type == "array" and not isinstance(value, list):
        print(f"{context}: 期望数组，得到 {type(value).__name__}")
        return False
    elif expected_type == "object" and not isinstance(value, dict):
        print(f"{context}: 期望对象，得到 {type(value).__name__}")
        return False
    elif expected_type == "string" and not isinstance(value, str):
        print(f"{context}: 期望字符串，得到 {type(value).__name__}")
        return False
    elif expected_type == "number" and not isinstance(value, (int, float)):
        print(f"{context}: 期望数字，得到 {type(value).__name__}")
        return False

    return True


def _validate_chat_format(row: Dict[str, Any], row_index: int) -> bool:
    """验证聊天特定格式。"""
    messages = row.get("messages", [])
    if not isinstance(messages, list) or len(messages) == 0:
        print(f"行 {row_index}: 'messages' 必须是非空列表")
        return False

    valid_roles = {"user", "assistant", "tool", "system"}
    for j, msg in enumerate(messages):
        if not isinstance(msg, dict):
            print(f"行 {row_index}, 消息 {j}: 必须是对象")
            return False
        if "role" not in msg or msg["role"] not in valid_roles:
            print(f"行 {row_index}, 消息 {j}: 无效角色。使用: {valid_roles}")
            return False
        if "content" not in msg:
            print(f"行 {row_index}, 消息 {j}: 缺少 'content' 字段")
            return False

    return True


def _validate_classification_format(row: Dict[str, Any], row_index: int) -> bool:
    """验证分类特定格式。"""
    if "text" not in row:
        print(f"行 {row_index}: 缺少 'text' 字段")
        return False
    if "label" not in row:
        print(f"行 {row_index}: 缺少 'label' 字段")
        return False

    return True


def _validate_tabular_format(row: Dict[str, Any], row_index: int) -> bool:
    """验证表格特定格式。"""
    if "data" not in row:
        print(f"行 {row_index}: 缺少 'data' 字段")
        return False
    if "columns" not in row:
        print(f"行 {row_index}: 缺少 'columns' 字段")
        return False

    data = row["data"]
    columns = row["columns"]

    if not isinstance(data, list):
        print(f"行 {row_index}: 'data' 必须是数组")
        return False
    if not isinstance(columns, list):
        print(f"行 {row_index}: 'columns' 必须是数组")
        return False

    return True


def validate_training_data(rows: List[Dict[str, Any]], template_name: str = "chat") -> bool:
    """
    根据模板验证训练数据结构。
    支持多种数据集类型，具有适当的验证。
    """
    template = load_dataset_template(template_name)
    if not template:
        print(f"❌ 无法加载模板 '{template_name}'，回退到基本验证")
        return _basic_validation(rows)

    return validate_by_template(rows, template)


def _basic_validation(rows: List[Dict[str, Any]]) -> bool:
    """当没有模板可用时的基本验证。"""
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            print(f"行 {i}: 必须是字典/对象")
            return False
    print(f"✓ 基本验证通过，共 {len(rows)} 行")
    return True


def add_rows(
    repo_id: str,
    rows: List[Dict[str, Any]],
    split: str = "train",
    validate: bool = True,
    template: str = "chat",
    token: Optional[str] = None,
) -> None:
    """
    通过上传新的数据块来流式更新数据集。
    增强了对多种数据集类型的验证。

    参数:
        repo_id: 仓库标识符 (username/dataset-name)
        rows: 训练示例列表
        split: 数据集拆分名称 (train, test, validation)
        validate: 是否在上传前验证数据结构
        template: 数据集模板类型 (chat, classification, qa, completion, tabular, custom)
        token: HuggingFace API 令牌
    """
    api = HfApi(token=token)

    if not rows:
        print("没有要添加的行。")
        return

    # 验证训练数据结构
    if validate and not validate_training_data(rows, template):
        print("❌ 验证失败。使用 --no-validate 跳过验证。")
        return

    # 创建以换行符分隔的 JSON 字符串
    jsonl_content = "\n".join(json.dumps(row) for row in rows)

    # 为此数据块生成唯一文件名
    timestamp = int(time.time() * 1000)
    filename = f"data/{split}-{timestamp}.jsonl"

    try:
        api.upload_file(
            path_or_fileobj=jsonl_content.encode("utf-8"),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"向 {split} 拆分添加 {len(rows)} 行",
        )
        print(f"✅ 已向 {repo_id} 添加 {len(rows)} 行 (拆分: {split})")
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return


def load_template(template_name: str = "system_prompt_template.txt") -> str:
    """从示例目录加载模板文件。"""
    template_path = EXAMPLES_DIR / template_name
    if template_path.exists():
        return template_path.read_text()
    else:
        print(f"⚠️ 模板 {template_name} 在 {template_path} 未找到")
        return ""


def quick_setup(repo_id: str, template_type: str = "chat", token: Optional[str] = None) -> None:
    """
    使用模板快速设置不同的数据集类型。

    参数:
        repo_id: 仓库标识符
        template_type: 数据集模板 (chat, classification, qa, completion, tabular, custom)
        token: HuggingFace API 令牌
    """
    print(f"🚀 使用 '{template_type}' 模板快速设置 {repo_id}...")

    # 加载模板配置
    template_config = load_dataset_template(template_type)
    if not template_config:
        print(f"❌ 无法加载模板 '{template_type}'。设置已取消。")
        return

    # 初始化仓库
    init_dataset(repo_id, token=token, private=True)

    # 使用模板系统提示配置
    system_prompt = template_config.get("system_prompt", "")
    if system_prompt:
        define_config(repo_id, system_prompt=system_prompt, token=token)

    # 添加模板示例
    examples = template_config.get("examples", [])
    if examples:
        add_rows(repo_id, examples, template=template_type, token=token)
        print(f"✅ 从模板添加了 {len(examples)} 个示例")

    print(f"✅ 已完成 {repo_id} 的快速设置")
    print(f"📊 数据集类型: {template_config.get('description', '无描述')}")

    # 显示后续步骤
    print(f"\n📋 后续步骤:")
    print(
        f"1. 添加更多数据: python scripts/dataset_manager.py add_rows --repo_id {repo_id} --template {template_type} --rows_json 'your_data.json'"
    )
    print(f"2. 查看统计信息: python scripts/dataset_manager.py stats --repo_id {repo_id}")
    print(f"3. 浏览: https://huggingface.co/datasets/{repo_id}")


def show_stats(repo_id: str, token: Optional[str] = None) -> None:
    """显示数据集的统计信息。"""
    api = HfApi(token=token)

    try:
        # 获取仓库信息
        repo_info = api.repo_info(repo_id, repo_type="dataset")
        print(f"\n📊 数据集统计信息: {repo_id}")
        print(f"创建时间: {repo_info.created_at}")
        print(f"更新时间: {repo_info.last_modified}")
        print(f"私有: {repo_info.private}")

        # 列出文件
        files = api.list_repo_files(repo_id, repo_type="dataset")
        data_files = [f for f in files if f.startswith("data/")]
        print(f"数据文件: {len(data_files)}")

        if "config.json" in files:
            print("✅ 配置存在")
        else:
            print("⚠️ 未找到配置")

    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")


def list_available_templates() -> None:
    """列出所有可用的数据集模板及其描述。"""
    templates_dir = EXAMPLES_DIR.parent / "templates"

    if not templates_dir.exists():
        print("❌ 未找到模板目录")
        return

    print("\n📋 可用数据集模板:")
    print("=" * 50)

    for template_file in templates_dir.glob("*.json"):
        try:
            with open(template_file) as f:
                template = json.load(f)

            name = template_file.stem
            desc = template.get("description", "无可用描述")
            template_type = template.get("type", name)

            print(f"\n🏷️  {name}")
            print(f"   类型: {template_type}")
            print(f"   描述: {desc}")

            # 显示必填字段
            schema = template.get("validation_schema", {})
            required = schema.get("required_fields", [])
            if required:
                print(f"   必填字段: {', '.join(required)}")

        except Exception as e:
            print(f"❌ 加载模板 {template_file.name} 时出错: {e}")

    print(
        f"\n💡 使用方法: python scripts/dataset_manager.py quick_setup --repo_id your-username/dataset-name --template TEMPLATE_NAME"
    )
    print(f"📚 示例模板目录: {templates_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face 数据集管理器")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 初始化命令
    init_parser = subparsers.add_parser("init", help="初始化新数据集")
    init_parser.add_argument("--repo_id", required=True, help="仓库 ID (user/repo_name)")
    init_parser.add_argument("--private", action="store_true", help="将仓库设为私有")

    # 配置命令
    config_parser = subparsers.add_parser("config", help="设置数据集配置")
    config_parser.add_argument("--repo_id", required=True, help="仓库 ID")
    config_parser.add_argument("--system_prompt", help="存储在配置中的系统提示")

    # 添加行命令
    add_parser = subparsers.add_parser("add_rows", help="向数据集添加行")
    add_parser.add_argument("--repo_id", required=True, help="仓库 ID")
    add_parser.add_argument("--split", default="train", help="数据集拆分 (例如: train, test)")
    add_parser.add_argument(
        "--template",
        default="chat",
        choices=[
            "chat",
            "classification",
            "qa",
            "completion",
            "tabular",
            "custom",
        ],
        help="用于验证的数据集模板类型",
    )
    add_parser.add_argument(
        "--rows_json",
        required=True,
        help="包含行列表的 JSON 字符串",
    )
    add_parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="跳过数据验证",
    )

    # 快速设置命令
    setup_parser = subparsers.add_parser("quick_setup", help="使用模板快速设置")
    setup_parser.add_argument("--repo_id", required=True, help="仓库 ID")
    setup_parser.add_argument(
        "--template",
        default="chat",
        choices=[
            "chat",
            "classification",
            "qa",
            "completion",
            "tabular",
            "custom",
        ],
        help="数据集模板类型",
    )

    # 统计命令
    stats_parser = subparsers.add_parser("stats", help="显示数据集统计信息")
    stats_parser.add_argument("--repo_id", required=True, help="仓库 ID")

    # 列出模板命令
    templates_parser = subparsers.add_parser("list_templates", help="列出可用的数据集模板")

    args = parser.parse_args()

    token = HF_TOKEN
    if not token:
        print("警告: 未设置 HF_TOKEN 环境变量。")

    if args.command == "init":
        init_dataset(args.repo_id, token=token, private=args.private)
    elif args.command == "config":
        define_config(args.repo_id, system_prompt=args.system_prompt, token=token)
    elif args.command == "add_rows":
        try:
            rows = json.loads(args.rows_json)
            if not isinstance(rows, list):
                raise ValueError("rows_json 必须是对象的 JSON 列表")
            add_rows(
                args.repo_id,
                rows,
                split=args.split,
                template=args.template,
                validate=args.validate,
                token=token,
            )
        except json.JSONDecodeError:
            print("错误: 为 --rows_json 提供的 JSON 无效")
    elif args.command == "quick_setup":
        quick_setup(args.repo_id, template_type=args.template, token=token)
    elif args.command == "stats":
        show_stats(args.repo_id, token=token)
    elif args.command == "list_templates":
        list_available_templates()
