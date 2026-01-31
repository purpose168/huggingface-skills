#!/usr/bin/env bash

# Hugging Face模型卡元数据提取脚本
# 功能：通过Hugging Face CLI获取模型卡，并提取YAML前置元数据（frontmatter）
# 作者：Hugging Face团队
# 创建日期：2024年

# 设置严格的错误处理模式
# set -e：遇到任何命令返回非零退出码时立即退出脚本
# set -u：使用未定义的变量时立即退出脚本
# set -o pipefail：管道中的任何命令失败都会导致整个管道失败
set -euo pipefail

# 显示帮助信息的函数
show_help() {
    cat << 'USAGE'
通过Hugging Face CLI获取模型卡并汇总前置元数据。

使用方法：
  hf_model_card_frontmatter.sh [模型ID ...]
  cat ids.txt | hf_model_card_frontmatter.sh

说明：
  通过`hf download`命令下载每个模型的README.md文件，提取YAML前置元数据，
  并以每行一个JSON对象（NDJSON格式）输出关键字段。
  如果设置了HF_TOKEN环境变量，会将其传递给hf CLI使用。

输出字段：
  id, license, pipeline_tag, library_name, tags, language,
  new_version, has_extra_gated_prompt

示例：
  hf_model_card_frontmatter.sh openai/gpt-oss-120b
  cat ids.txt | hf_model_card_frontmatter.sh | jq -s '.'
  hf_model_card_frontmatter.sh meta-llama/Meta-Llama-3-8B \
    | jq -s 'map({id, license, has_extra_gated_prompt})'
USAGE
}

# 检查是否请求帮助信息
# ${1:-}：获取第一个参数，如果未设置则返回空字符串
# --help：标准的帮助选项
if [[ "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

# 检查hf CLI是否已安装
# command -v：检查命令是否存在
# >/dev/null 2>&1：将标准输出和标准错误都重定向到空设备（不显示）
if ! command -v hf >/dev/null 2>&1; then
    echo "错误：需要hf CLI但未安装" >&2
    exit 1
fi

# 检查python3是否已安装
if ! command -v python3 >/dev/null 2>&1; then
    echo "错误：需要python3但未安装" >&2
    exit 1
fi

# 准备token参数数组
# 如果设置了HF_TOKEN环境变量，则将其作为参数传递给hf CLI
token_args=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    token_args=(--token "$HF_TOKEN")
fi

# 创建临时目录用于存储下载的文件
# mktemp -d：创建一个唯一的临时目录
tmp_dir=$(mktemp -d)

# 清理函数：删除临时目录
cleanup() {
    rm -rf "$tmp_dir"
}

# 设置退出陷阱：脚本退出时自动执行清理函数
# trap cleanup EXIT：确保无论脚本如何退出都会执行清理
trap cleanup EXIT

# 发送错误信息的函数
# 参数1：模型ID
# 参数2：错误消息
emit_error() {
    local model_id="$1"
    local message="$2"
    # 使用Python生成JSON格式的错误输出
    python3 - << 'PY' "$model_id" "$message"
import json
import sys

model_id = sys.argv[1]
message = sys.argv[2]
print(json.dumps({"id": model_id, "error": message}))
PY
}

# 解析README.md文件的函数
# 参数1：模型ID
# 参数2：README.md文件路径
parse_readme() {
    local model_id="$1"
    local readme_path="$2"

    # 使用环境变量传递参数给Python脚本
    # 这样可以避免在Python脚本中处理shell变量转义问题
    MODEL_ID="$model_id" README_PATH="$readme_path" python3 - << 'PY'
import json
import os
import sys

# 从环境变量获取参数
model_id = os.environ.get("MODEL_ID", "")
readme_path = os.environ.get("README_PATH", "")

# 尝试读取README.md文件
try:
    with open(readme_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
except OSError:
    # 文件读取失败，输出错误信息
    print(json.dumps({"id": model_id, "error": "readme_missing"}))
    sys.exit(0)

# 提取YAML前置元数据（frontmatter）
# YAML前置元数据通常位于两个---之间
frontmatter = []
in_block = False
for line in lines:
    if line.strip() == "---":
        if in_block:
            break
        in_block = True
        continue
    if in_block:
        frontmatter.append(line)

# 如果没有找到前置元数据，输出错误
if not frontmatter:
    print(json.dumps({"id": model_id, "error": "frontmatter_missing"}))
    sys.exit(0)

# 解析YAML前置元数据
key = None
out = {}

for line in frontmatter:
    stripped = line.strip()
    # 跳过空行和注释行
    if not stripped or line.lstrip().startswith("#"):
        continue

    # 处理键值对行（格式：key: value）
    if ":" in line and not line.lstrip().startswith("- "):
        key_candidate, value = line.split(":", 1)
        key_candidate = key_candidate.strip()
        value = value.strip()
        # 验证键名是否有效（只包含字母、数字、下划线或连字符）
        if key_candidate and all(c.isalnum() or c in "_-" for c in key_candidate):
            key = key_candidate
            # 处理多行值标记（YAML语法）
            if value in ("|", "|-", ">", ">-") or value == "":
                out[key] = None
                continue
            # 处理数组值（格式：[item1, item2, ...]）
            if value.startswith("[") and value.endswith("]"):
                items = [v.strip() for v in value.strip("[]").split(",") if v.strip()]
                out[key] = items
            else:
                out[key] = value
            continue

    # 处理列表项（格式：- item）
    if line.lstrip().startswith("- ") and key:
        item = line.strip()[2:]
        if key not in out or out[key] is None:
            out[key] = []
        if isinstance(out[key], list):
            out[key].append(item)

# 构建输出结果，只包含需要的字段
result = {
    "id": model_id,
    "license": out.get("license"),
    "pipeline_tag": out.get("pipeline_tag"),
    "library_name": out.get("library_name"),
    "tags": out.get("tags", []),
    "language": out.get("language", []),
    "new_version": out.get("new_version"),
    "has_extra_gated_prompt": "extra_gated_prompt" in out,
}

# 输出JSON格式的结果
print(json.dumps(result))
PY
}

# 处理单个模型ID的函数
# 参数：模型ID
process_id() {
    local model_id="$1"

    # 跳过空模型ID
    if [[ -z "$model_id" ]]; then
        return 0
    fi

    # 创建安全的本地目录名（将斜杠替换为下划线）
    # tr '/' '_'：将所有斜杠字符替换为下划线
    local safe_id
    safe_id=$(printf '%s' "$model_id" | tr '/' '_')
    local local_dir="$tmp_dir/$safe_id"

    # 使用hf CLI下载模型的README.md文件
    # hf download：下载Hugging Face模型文件
    # --repo-type model：指定仓库类型为模型
    # --local-dir：指定本地下载目录
    # "${token_args[@]}"：传递token参数数组
    # >/dev/null 2>&1：隐藏所有输出
    if ! hf download "$model_id" README.md --repo-type model --local-dir "$local_dir" "${token_args[@]}" >/dev/null 2>&1; then
        emit_error "$model_id" "download_failed"
        return 0
    fi

    # 检查README.md文件是否存在
    local readme_path="$local_dir/README.md"
    if [[ ! -f "$readme_path" ]]; then
        emit_error "$model_id" "readme_missing"
        return 0
    fi

    # 解析README.md文件并输出结果
    parse_readme "$model_id" "$readme_path"
}

# 如果提供了命令行参数，则处理每个模型ID
# $#：命令行参数的数量
if [[ $# -gt 0 ]]; then
    for model_id in "$@"; do
        process_id "$model_id"
    done
    exit 0
fi

# 如果没有命令行参数且标准输入是终端，则显示帮助信息
# -t 0：检查文件描述符0（标准输入）是否为终端
if [[ -t 0 ]]; then
    show_help
    exit 1
fi

# 从标准输入读取模型ID（每行一个）
# IFS=：禁用字段分隔符
# -r：禁用反斜杠转义
# read -r：读取一行输入
while IFS= read -r model_id; do
    process_id "$model_id"
done
