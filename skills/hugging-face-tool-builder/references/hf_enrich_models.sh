#!/usr/bin/env bash

# HuggingFace模型信息丰富化脚本
# 用途:从HuggingFace API获取模型的详细元数据信息
# 作者:Shell翻译注释专家
# 创建日期:2026-01-30

# 设置严格的错误处理选项
# set -e: 任何命令返回非零退出状态时立即退出脚本
# set -u: 使用未设置的变量时报错并退出
# set -o pipefail: 管道中任何命令失败都会导致整个管道返回失败状态
set -euo pipefail

# 显示帮助信息的函数
show_help() {
    cat << 'USAGE'
从标准输入读取模型ID,每行输出一个JSON对象(NDJSON格式)。

使用方法:
  hf_enrich_models.sh [模型ID ...]
  cat ids.txt | hf_enrich_models.sh
  baseline_hf_api.sh 50 | jq -r '.[].id' | hf_enrich_models.sh

描述:
  读取以换行符分隔的模型ID,并获取每个模型的基本元数据。
  输出包含id、downloads、likes、pipeline_tag、tags的NDJSON格式数据。
  如果设置了HF_TOKEN环境变量,则使用该令牌进行身份验证。

示例:
  hf_enrich_models.sh gpt2 distilbert-base-uncased
  baseline_hf_api.sh 50 | jq -r '.[].id' | hf_enrich_models.sh | jq -s 'sort_by(.downloads)'
  HF_TOKEN=your_token hf_enrich_models.sh microsoft/DialoGPT-medium
USAGE
}

# 检查是否请求帮助信息
# ${1:-} 表示使用第一个参数,如果未设置则为空
if [[ "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

# 检查jq命令是否可用
# command -v: 检查命令是否存在
# >/dev/null 2>&1: 将标准输出和标准错误都重定向到空设备(不显示输出)
if ! command -v jq >/dev/null 2>&1; then
    echo "错误: 需要安装jq但未找到" >&2
    # >&2: 将错误信息重定向到标准错误输出
    exit 1
fi

# 初始化HTTP请求头数组
headers=()
# 如果设置了HF_TOKEN环境变量,则添加授权头
# -n: 检查字符串是否非空
# ${HF_TOKEN:-}: 使用HF_TOKEN变量,如果未设置则为空
if [[ -n "${HF_TOKEN:-}" ]]; then
    # -H: 添加HTTP请求头
    # Bearer令牌用于HuggingFace API的身份验证
    headers=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

# 发出错误信息的函数
# 参数1: 模型ID
# 参数2: 错误消息
emit_error() {
    local model_id="$1"
    local message="$2"
    # jq -cn: 创建新的JSON对象,不读取输入(-c表示紧凑输出)
    # --arg: 定义jq变量
    # 构建包含id和error字段的JSON对象
    jq -cn --arg id "$model_id" --arg error "$message" '{id: $id, error: $error}'
}

# 处理单个模型ID的函数
# 参数: 模型ID
process_id() {
    local model_id="$1"

    # 如果模型ID为空,直接返回
    # -z: 检查字符串是否为空
    if [[ -z "$model_id" ]]; then
        return 0
    fi

    # 构建HuggingFace API的URL
    local url="https://huggingface.co/api/models/${model_id}"
    local response
    # 使用curl发送HTTP请求获取模型信息
    # -s: 静默模式,不显示进度信息
    # "${headers[@]}": 展开headers数组中的所有元素
    # 2>/dev/null: 将标准错误重定向到空设备
    # || true: 即使命令失败也继续执行(防止set -e导致脚本退出)
    response=$(curl -s "${headers[@]}" "$url" 2>/dev/null || true)

    # 如果响应为空,发出错误信息
    if [[ -z "$response" ]]; then
        emit_error "$model_id" "request_failed"
        return 0
    fi

    # 验证响应是否为有效的JSON
    # jq -e: 如果输出为null或false则返回退出状态1
    # <<<"$response": 将response字符串作为输入传递给jq
    if ! jq -e . >/dev/null 2>&1 <<<"$response"; then
        emit_error "$model_id" "invalid_json"
        return 0
    fi

    # 检查响应中是否包含错误字段(表示模型不存在)
    if jq -e '.error' >/dev/null 2>&1 <<<"$response"; then
        emit_error "$model_id" "not_found"
        return 0
    fi

    # 提取并格式化所需的模型信息
    # jq -c: 紧凑输出(单行JSON)
    # --arg id: 定义jq变量id
    # // 运算符: 如果左侧为null或false,则使用右侧的默认值
    jq -c --arg id "$model_id" '{
        id: (.id // $id),              # 模型ID,如果API返回的id为空则使用传入的id
        downloads: (.downloads // 0),  # 下载次数,默认为0
        likes: (.likes // 0),          # 点赞数,默认为0
        pipeline_tag: (.pipeline_tag // "unknown"),  # 管道标签,默认为"unknown"
        tags: (.tags // [])            # 标签列表,默认为空数组
    }' <<<"$response" 2>/dev/null || emit_error "$model_id" "parse_failed"
}

# 如果提供了命令行参数,处理每个参数作为模型ID
# $# 表示传递给脚本的参数个数
# -gt: 大于
if [[ $# -gt 0 ]]; then
    for model_id in "$@"; do
        # "$@" 表示所有命令行参数
        process_id "$model_id"
    done
    exit 0
fi

# 如果标准输入是终端(交互式模式),显示帮助信息
# -t 0: 检查文件描述符0(标准输入)是否为终端
if [[ -t 0 ]]; then
    show_help
    exit 1
fi

# 从标准输入逐行读取模型ID并处理
# IFS=: 设置内部字段分隔符为空(保留空白字符)
# read -r: 原始读取,不处理反斜杠转义
# model_id: 存储读取的行到变量model_id
while IFS= read -r model_id; do
    process_id "$model_id"
done
