#!/usr/bin/env bash

# -*- coding: utf-8 -*-
# Hugging Face 模型论文工具（带身份验证功能）
# 功能：获取 Hugging Face 模型引用的论文，支持通过 HF_TOKEN 环境变量进行身份验证
# 作者：Shell翻译注释专家
# 创建日期：2026-01-30

# 设置严格的错误处理选项
# set -e: 任何命令返回非零退出码时立即退出脚本
# set -u: 使用未定义的变量时报错
# set -o pipefail: 管道命令中任何命令失败都会导致整个管道失败
set -euo pipefail

# 显示帮助信息的函数
show_help() {
    cat << EOF
Hugging Face 模型论文工具（带身份验证功能）

本工具用于获取 Hugging Face 模型引用的论文。
支持通过 HF_TOKEN 环境变量进行身份验证。

使用方法：
    $0 [选项]

选项：
    MODEL_ID            指定要分析的模型（例如：microsoft/DialoGPT-medium）
    --trending [N]      显示前 N 个热门模型的论文（默认：5）
    --help              显示此帮助信息

环境变量：
    HF_TOKEN            Hugging Face API 令牌（可选，用于私有模型）

示例：
    # 获取特定模型的论文
    $0 microsoft/DialoGPT-medium

    # 使用身份验证获取论文
    HF_TOKEN=your_token_here $0 your-private-model

    # 获取前 3 个热门模型的论文
    $0 --trending 3

EOF
}

# 进行身份验证 API 调用的函数
# 参数：
#   $1 - API URL 地址
# 返回值：
#   API 响应的 JSON 数据
hf_api_call() {
    local url="$1"  # API URL
    local headers=()  # HTTP 请求头数组

    # 如果设置了 HF_TOKEN 环境变量，则添加身份验证头
    # [[ -n "${HF_TOKEN:-}" ]]：检查 HF_TOKEN 变量是否非空
    # ${HF_TOKEN:-}：如果 HF_TOKEN 未设置，则返回空字符串而不报错
    if [[ -n "${HF_TOKEN:-}" ]]; then
        # headers+=()：向数组添加元素
        # -H：添加 HTTP 请求头
        headers+=(-H "Authorization: Bearer $HF_TOKEN")
    fi

    # 使用 curl 发送 HTTP 请求
    # -s：静默模式，不显示进度条或错误信息
    # "${headers[@]}"：展开数组中的所有元素
    # 2>/dev/null：将标准错误输出重定向到空设备（不显示错误）
    # || echo '{"error": "Network error"}'：如果 curl 失败，返回错误 JSON
    curl -s "${headers[@]}" "$url" 2>/dev/null || echo '{"error": "Network error"}'
}

# 从文本中提取论文信息的函数
# 参数：
#   $1 - 要分析的文本内容
#   $2 - 标题文本
extract_papers() {
    local text="$1"  # 文本内容
    local title="$2"  # 标题

    echo "$title"  # 输出标题

    # 查找 ArXiv URL
    # grep -oE：只输出匹配的部分，使用扩展正则表达式
    # 'https?://arxiv\.org/[^[:space:]\])]+'：匹配 ArXiv URL
    # [^[:space:]\])]：匹配非空白字符、非右括号、非右方括号
    # head -5：只取前 5 个匹配结果
    local arxiv_urls=$(echo "$text" | grep -oE 'https?://arxiv\.org/[^[:space:]\])]+' | head -5)
    if [[ -n "$arxiv_urls" ]]; then
        echo "ArXiv 论文："
        # sed 's/^/  • /'：在每行开头添加 "  • "
        echo "$arxiv_urls" | sed 's/^/  • /'
    fi

    # 查找 DOI URL
    # 'https?://doi\.org/[^[:space:]\])]+'：匹配 DOI URL
    local doi_urls=$(echo "$text" | grep -oE 'https?://doi\.org/[^[:space:]\])]+' | head -3)
    if [[ -n "$doi_urls" ]]; then
        echo "DOI 论文："
        echo "$doi_urls" | sed 's/^/  • /'
    fi

    # 查找格式为 YYYY.NNNNN 的 ArXiv ID
    # 'arXiv:[0-9]{4}\.[0-9]{4,5}'：匹配 arXiv:2023.12345 格式的 ID
    local arxiv_ids=$(echo "$text" | grep -oE 'arXiv:[0-9]{4}\.[0-9]{4,5}' | head -5)
    if [[ -n "$arxiv_ids" ]]; then
        echo "ArXiv ID："
        echo "$arxiv_ids" | sed 's/^/  • /'
    fi

    # 检查论文提及
    # grep -qi：不区分大小写搜索，不显示匹配行
    # "paper\|publication\|citation"：匹配 paper、publication 或 citation
    # \|：或运算符
    if echo "$text" | grep -qi "paper\|publication\|citation"; then
        # -A1 -B1：显示匹配行及其前后各 1 行
        # head -6：只取前 6 行
        local paper_mentions=$(echo "$text" | grep -i -A1 -B1 "paper\|publication" | head -6)
        if [[ -n "$paper_mentions" ]]; then
            echo "论文提及："
            echo "$paper_mentions" | sed 's/^/  /'  # 在每行开头添加两个空格
        fi
    fi

    # 如果没有找到任何论文信息
    if [[ -z "$arxiv_urls" && -z "$doi_urls" && -z "$arxiv_ids" ]]; then
        echo "在模型卡片中未找到论文"
    fi
}

# 获取模型论文的函数
# 参数：
#   $1 - 模型 ID（例如：microsoft/DialoGPT-medium）
get_model_papers() {
    local model_id="$1"  # 模型 ID

    echo "=== $model_id ==="

    # 从 API 获取模型信息（带身份验证）
    local api_url="https://huggingface.co/api/models/$model_id"
    local response=$(hf_api_call "$api_url")

    # 检查 API 响应中是否包含错误
    # grep -q：静默模式，不输出匹配行，只返回退出状态
    if echo "$response" | grep -q '"error"'; then
        echo "错误：无法获取模型 '$model_id'"
        # 检查是否未设置 HF_TOKEN
        if [[ -z "${HF_TOKEN:-}" ]]; then
            echo "注意：这可能是一个私有模型。尝试设置 HF_TOKEN 环境变量。"
        fi
        return 1  # 返回非零退出码表示失败
    fi

    # 解析基本信息
    # jq：命令行 JSON 处理器
    # -r：输出原始字符串（不带引号）
    # '.downloads // 0'：获取 downloads 字段，如果不存在则返回 0
    local downloads=$(echo "$response" | jq -r '.downloads // 0')
    local likes=$(echo "$response" | jq -r '.likes // 0')
    echo "下载量：$downloads | 点赞数：$likes"

    # 获取模型卡片（README.md 文件）
    local card_url="https://huggingface.co/$model_id/raw/main/README.md"
    # curl -s：静默模式
    # 2>/dev/null：将错误输出重定向到空设备
    # || echo ""：如果 curl 失败，返回空字符串
    local card_content=$(curl -s "$card_url" 2>/dev/null || echo "")

    if [[ -n "$card_content" ]]; then
        # 如果成功获取模型卡片，提取论文信息
        extract_papers "$card_content" "来自模型卡片的论文："
    else
        echo "无法获取模型卡片"
    fi

    # 检查标签中的 arxiv 引用
    # jq -r '.tags[]'：获取 tags 数组中的所有元素
    # 2>/dev/null：将错误输出重定向到空设备
    # grep arxiv：过滤包含 arxiv 的标签
    # || true：即使 grep 失败也返回成功（避免管道失败）
    local arxiv_tag=$(echo "$response" | jq -r '.tags[]' 2>/dev/null | grep arxiv || true)
    if [[ -n "$arxiv_tag" ]]; then
        echo "来自标签的 ArXiv：$arxiv_tag"
    fi

    echo  # 输出空行分隔不同模型的信息
}

# 获取热门模型的函数
# 参数：
#   $1 - 要获取的模型数量（默认：5）
get_trending_models() {
    local limit="${1:-5}"  # 设置默认值为 5
    # ${1:-5}：如果第一个参数未设置或为空，则使用 5

    echo "正在获取前 $limit 个热门模型..."

    # 构建 API URL
    local trending_url="https://huggingface.co/api/trending?type=model&limit=$limit"
    local response=$(hf_api_call "$trending_url")

    # 处理响应并获取每个模型的论文
    # jq -r '.recentlyTrending[] | .repoData.id'：提取模型 ID
    # head -"$limit"：只取前 limit 个结果
    # while read -r model_id：逐行读取模型 ID
    # -r：不处理反斜杠转义
    echo "$response" | jq -r '.recentlyTrending[] | .repoData.id' | head -"$limit" | while read -r model_id; do
        if [[ -n "$model_id" ]]; then
            get_model_papers "$model_id"
        fi
    done
}

# 主程序入口
# $#：传递给脚本的参数数量
# -eq：等于
if [[ $# -eq 0 ]]; then
    echo "错误：未提供参数"
    show_help
    exit 1  # 退出码 1 表示错误
fi

# 检查第一个参数
if [[ "$1" == "--help" ]]; then
    # 如果是 --help，显示帮助信息
    show_help
    exit 0  # 退出码 0 表示成功
elif [[ "$1" == "--trending" ]]; then
    # 如果是 --trending，获取热门模型
    # [[ -n "${2:-}" ]]：检查第二个参数是否存在且非空
    # [[ "$2" =~ ^[0-9]+$ ]]：检查第二个参数是否为纯数字
    # =~：正则表达式匹配
    # ^[0-9]+$：匹配一个或多个数字
    if [[ -n "${2:-}" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
        get_trending_models "$2"
    else
        get_trending_models 5  # 默认获取 5 个热门模型
    fi
else
    # 否则，将第一个参数作为模型 ID 处理
    get_model_papers "$1"
fi
