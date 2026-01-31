#!/bin/bash

# 在 Hugging Face 上查找与研究论文关联的模型
# 用途：./find_models_by_paper.sh [arXiv_id|search_term]
# 可选：设置 HF_TOKEN 环境变量以访问私有/受限模型

# set -e：遇到错误立即退出脚本（任何命令返回非零状态码时）
# 这是一种良好的Shell脚本实践，可以防止错误在脚本中继续传播
set -e

# 输出颜色定义（ANSI转义码）
# 这些转义序列用于在终端中显示彩色文本
# \033[0;31m：红色（用于错误信息）
RED='\033[0;31m'
# \033[0;32m：绿色（用于成功信息）
GREEN='\033[0;32m'
# \033[1;33m：黄色加粗（用于警告和提示信息）
YELLOW='\033[1;33m'
# \033[0;34m：蓝色（用于一般信息）
BLUE='\033[0;34m'
# \033[0m：重置所有颜色属性（恢复默认文本颜色）
NC='\033[0m' # No Color（无颜色）

# 帮助信息函数
# 显示脚本的使用说明和示例
show_help() {
    # echo -e：启用转义序列解释（如颜色代码）
    echo -e "${BLUE}在 Hugging Face 上查找与研究论文关联的模型${NC}"
    echo ""
    echo -e "${YELLOW}使用方法：${NC}"
    echo "  $0 [选项] [搜索词|arXiv_ID]"
    echo ""
    echo -e "${YELLOW}选项：${NC}"
    echo "  --help    显示此帮助信息"
    echo "  --token   使用 HF_TOKEN 环境变量（如果已设置）"
    echo ""
    echo -e "${YELLOW}环境变量：${NC}"
    echo "  HF_TOKEN  可选：用于访问私有/受限模型的 Hugging Face 令牌"
    echo ""
    echo -e "${YELLOW}示例：${NC}"
    echo "  $0 1910.01108                    # 通过 arXiv ID 搜索"
    echo "  $0 distilbert                     # 通过模型名称搜索"
    echo "  $0 transformer                    # 通过关键词搜索"
    echo "  HF_TOKEN=your_token $0 1910.01108  # 使用身份验证"
    echo ""
    echo -e "${YELLOW}说明：${NC}"
    echo "此脚本查找与研究论文关联的 Hugging Face 模型。"
    echo "它搜索在标签中包含 arXiv ID 或在元数据中提及论文的模型。"
    echo ""
    echo -e "${YELLOW}注意事项：${NC}"
    echo "• 对于公开模型，HF_TOKEN 是可选的"
    echo "• 使用 HF_TOKEN 访问私有仓库或受限模型"
    echo "• HF_TOKEN 可以为大量使用提供更高的速率限制"
}

# 解析命令行参数
USE_TOKEN=false
POSITIONAL_ARGS=()

# while循环：遍历所有命令行参数
# $#：参数的总数量
# $1：第一个参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --token)
            USE_TOKEN=true
            # shift：将参数位置向左移动一位（丢弃$1，原来的$2变成新的$1）
            shift
            ;;
        -*)
            echo -e "${RED}未知选项：$1${NC}"
            show_help
            exit 1
            ;;
        *)
            # 将非选项参数添加到位置参数数组中
            # +=：向数组追加元素
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# 将位置参数重新设置为脚本参数
# "${POSITIONAL_ARGS[@]}"：展开数组中的所有元素
set -- "${POSITIONAL_ARGS[@]}"

# 检查是否提供了搜索词
if [[ $# -eq 0 ]]; then
    echo -e "${RED}错误：请提供搜索词或 arXiv ID${NC}"
    echo -e "使用 ${YELLOW}$0 --help${NC} 查看使用信息"
    exit 1
fi

SEARCH_TERM="$1"

# 如果 HF_TOKEN 可用，则设置认证头
# -n：检查变量是否非空
# &&：逻辑与运算符（前一个命令成功时才执行后一个）
# ||：逻辑或运算符（前一个命令失败时才执行后一个）
if [[ -n "$HF_TOKEN" ]] && [[ "$USE_TOKEN" == true || -n "$HF_TOKEN" ]]; then
    # -H：添加HTTP请求头
    # Bearer：Bearer令牌认证方式
    AUTH_HEADER="-H \"Authorization: Bearer $HF_TOKEN\""
    echo -e "${BLUE}使用 HF_TOKEN 进行身份验证${NC}"
else
    AUTH_HEADER=""
    if [[ -n "$HF_TOKEN" ]]; then
        echo -e "${YELLOW}找到 HF_TOKEN 但未使用（添加 --token 标志以使用）${NC}"
    fi
fi

# 检查输入是否看起来像 arXiv ID（格式：YYYY.NNNNN 或 YYYY.NNNNNNN）
# =~：正则表达式匹配运算符
# ^[0-9]{4}\.[0-9]{4,7}$：匹配4位数字、点号、4-7位数字的模式
if [[ "$SEARCH_TERM" =~ ^[0-9]{4}\.[0-9]{4,7}$ ]]; then
    echo -e "${BLUE}正在查找与 arXiv 论文关联的模型：$SEARCH_TERM${NC}"
    # %3A 是冒号（:）的URL编码
    SEARCH_QUERY="arxiv%3A$SEARCH_TERM"
    IS_ARXIV_SEARCH=true
else
    echo -e "${BLUE}正在查找与以下内容相关的模型：$SEARCH_TERM${NC}"
    SEARCH_QUERY="$SEARCH_TERM"
    IS_ARXIV_SEARCH=false
fi

# 从标签中提取 arXiv ID 的函数
# 参数：tags - 包含标签的JSON数组字符串
# 返回：提取出的arXiv ID列表
extract_arxiv_ids() {
    local tags="$1"
    # jq：命令行JSON处理工具
    # -r：输出原始字符串（不添加引号）
    # .[]：遍历数组中的每个元素
    # select(. | startswith("arxiv:"))：选择以"arxiv:"开头的元素
    # split(":")[1]：按冒号分割并取第二部分（即arXiv ID）
    # 2>/dev/null：将错误输出重定向到空设备（不显示错误）
    # || true：如果命令失败，返回成功状态（避免脚本退出）
    echo "$tags" | jq -r '.[] | select(. | startswith("arxiv:")) | split(":")[1]' 2>/dev/null || true
}

# 从 arXiv ID 获取论文标题的函数
# 参数：arxiv_id - arXiv论文ID
# 返回：论文标题
get_paper_title() {
    local arxiv_id="$1"
    # 尝试从 Hugging Face 标签获取论文标题（如果可用）
    # 这是一个简化的方法 - 实际应用中，您可能需要调用 arXiv API
    echo "论文标题 (arXiv:$arxiv_id)"
}

# 搜索模型
API_URL="https://huggingface.co/api/models"
echo -e "${YELLOW}正在搜索 Hugging Face API...${NC}"

# 构建带认证的curl命令（如果可用）
CURL_CMD="curl -s $AUTH_HEADER \"$API_URL?search=$SEARCH_QUERY&limit=50\""
echo -e "${BLUE}API 查询：$API_URL?search=$SEARCH_QUERY&limit=50${NC}"

# 执行API调用
# curl：命令行HTTP客户端工具
# -s：静默模式（不显示进度或错误信息）
# -H：添加HTTP请求头
if [[ -n "$HF_TOKEN" ]]; then
    RESPONSE=$(curl -s -H "Authorization: Bearer $HF_TOKEN" "$API_URL?search=$SEARCH_QUERY&limit=50" || true)
else
    RESPONSE=$(curl -s "$API_URL?search=$SEARCH_QUERY&limit=50" || true)
fi

# 检查是否获得了有效响应
# -z：检查字符串是否为空
# ==：字符串相等比较
if [[ -z "$RESPONSE" ]] || [[ "$RESPONSE" == "[]" ]]; then
    echo -e "${RED}未找到搜索词对应的模型：$SEARCH_TERM${NC}"
    
    # 如果 arXiv 搜索失败，尝试不使用 arxiv: 前缀进行搜索
    if [[ "$IS_ARXIV_SEARCH" == true ]]; then
        echo -e "${YELLOW}正在尝试不使用 arxiv: 前缀进行更广泛的搜索...${NC}"
        SEARCH_QUERY="$SEARCH_TERM"
        IS_ARXIV_SEARCH=false
        
        if [[ -n "$HF_TOKEN" ]]; then
            RESPONSE=$(curl -s -H "Authorization: Bearer $HF_TOKEN" "$API_URL?search=$SEARCH_QUERY&limit=50" || true)
        else
            RESPONSE=$(curl -s "$API_URL?search=$SEARCH_QUERY&limit=50" || true)
        fi
        
        if [[ -z "$RESPONSE" ]] || [[ "$RESPONSE" == "[]" ]]; then
            echo -e "${RED}仍然没有找到结果。请尝试不同的搜索词。${NC}"
            exit 1
        fi
    else
        exit 1
    fi
fi

# 处理搜索结果
echo -e "${GREEN}找到模型！正在处理结果...${NC}"

# 使用 jq 处理 JSON 响应并查找与论文关联的模型
# jq 是一个强大的命令行 JSON 处理器
MODELS_WITH_PAPERS=$(echo "$RESPONSE" | jq -r '
  .[] |
  select(.id != null) |
  {
    id: .id,
    # 筛选以"arxiv:"开头的标签并用分号连接
    arxiv_tags: [.tags[] | select(. | startswith("arxiv:"))] | join("; "),
    # // 0：如果字段为null，则使用默认值0
    downloads: (.downloads // 0),
    likes: (.likes // 0),
    task: (.pipeline_tag // "unknown"),
    library: (.library_name // "unknown")
  }
  # @base64：将结果编码为 base64（便于在 Shell 中处理）
  | @base64' 2>/dev/null || true)

# 统计结果总数
# jq length：计算数组长度
TOTAL_MODELS=$(echo "$RESPONSE" | jq 'length' 2>/dev/null || echo "0")
# wc -l：统计行数
MODELS_WITH_PAPERS_COUNT=$(echo "$MODELS_WITH_PAPERS" | wc -l)

echo -e "${BLUE}结果摘要：${NC}"
echo -e "  找到的模型总数：$TOTAL_MODELS"
echo -e "  与论文关联的模型数：$MODELS_WITH_PAPERS_COUNT"
echo ""

# 检查是否找到与论文关联的模型
if [[ -z "$MODELS_WITH_PAPERS" ]]; then
    # 如果没有找到论文关联，显示所有匹配的模型
    echo -e "${YELLOW}未找到明确的论文关联。显示所有匹配的模型：${NC}"
    echo "$RESPONSE" | jq -r '
      .[] |
      select(.id != null) |
      "📦 \(.id)
   任务: \(.pipeline_tag // "unknown")
   下载量: \(.downloads // 0)
   点赞数: \(.likes // 0)
   库: \(.library_name // "unknown")
   ---"
    ' 2>/dev/null || echo "解析响应失败"
else
    # 显示与论文关联的模型
    echo -e "${GREEN}与论文关联的模型：${NC}"
    # while read -r：逐行读取输入，-r 表示不处理反斜杠转义
    echo "$MODELS_WITH_PAPERS" | while read -r model_data; do
        if [[ -n "$model_data" ]]; then
            # 解码 base64 并显示格式化输出
            # base64 -d：解码 base64 编码的数据
            echo "$model_data" | base64 -d | jq -r '
                "📄 \(.id)
   arXiv: \(.arxiv_tags)
   任务: \(.task)
   下载量: \(.downloads)
   点赞数: \(.likes)
   库: \(.library)
   ---"
            ' 2>/dev/null || echo "解析模型数据失败"
        fi
    done
fi

# 额外的搜索提示
echo ""
echo -e "${BLUE}搜索提示：${NC}"
echo "• 尝试使用完整的 arXiv ID 搜索（例如：1910.01108）"
echo "• 尝试使用论文标题关键词搜索"
echo "• 尝试使用模型名称搜索"
echo "• 使用 HF_TOKEN 访问私有模型或获得更高的速率限制"
echo ""
echo -e "${BLUE}可尝试的示例：${NC}"
echo "  $0 1910.01108                    # DistilBERT 论文"
echo "  $0 1810.04805                    # BERT 论文" 
echo "  $0 1706.03762                    # Attention is All You Need 论文"
echo "  $0 roberta                       # RoBERTa 模型"
echo "  $0 transformer                   # Transformer 模型"
echo "  HF_TOKEN=your_token $0 1910.01108  # 使用身份验证"
