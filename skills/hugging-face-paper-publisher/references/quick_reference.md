# 快速参考指南

## 核心命令

### 论文索引
```bash
# 从 arXiv 索引论文  # 从 arXiv 获取论文信息并建立索引
python scripts/paper_manager.py index --arxiv-id "2301.12345"

# 检查论文是否存在  # 验证指定 arXiv ID 的论文是否已被索引
python scripts/paper_manager.py check --arxiv-id "2301.12345"
```

### 链接论文
```bash
# 链接到模型  # 将论文与 Hugging Face 模型仓库关联
python scripts/paper_manager.py link \
  --repo-id "username/model" \
  --repo-type "model" \
  --arxiv-id "2301.12345"

# 链接到数据集  # 将论文与 Hugging Face 数据集仓库关联
python scripts/paper_manager.py link \
  --repo-id "username/dataset" \
  --repo-type "dataset" \
  --arxiv-id "2301.12345"

# 批量链接多篇论文  # 同时将一个仓库与多篇论文关联
python scripts/paper_manager.py link \
  --repo-id "username/model" \
  --repo-type "model" \
  --arxiv-ids "2301.12345,2302.67890"
```

### 创建论文
```bash
# 标准模板  # 使用传统学术论文模板创建论文
python scripts/paper_manager.py create \
  --template "standard" \
  --title "Paper Title" \
  --output "paper.md"

# 现代模板  # 使用现代网页友好格式创建论文
python scripts/paper_manager.py create \
  --template "modern" \
  --title "Paper Title" \
  --authors "Author1, Author2" \
  --abstract "Abstract text" \
  --output "paper.md"

# ML 报告模板  # 使用机器学习实验报告模板
python scripts/paper_manager.py create \
  --template "ml-report" \
  --title "Experiment Report" \
  --output "report.md"

# arXiv 风格模板  # 使用 arXiv 期刊格式
python scripts/paper_manager.py create \
  --template "arxiv" \
  --title "Paper Title" \
  --output "paper.md"
```

### 引用生成
```bash
# 生成 BibTeX 格式引用  # 为指定论文生成 BibTeX 引用格式
python scripts/paper_manager.py citation \
  --arxiv-id "2301.12345" \
  --format "bibtex"
```

### 论文信息查询
```bash
# JSON 格式输出  # 以 JSON 格式显示论文详细信息
python scripts/paper_manager.py info \
  --arxiv-id "2301.12345" \
  --format "json"

# 文本格式输出  # 以纯文本格式显示论文详细信息
python scripts/paper_manager.py info \
  --arxiv-id "2301.12345" \
  --format "text"
```

## URL 格式规范

### Hugging Face 论文页面
- 查看论文页面: `https://huggingface.co/papers/{arxiv-id}`  # 通过 arXiv ID 访问 Hugging Face 论文页面
- 示例: `https://huggingface.co/papers/2301.12345`

### arXiv
- 摘要页面: `https://arxiv.org/abs/{arxiv-id}`  # 访问论文摘要页面
- PDF 下载: `https://arxiv.org/pdf/{arxiv-id}.pdf`  # 直接下载论文 PDF
- 示例: `https://arxiv.org/abs/2301.12345`

## YAML 元数据格式

### 模型卡片 (Model Card)
```yaml
---  # YAML 前置元数据分隔符
language:  # 支持的语言列表
  - en
license: apache-2.0  # 开源许可证类型
tags:  # 模型标签，用于分类和搜索
  - text-generation  # 文本生成
  - transformers  # 基于 Transformers 框架
library_name: transformers  # 依赖的库
---
```

### 数据集卡片 (Dataset Card)
```yaml
---  # YAML 前置元数据分隔符
language:  # 支持的语言列表
  - en
license: cc-by-4.0  # 知识共享许可证
task_categories:  # 任务类别
  - text-generation  # 文本生成任务
size_categories:  # 数据集规模类别
  - 10K<n<100K  # 数据集大小在 1 万到 10 万之间
---
```

## arXiv ID 格式

以下所有格式均有效：
- `2301.12345`  # 标准 arXiv ID 格式（年份.序号）
- `arxiv:2301.12345`  # 带前缀的格式
- `https://arxiv.org/abs/2301.12345`  # arXiv 摘要页面 URL
- `https://arxiv.org/pdf/2301.12345.pdf`  # arXiv PDF 下载 URL

## 环境配置

### 设置访问令牌
```bash
export HF_TOKEN="your_token"  # 设置 Hugging Face 访问令牌环境变量
```

### 或使用 .env 文件
```bash
echo "HF_TOKEN=your_token" > .env  # 将令牌保存到 .env 配置文件中
```

## 常见工作流程

### 1. 索引与链接
```bash
# 第一步：索引论文  # 从 arXiv 获取论文信息
python scripts/paper_manager.py index --arxiv-id "2301.12345"

# 第二步：链接到仓库  # 将论文与模型或数据集关联
python scripts/paper_manager.py link --repo-id "user/model" --arxiv-id "2301.12345"
```

### 2. 创建与发布
```bash
# 第一步：创建论文  # 使用模板创建新论文
python scripts/paper_manager.py create --template "modern" --title "Title" --output "paper.md"

# 第二步：编辑论文  # 手动编辑 paper.md 文件
# Edit paper.md

# 第三步：提交到 arXiv  # 将论文提交到 arXiv 并获取 ID
# Submit to arXiv → get ID

# 第四步：索引新论文  # 使用新获取的 arXiv ID 索引论文
python scripts/paper_manager.py index --arxiv-id "NEW_ID"

# 第五步：链接到仓库  # 将新论文与模型关联
python scripts/paper_manager.py link --repo-id "user/model" --arxiv-id "NEW_ID"
```

### 3. 批量链接
```bash
# 使用循环批量链接多篇论文到同一仓库  # 遍历多个 arXiv ID 并逐个链接
for id in "2301.12345" "2302.67890"; do
  python scripts/paper_manager.py link --repo-id "user/model" --arxiv-id "$id"
done
```

## 故障排除

### 论文未找到
访问 `https://huggingface.co/papers/{arxiv-id}` 触发索引  # 手动访问论文页面可触发 Hugging Face 的自动索引

### 权限被拒绝
检查 `HF_TOKEN` 是否已设置且具有写入权限  # 确保令牌有效且拥有仓库写入权限

### arXiv API 错误
稍等片刻后重试 - arXiv 有速率限制  # arXiv API 对请求频率有限制，过快请求会被拒绝

## 使用技巧

1. 链接前务必检查论文是否存在  # 避免链接无效的 arXiv ID
2. 使用模板以保持一致性  # 统一使用模板可确保论文格式规范
3. 在模型卡片中包含完整引用  # 提供完整的文献引用信息
4. 将论文链接到所有相关资源  # 确保模型、数据集等资源都关联到对应论文
5. 保持引用信息更新  # 及时更新论文引用和链接信息

## 可用模板

- `standard` - 传统学术论文格式  # 适用于传统学术期刊投稿
- `modern` - 网页友好格式（Distill 风格）  # 适合在线阅读和展示
- `arxiv` - arXiv 期刊格式  # 符合 arXiv 投稿规范
- `ml-report` - 机器学习实验文档  # 专门用于记录 ML 实验结果

## 文件位置说明

- 脚本文件: `scripts/paper_manager.py`  # 核心管理脚本
- 模板文件: `templates/*.md`  # 各种论文模板
- 示例文件: `examples/example_usage.md`  # 使用示例
- 本指南: `references/quick_reference.md`  # 快速参考指南

## 获取帮助

```bash
# 查看命令帮助  # 显示所有可用命令和选项
python scripts/paper_manager.py --help

# 查看子命令帮助  # 显示特定子命令的详细用法
python scripts/paper_manager.py link --help
```

## 额外资源

- [完整文档](../SKILL.md)  # 详细的技能说明文档
- [使用示例](../examples/example_usage.md)  # 更多实际应用案例
- [Hugging Face 论文页面](https://huggingface.co/papers)  # 论文浏览平台
- [tfrere 的模板](https://huggingface.co/spaces/tfrere/research-article-template)  # 推荐的研究论文模板
