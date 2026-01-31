---
name: hugging-face-paper-publisher
description: 在Hugging Face Hub上发布和管理研究论文。支持创建论文页面、将论文链接到模型/数据集、声明作者身份以及生成专业的基于markdown的研究文章。
---

# 概述
此技能为AI工程师和研究人员提供了在Hugging Face Hub上发布、管理和链接研究论文的综合工具。它简化了从论文创建到发布的工作流程，包括与arXiv的集成、模型/数据集链接以及作者身份管理。

## 与HF生态系统的集成
- **论文页面**：在Hugging Face Hub上索引和发现论文
- **arXiv集成**：从arXiv ID自动索引论文
- **模型/数据集链接**：通过元数据将论文与相关工件连接
- **作者身份验证**：声明和验证论文作者身份
- **研究文章模板**：生成专业的、现代的科学论文

# 版本
1.0.0

# 依赖项
- huggingface_hub>=0.26.0
- pyyaml>=6.0.3
- requests>=2.32.5
- markdown>=3.5.0
- python-dotenv>=1.2.1

# 核心功能

## 1. 论文页面管理
- **索引论文**：从arXiv将论文添加到Hugging Face
- **声明作者身份**：验证和声明已发布论文的作者身份
- **管理可见性**：控制哪些论文出现在您的个人资料中
- **论文发现**：在HF生态系统中查找和探索论文

## 2. 将论文链接到工件
- **模型卡片**：向模型元数据添加论文引用
- **数据集卡片**：通过README将论文链接到数据集
- **自动标记**：Hub自动生成arxiv:<PAPER_ID>标签
- **引用管理**：维护正确的归属和引用

## 3. 研究文章创建
- **Markdown模板**：生成专业的论文格式
- **现代设计**：干净、可读的研究文章布局
- **动态目录**：自动生成目录
- **章节结构**：标准科学论文组织
- **LaTeX数学公式**：支持方程和技术符号

## 4. 元数据管理
- **YAML前置matter**：正确的模型/数据集卡片元数据
- **引用跟踪**：跨仓库维护论文引用
- **版本控制**：跟踪论文更新和修订
- **多论文支持**：将多篇论文链接到单个工件

# 使用说明

此技能包括`scripts/`中的Python脚本用于论文发布操作。

### 前提条件
- 安装依赖项：`uv add huggingface_hub pyyaml requests markdown python-dotenv`
- 设置`HF_TOKEN`环境变量，使用具有写入访问权限的令牌
- 激活虚拟环境：`source .venv/bin/activate`

> **所有路径都相对于包含此SKILL.md文件的目录。**
> 在运行任何脚本之前，先`cd`到该目录或使用完整路径。

### 方法1：从arXiv索引论文

从arXiv将论文添加到Hugging Face论文页面。

**基本用法：**
```bash
uv run scripts/paper_manager.py index \
  --arxiv-id "2301.12345"
```

**检查论文是否存在：**
```bash
uv run scripts/paper_manager.py check \
  --arxiv-id "2301.12345"
```

**直接访问URL：**
您也可以直接访问`https://huggingface.co/papers/{arxiv-id}`来索引论文。

### 方法2：将论文链接到模型/数据集

使用正确的YAML元数据向模型或数据集README添加论文引用。

**添加到模型卡片：**
```bash
uv run scripts/paper_manager.py link \
  --repo-id "username/model-name" \
  --repo-type "model" \
  --arxiv-id "2301.12345"
```

**添加到数据集卡片：**
```bash
uv run scripts/paper_manager.py link \
  --repo-id "username/dataset-name" \
  --repo-type "dataset" \
  --arxiv-id "2301.12345"
```

**添加多篇论文：**
```bash
uv run scripts/paper_manager.py link \
  --repo-id "username/model-name" \
  --repo-type "model" \
  --arxiv-ids "2301.12345,2302.67890,2303.11111"
```

**使用自定义引用：**
```bash
uv run scripts/paper_manager.py link \
  --repo-id "username/model-name" \
  --repo-type "model" \
  --arxiv-id "2301.12345" \
  --citation "$(cat citation.txt)"
```

#### 链接如何工作

当您向模型或数据集README添加arXiv论文链接时：
1. Hub从链接中提取arXiv ID
2. 自动添加标签`arxiv:<PAPER_ID>`到仓库
3. 用户可以点击标签查看论文页面
4. 论文页面显示引用此论文的所有模型/数据集
5. 可以通过过滤器和搜索发现论文

### 方法3：声明作者身份

验证您在Hugging Face上发布的论文的作者身份。

**开始声明流程：**
```bash
uv run scripts/paper_manager.py claim \
  --arxiv-id "2301.12345" \
  --email "your.email@institution.edu"
```

**手动流程：**
1. 导航到您的论文页面：`https://huggingface.co/papers/{arxiv-id}`
2. 在作者列表中找到您的名字
3. 点击您的名字并选择"声明作者身份"
4. 等待管理团队验证

**检查作者身份状态：**
```bash
uv run scripts/paper_manager.py check-authorship \
  --arxiv-id "2301.12345"
```

### 方法4：管理论文可见性

控制哪些已验证的论文出现在您的公开个人资料中。

**列出您的论文：**
```bash
uv run scripts/paper_manager.py list-my-papers
```

**切换可见性：**
```bash
uv run scripts/paper_manager.py toggle-visibility \
  --arxiv-id "2301.12345" \
  --show true
```

**在设置中管理：**
导航到您的账户设置 → 论文部分，为每篇论文切换"显示在个人资料上"。

### 方法5：创建研究文章

使用现代模板生成专业的基于markdown的研究论文。

**从模板创建：**
```bash
uv run scripts/paper_manager.py create \
  --template "standard" \
  --title "Your Paper Title" \
  --output "paper.md"
```

**可用模板：**
- `standard` - 传统科学论文结构
- `modern` - 受Distill启发的干净、网络友好的格式
- `arxiv` - arXiv风格格式
- `ml-report` - 机器学习实验报告

**生成完整论文：**
```bash
uv run scripts/paper_manager.py create \
  --template "modern" \
  --title "Fine-Tuning Large Language Models with LoRA" \
  --authors "Jane Doe, John Smith" \
  --abstract "$(cat abstract.txt)" \
  --output "paper.md"
```

**转换为HTML：**
```bash
uv run scripts/paper_manager.py convert \
  --input "paper.md" \
  --output "paper.html" \
  --style "modern"
```

### 论文模板结构

**标准研究论文章节：**
```markdown
---
title: Your Paper Title
authors: Jane Doe, John Smith
affiliations: University X, Lab Y
date: 2025-01-15
arxiv: 2301.12345
tags: [machine-learning, nlp, fine-tuning]
---

# 摘要
论文的简要总结...

# 1. 引言
背景和动机...

# 2. 相关工作
之前的研究和上下文...

# 3. 方法论
方法和实现...

# 4. 实验
设置、数据集和过程...

# 5. 结果
发现和分析...

# 6. 讨论
解释和含义...

# 7. 结论
总结和未来工作...

# 参考文献
```

**现代模板功能：**
- 动态目录
- 响应式网页设计
- 代码语法高亮
- 交互式图表
- 数学公式渲染（LaTeX）
- 引用管理
- 作者附属机构链接

### 命令参考

**索引论文：**
```bash
uv run scripts/paper_manager.py index --arxiv-id "2301.12345"
```

**链接到仓库：**
```bash
uv run scripts/paper_manager.py link \
  --repo-id "username/repo-name" \
  --repo-type "model|dataset|space" \
  --arxiv-id "2301.12345" \
  [--citation "Full citation text"] \
  [--create-pr]
```

**声明作者身份：**
```bash
uv run scripts/paper_manager.py claim \
  --arxiv-id "2301.12345" \
  --email "your.email@edu"
```

**管理可见性：**
```bash
uv run scripts/paper_manager.py toggle-visibility \
  --arxiv-id "2301.12345" \
  --show true|false
```

**创建研究文章：**
```bash
uv run scripts/paper_manager.py create \
  --template "standard|modern|arxiv|ml-report" \
  --title "Paper Title" \
  [--authors "Author1, Author2"] \
  [--abstract "Abstract text"] \
  [--output "filename.md"]
```

**将Markdown转换为HTML：**
```bash
uv run scripts/paper_manager.py convert \
  --input "paper.md" \
  --output "paper.html" \
  [--style "modern|classic"]
```

**检查论文状态：**
```bash
uv run scripts/paper_manager.py check --arxiv-id "2301.12345"
```

**列出您的论文：**
```bash
uv run scripts/paper_manager.py list-my-papers
```

**搜索论文：**
```bash
uv run scripts/paper_manager.py search --query "transformer attention"
```

### YAML元数据格式

将论文链接到模型或数据集时，需要正确的YAML前置matter：

**模型卡片示例：**
```yaml
---
language:
  - en
license: apache-2.0
tags:
  - text-generation
  - transformers
  - llm
library_name: transformers
---

# Model Name

This model is based on the approach described in [Our Paper](https://arxiv.org/abs/2301.12345).

## Citation

```bibtex
@article{doe2023paper,
  title={Your Paper Title},
  author={Doe, Jane and Smith, John},
  journal={arXiv preprint arXiv:2301.12345},
  year={2023}
}
```
```

**数据集卡片示例：**
```yaml
---
language:
  - en
license: cc-by-4.0
task_categories:
  - text-generation
  - question-answering
size_categories:
  - 10K<n<100K
---

# Dataset Name

Dataset introduced in [Our Paper](https://arxiv.org/abs/2301.12345).

For more details, see the [paper page](https://huggingface.co/papers/2301.12345).
```

Hub会自动从这些链接中提取arXiv ID并创建`arxiv:2301.12345`标签。

### 集成示例

**工作流1：发布新研究**
```bash
# 1. 创建研究文章
uv run scripts/paper_manager.py create \
  --template "modern" \
  --title "Novel Fine-Tuning Approach" \
  --output "paper.md"

# 2. 用您的内容编辑paper.md

# 3. 提交到arXiv（外部过程）
# 上传到arxiv.org，获取arXiv ID

# 4. 在Hugging Face上索引
uv run scripts/paper_manager.py index --arxiv-id "2301.12345"

# 5. 链接到您的模型
uv run scripts/paper_manager.py link \
  --repo-id "your-username/your-model" \
  --repo-type "model" \
  --arxiv-id "2301.12345"

# 6. 声明作者身份
uv run scripts/paper_manager.py claim \
  --arxiv-id "2301.12345" \
  --email "your.email@edu"
```

**工作流2：链接现有论文**
```bash
# 1. 检查论文是否存在
uv run scripts/paper_manager.py check --arxiv-id "2301.12345"

# 2. 如需要，索引论文
uv run scripts/paper_manager.py index --arxiv-id "2301.12345"

# 3. 链接到多个仓库
uv run scripts/paper_manager.py link \
  --repo-id "username/model-v1" \
  --repo-type "model" \
  --arxiv-id "2301.12345"

uv run scripts/paper_manager.py link \
  --repo-id "username/training-data" \
  --repo-type "dataset" \
  --arxiv-id "2301.12345"

uv run scripts/paper_manager.py link \
  --repo-id "username/demo-space" \
  --repo-type "space" \
  --arxiv-id "2301.12345"
```

**工作流3：更新模型添加论文引用**
```bash
# 1. 获取当前README
huggingface-cli download username/model-name README.md

# 2. 添加论文链接
uv run scripts/paper_manager.py link \
  --repo-id "username/model-name" \
  --repo-type "model" \
  --arxiv-id "2301.12345" \
  --citation "Full citation for the paper"

# 脚本将：
# - 添加YAML元数据（如果缺失）
# - 在README中插入arXiv链接
# - 添加格式化引用
# - 保留现有内容
```

### 最佳实践

1. **论文索引**
   - 论文在arXiv发布后立即索引
   - 在模型/数据集中包含完整引用信息
   - 在相关仓库中使用一致的论文引用

2. **元数据管理**
   - 为所有模型/数据集卡片添加YAML前置matter
   - 包含正确的许可信息
   - 使用相关任务类别和领域进行标记

3. **作者身份**
   - 在您列为作者的论文上声明作者身份
   - 使用机构电子邮件地址进行验证
   - 保持论文可见性设置更新

4. **仓库链接**
   - 将论文链接到所有相关的模型、数据集和Spaces
   - 在README描述中包含论文上下文
   - 添加BibTeX引用以便轻松参考

5. **研究文章**
   - 在项目内一致使用模板
   - 在论文中包含代码和数据链接
   - 生成网络友好的HTML版本以共享

### 高级用法

**批量链接论文：**
```bash
# 将多篇论文链接到一个仓库
for arxiv_id in "2301.12345" "2302.67890" "2303.11111"; do
  uv run scripts/paper_manager.py link \
    --repo-id "username/model-name" \
    --repo-type "model" \
    --arxiv-id "$arxiv_id"
done
```

**提取论文信息：**
```bash
# 从arXiv获取论文元数据
uv run scripts/paper_manager.py info \
  --arxiv-id "2301.12345" \
  --format "json"
```

**生成引用：**
```bash
# 创建BibTeX引用
uv run scripts/paper_manager.py citation \
  --arxiv-id "2301.12345" \
  --format "bibtex"
```

**验证链接：**
```bash
# 检查仓库中的所有论文链接
uv run scripts/paper_manager.py validate \
  --repo-id "username/model-name" \
  --repo-type "model"
```

### 错误处理

- **论文未找到**：arXiv ID不存在或尚未索引
- **权限被拒绝**：HF_TOKEN缺乏对仓库的写入访问权限
- **YAML无效**：README前置matter中的元数据格式错误
- **作者身份失败**：电子邮件与论文作者记录不匹配
- **已被声明**：另一位用户已声明作者身份
- **速率限制**：短时间内API请求过多

### 故障排除

**问题**："在Hugging Face上找不到论文"
- **解决方案**：访问`hf.co/papers/{arxiv-id}`以触发索引

**问题**："作者身份声明未验证"
- **解决方案**：等待管理员审查或联系HF支持并提供证明

**问题**："arXiv标签未出现"
- **解决方案**：确保README包含正确的arXiv URL格式

**问题**："无法链接到仓库"
- **解决方案**：验证HF_TOKEN具有写入权限

**问题**："模板渲染错误"
- **解决方案**：检查markdown语法和YAML前置matter格式

### 资源和参考

- **Hugging Face论文页面**：[hf.co/papers](https://huggingface.co/papers)
- **模型卡片指南**：[hf.co/docs/hub/model-cards](https://huggingface.co/docs/hub/en/model-cards)
- **数据集卡片指南**：[hf.co/docs/hub/datasets-cards](https://huggingface.co/docs/hub/en/datasets-cards)
- **研究文章模板**：[tfrere/research-article-template](https://huggingface.co/spaces/tfrere/research-article-template)
- **arXiv格式指南**：[arxiv.org/help/submit](https://arxiv.org/help/submit)

### 与tfrere的研究模板集成

此技能补充[tfrere的研究文章模板](https://huggingface.co/spaces/tfrere/research-article-template)，提供：

- 自动论文索引工作流
- 仓库链接功能
- 元数据管理工具
- 引用生成实用程序

您可以使用tfrere的模板进行写作，然后使用此技能在Hugging Face Hub上发布和链接论文。

### 常见模式

**模式1：新论文发布**
```bash
# 写作 → 发布 → 索引 → 链接
uv run scripts/paper_manager.py create --template modern --output paper.md
# （提交到arXiv）
uv run scripts/paper_manager.py index --arxiv-id "2301.12345"
uv run scripts/paper_manager.py link --repo-id "user/model" --arxiv-id "2301.12345"
```

**模式2：发现现有论文**
```bash
# 搜索 → 检查 → 链接
uv run scripts/paper_manager.py search --query "transformers"
uv run scripts/paper_manager.py check --arxiv-id "2301.12345"
uv run scripts/paper_manager.py link --repo-id "user/model" --arxiv-id "2301.12345"
```

**模式3：作者作品集管理**
```bash
# 声明 → 验证 → 组织
uv run scripts/paper_manager.py claim --arxiv-id "2301.12345"
uv run scripts/paper_manager.py list-my-papers
uv run scripts/paper_manager.py toggle-visibility --arxiv-id "2301.12345" --show true
```

### API集成

**Python脚本示例：**
```python
from scripts.paper_manager import PaperManager

pm = PaperManager(hf_token="your_token")

# 索引论文
pm.index_paper("2301.12345")

# 链接到模型
pm.link_paper(
    repo_id="username/model",
    repo_type="model",
    arxiv_id="2301.12345",
    citation="Full citation text"
)

# 检查状态
status = pm.check_paper("2301.12345")
print(status)
```

### 未来增强

计划在将来版本中添加的功能：
- 支持非arXiv论文（会议论文集、期刊）
- 从DOI自动格式化引用
- 论文比较和版本控制工具
- 协作论文写作功能
- 与LaTeX工作流程集成
- 自动提取图表和表格
- 论文指标和影响跟踪