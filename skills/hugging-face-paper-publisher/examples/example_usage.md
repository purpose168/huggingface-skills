# 示例用法: HF Paper Publisher 技能

本文档演示了在 Hugging Face Hub 上发布研究论文的常见工作流程。

## 示例 1: 索引现有的 arXiv 论文

如果您已经在 arXiv 上发表了论文,并希望使其在 Hugging Face 上可被发现:

```bash
# 检查论文是否存在  # 使用 arXiv ID 查询论文是否已被索引到 Hugging Face
python scripts/paper_manager.py check --arxiv-id "2301.12345"

# 索引论文  # 将 arXiv 论文添加到 Hugging Face 论文数据库
python scripts/paper_manager.py index --arxiv-id "2301.12345"

# 获取论文信息  # 查询论文的详细信息,包括 Hugging Face URL 和 arXiv URL
python scripts/paper_manager.py info --arxiv-id "2301.12345"
```

预期输出:
```json
{
  "exists": true,  # 论文已存在于 Hugging Face
  "url": "https://huggingface.co/papers/2301.12345",  # Hugging Face 论文页面链接
  "arxiv_id": "2301.12345",  # arXiv 论文标识符
  "arxiv_url": "https://arxiv.org/abs/2301.12345"  # arXiv 原始论文链接
}
```

## 示例 2: 将论文链接到您的模型

索引论文后,将其链接到您的模型仓库:

```bash
# 链接单个论文  # 将一篇论文关联到指定的模型仓库
python scripts/paper_manager.py link \
  --repo-id "username/my-awesome-model" \  # 模型仓库的完整 ID (用户名/模型名)
  --repo-type "model" \  # 仓库类型: model、dataset 或 space
  --arxiv-id "2301.12345"  # 要链接的 arXiv 论文 ID

# 链接多个论文  # 一次性将多篇论文关联到模型仓库,用逗号分隔
python scripts/paper_manager.py link \
  --repo-id "username/my-awesome-model" \
  --repo-type "model" \
  --arxiv-ids "2301.12345,2302.67890"  # 多个 arXiv ID,用逗号分隔
```

此操作将执行以下步骤:
1. 下载模型的 README.md 文件
2. 添加或更新 YAML 前置元数据
3. 插入带链接的论文引用
4. 上传更新后的 README 文件
5. Hub 自动创建 `arxiv:2301.12345` 标签

## 示例 3: 将论文链接到数据集

对数据集执行相同的流程:

```bash
python scripts/paper_manager.py link \
  --repo-id "username/my-dataset" \  # 数据集仓库 ID
  --repo-type "dataset" \  # 仓库类型为 dataset
  --arxiv-id "2301.12345" \  # arXiv 论文 ID
  --citation "$(cat citation.bib)"  # 可选:直接提供 BibTeX 格式的引用信息
```

## 示例 4: 创建新的研究文章

从模板生成研究论文:

```bash
# 使用标准模板创建  # 标准学术论文模板,包含完整的论文结构
python scripts/paper_manager.py create \
  --template "standard" \  # 模板类型: standard、modern 或 ml-report
  --title "Efficient Fine-Tuning of Large Language Models" \  # 论文标题
  --authors "Jane Doe, John Smith" \  # 作者列表,用逗号分隔
  --abstract "We propose a novel approach to fine-tuning..." \  # 论文摘要
  --output "paper.md"  # 输出文件路径

# 使用现代模板创建  # 现代风格的论文模板,视觉效果更佳
python scripts/paper_manager.py create \
  --template "modern" \
  --title "Vision Transformers for Medical Imaging" \
  --output "medical_vit_paper.md"

# 创建机器学习实验报告  # 专门用于 ML 实验结果报告的模板
python scripts/paper_manager.py create \
  --template "ml-report" \
  --title "BERT Fine-tuning Experiment Results" \
  --output "bert_experiment_report.md"
```

## 示例 5: 生成引用

获取论文的格式化引用:

```bash
# BibTeX 格式  # 生成 BibTeX 格式的引用,适合学术写作
python scripts/paper_manager.py citation \
  --arxiv-id "2301.12345" \  # arXiv 论文 ID
  --format "bibtex"  # 引用格式: bibtex、apa、mla 等
```

输出:
```bibtex
@article{arxiv2301_12345,  # BibTeX 条目标识符
  title={Efficient Fine-Tuning of Large Language Models},  # 论文标题
  author={Doe, Jane and Smith, John},  # 作者列表
  journal={arXiv preprint arXiv:2301.12345},  # 期刊/预印本信息
  year={2023}  # 发表年份
}
```

## 示例 6: 完整工作流程 - 新论文

从论文创建到发布的完整流程:

```bash
# 步骤 1: 创建研究文章  # 使用模板生成论文框架
python scripts/paper_manager.py create \
  --template "modern" \
  --title "Novel Architecture for Multimodal Learning" \
  --authors "Alice Chen, Bob Kumar" \
  --output "multimodal_paper.md"

# 步骤 2: 编辑论文 (使用您喜欢的编辑器)  # 使用文本编辑器完善论文内容
# vim multimodal_paper.md

# 步骤 3: 提交到 arXiv (外部流程)  # 访问 arxiv.org 上传论文
# 上传到 arxiv.org,接收 arXiv ID: 2312.99999

# 步骤 4: 在 Hugging Face 上索引  # 将论文添加到 Hugging Face 论文库
python scripts/paper_manager.py index --arxiv-id "2312.99999"

# 步骤 5: 链接到您的模型/数据集  # 将论文关联到相关资源
python scripts/paper_manager.py link \
  --repo-id "alice/multimodal-model-v1" \
  --repo-type "model" \
  --arxiv-id "2312.99999"

python scripts/paper_manager.py link \
  --repo-id "alice/multimodal-dataset" \
  --repo-type "dataset" \
  --arxiv-id "2312.99999"

# 步骤 6: 为 README 生成引用  # 生成 BibTeX 引用并保存到文件
python scripts/paper_manager.py citation \
  --arxiv-id "2312.99999" \
  --format "bibtex" > citation.bib
```

## 示例 7: 批量链接论文

将多篇论文链接到多个仓库:

```bash
#!/bin/bash  # Bash 脚本声明

# 论文列表  # 定义要处理的 arXiv ID 数组
PAPERS=("2301.12345" "2302.67890" "2303.11111")

# 模型列表  # 定义要关联的模型仓库数组
MODELS=("username/model-a" "username/model-b" "username/model-c")

# 将每篇论文链接到每个模型  # 双重循环实现批量关联
for paper in "${PAPERS[@]}"; do  # 遍历所有论文
  for model in "${MODELS[@]}"; do  # 遍历所有模型
    echo "Linking $paper to $model..."  # 显示当前处理进度
    python scripts/paper_manager.py link \
      --repo-id "$model" \
      --repo-type "model" \
      --arxiv-id "$paper"
  done
done
```

## 示例 8: 使用论文信息更新模型卡

获取论文信息并手动更新模型卡:

```bash
# 获取论文信息  # 查询论文的详细信息并以文本格式输出
python scripts/paper_manager.py info \
  --arxiv-id "2301.12345" \
  --format "text" > paper_info.txt  # 保存到文件

# 查看信息  # 显示论文详细信息
cat paper_info.txt

# 手动整合到您的模型卡中,或使用 link 命令自动处理
```

## 示例 9: 搜索和发现论文

```bash
# 搜索论文 (打开浏览器)  # 在 Hugging Face 上搜索相关论文
python scripts/paper_manager.py search \
  --query "transformer attention mechanism"  # 搜索关键词
```

## 示例 10: 使用 tfrere 的模板

此技能与 [tfrere 的研究文章模板](https://huggingface.co/spaces/tfrere/research-article-template) 配合使用:

```bash
# 1. 使用 tfrere 的 Space 创建精美的网页论文
# 访问: https://huggingface.co/spaces/tfrere/research-article-template

# 2. 将论文内容导出为 Markdown 格式

# 3. 提交到 arXiv  # 获取 arXiv ID

# 4. 使用此技能进行索引和链接
python scripts/paper_manager.py index --arxiv-id "YOUR_ARXIV_ID"  # 替换为实际的 arXiv ID
python scripts/paper_manager.py link \
  --repo-id "your-username/your-model" \  # 替换为您的仓库 ID
  --arxiv-id "YOUR_ARXIV_ID"
```

## 示例 11: 错误处理

```bash
# 在链接之前检查论文是否存在  # 使用条件判断避免错误
if python scripts/paper_manager.py check --arxiv-id "2301.12345" | grep -q '"exists": true'; then
  echo "Paper exists, proceeding with link..."  # 论文存在,继续链接
  python scripts/paper_manager.py link \
    --repo-id "username/model" \
    --arxiv-id "2301.12345"
else
  echo "Paper doesn't exist, indexing first..."  # 论文不存在,先索引
  python scripts/paper_manager.py index --arxiv-id "2301.12345"
  python scripts/paper_manager.py link \
    --repo-id "username/model" \
    --arxiv-id "2301.12345"
fi
```

## 示例 12: CI/CD 集成

添加到您的 `.github/workflows/update-paper.yml`:

```yaml
name: Update Paper Links  # 工作流名称

on:  # 触发条件
  push:
    branches: [main]  # 推送到 main 分支时触发
  workflow_dispatch:  # 支持手动触发

jobs:
  update:
    runs-on: ubuntu-latest  # 运行环境
    steps:
      - uses: actions/checkout@v3  # 检出代码

      - name: Set up Python  # 设置 Python 环境
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'  # Python 版本

      - name: Install dependencies  # 安装依赖
        run: |
          pip install huggingface_hub pyyaml requests python-dotenv  # 安装必要的 Python 包

      - name: Link paper to model  # 将论文链接到模型
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}  # 从 GitHub Secrets 获取 Hugging Face Token
        run: |
          python scripts/paper_manager.py link \
            --repo-id "${{ github.repository_owner }}/model-name" \  # 动态获取仓库所有者
            --repo-type "model" \
            --arxiv-id "2301.12345"
```

## 提示和最佳实践

1. **在索引之前始终检查论文是否存在** 以避免不必要的操作
2. **使用有意义的提交消息** 在将论文链接到仓库时
3. **在模型卡中包含完整引用** 以确保正确的归属
4. **将论文链接到所有相关工件** (模型、数据集、Space)
5. **生成 BibTeX 引用** 以方便他人引用
6. **在您的 HF 配置文件设置中保持论文可见性更新**
7. **在研究组内一致使用模板** 以保持统一风格
8. **对论文进行版本控制** 与代码一起管理

## 故障排除

### 索引后找不到论文

```bash
# 直接访问 URL 以触发索引  # 有时需要手动访问页面来触发索引
open "https://huggingface.co/papers/2301.12345"

# 等待几秒钟,然后再次检查  # 索引可能需要一些时间
python scripts/paper_manager.py check --arxiv-id "2301.12345"
```

### 链接时权限被拒绝

```bash
# 验证您的令牌具有写入权限  # 检查 HF_TOKEN 是否正确设置
echo $HF_TOKEN

# 如果缺失则设置令牌  # 设置环境变量
export HF_TOKEN="your_token_here"

# 或使用 .env 文件  # 将令牌保存在 .env 文件中
echo "HF_TOKEN=your_token_here" > .env
```

### arXiv ID 格式问题

```bash
# 脚本处理各种格式:  # 支持多种 arXiv ID 格式
python scripts/paper_manager.py check --arxiv-id "2301.12345"  # 标准 ID 格式
python scripts/paper_manager.py check --arxiv-id "arxiv:2301.12345"  # 带前缀格式
python scripts/paper_manager.py check --arxiv-id "https://arxiv.org/abs/2301.12345"  # 完整 URL 格式

# 所有格式都是等效的,将被规范化为标准格式
```

## 后续步骤

- 探索 [Paper Pages 文档](https://huggingface.co/docs/hub/en/paper-pages)
- 查看 [tfrere 的研究模板](https://huggingface.co/spaces/tfrere/research-article-template)
- 浏览 [HF 上的论文](https://huggingface.co/papers)
- 了解 [模型卡](https://huggingface.co/docs/hub/en/model-cards)
