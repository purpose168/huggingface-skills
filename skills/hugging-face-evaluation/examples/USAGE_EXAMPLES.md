# 使用示例

本文档提供了向 HuggingFace 模型卡片添加评估结果的两种方法的实用示例。

## 目录
1. [设置](#setup)
2. [方法 1：从 README 提取](#method-1-extract-from-readme)
3. [方法 2：从 Artificial Analysis 导入](#method-2-import-from-artificial-analysis)
4. [独立脚本与集成脚本](#standalone-vs-integrated)
5. [常见工作流](#common-workflows)


## 设置

### 初始配置

```bash
# 导航到技能目录
cd hf_evaluation_skill

# 安装依赖
uv add huggingface_hub python-dotenv pyyaml requests

# 配置环境变量
cp examples/.env.example .env
# 编辑 .env 文件，添加您的令牌
```

您的 `.env` 文件应包含：
```env
HF_TOKEN=hf_your_write_token_here
AA_API_KEY=aa_your_api_key_here  # 对于 AA 导入是可选的
```

### 验证安装

```bash
cd scripts
python3 test_extraction.py
```


## 方法 1：从 README 提取

从模型现有的 README 中提取评估表格。

### 基本提取

```bash
# 预览将提取的内容（干运行）
python3 scripts/evaluation_manager.py extract-readme \
  --repo-id "meta-llama/Llama-3.3-70B-Instruct" \
  --dry-run
```

### 将提取应用到您的模型

```bash
# 直接提取并更新模型卡片
python3 scripts/evaluation_manager.py extract-readme \
  --repo-id "your-username/your-model-7b"
```

### 自定义任务和数据集名称

```bash
python3 scripts/evaluation_manager.py extract-readme \
  --repo-id "your-username/your-model-7b" \
  --task-type "text-generation" \
  --dataset-name "Standard Benchmarks" \
  --dataset-type "llm_benchmarks"
```

### 创建拉取请求（对于您不拥有的模型）

```bash
python3 scripts/evaluation_manager.py extract-readme \
  --repo-id "organization/community-model" \
  --create-pr
```

### README 格式示例

您的模型 README 应包含如下表格：

```markdown
## 评估结果

| 基准测试     | 分数 |
|---------------|-------|
| MMLU          | 85.2  |
| HumanEval     | 72.5  |
| GSM8K         | 91.3  |
| HellaSwag     | 88.9  |
```


## 方法 2：从 Artificial Analysis 导入

直接从 Artificial Analysis API 获取基准测试分数。

### 集成方法（推荐）

```bash
# 导入 Claude Sonnet 4.5 的分数
python3 scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "your-username/claude-mirror"
```

### 创建拉取请求

```bash
# 创建 PR 而不是直接提交
python3 scripts/evaluation_manager.py import-aa \
  --creator-slug "openai" \
  --model-name "gpt-4" \
  --repo-id "your-username/gpt-4-mirror" \
  --create-pr
```

### 独立脚本

对于简单的一次性导入，使用独立脚本：

```bash
# 导航到 examples 目录
cd examples

# 运行独立脚本
AA_API_KEY="your-key" HF_TOKEN="your-token" \
python3 artificial_analysis_to_hub.py \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "your-username/your-repo"
```

### 查找 Creator Slug 和 Model Name

1. 访问 [Artificial Analysis](https://artificialanalysis.ai/)
2. 导航到您要导入的模型
3. URL 格式为：`https://artificialanalysis.ai/models/{creator-slug}/{model-name}`
4. 或查看他们的 [API 文档](https://artificialanalysis.ai/api)

常见示例：
- Anthropic: `--creator-slug "anthropic" --model-name "claude-sonnet-4"`
- OpenAI: `--creator-slug "openai" --model-name "gpt-4-turbo"`
- Meta: `--creator-slug "meta" --model-name "llama-3-70b"`


## 独立脚本与集成脚本

### 独立脚本特点
- ✓ 简单，单一用途
- ✓ 可以通过 URL 使用 `uv run` 运行
- ✓ 最小依赖
- ✗ 无法从 README 提取
- ✗ 无验证功能
- ✗ 无干运行模式

**使用场景：** 您只需要 AA 导入并想要一个简单的脚本。

### 集成脚本特点
- ✓ 同时支持 README 提取和 AA 导入
- ✓ 验证和显示命令
- ✓ 干运行预览模式
- ✓ 更好的错误处理
- ✓ 与现有评估合并
- ✓ 更灵活的选项

**使用场景：** 您需要完整的评估管理功能。


## 常见工作流

### 工作流 1：带有 README 表格的新模型

您刚刚创建了一个在 README 中包含评估表格的模型。

```bash
# 步骤 1：预览提取
python3 scripts/evaluation_manager.py extract-readme \
  --repo-id "your-username/new-model-7b" \
  --dry-run

# 步骤 2：如果看起来不错，应用提取
python3 scripts/evaluation_manager.py extract-readme \
  --repo-id "your-username/new-model-7b"

# 步骤 3：验证
python3 scripts/evaluation_manager.py validate \
  --repo-id "your-username/new-model-7b"

# 步骤 4：查看结果
python3 scripts/evaluation_manager.py show \
  --repo-id "your-username/new-model-7b"
```

### 工作流 2：在 AA 上进行基准测试的模型

您的模型在 Artificial Analysis 上有新的基准测试。

```bash
# 导入分数并创建 PR 以供审核
python3 scripts/evaluation_manager.py import-aa \
  --creator-slug "your-org" \
  --model-name "your-model" \
  --repo-id "your-org/your-model-hf" \
  --create-pr
```

### 工作流 3：结合两种方法

您同时拥有 README 表格和 AA 分数。

```bash
# 步骤 1：从 README 提取
python3 scripts/evaluation_manager.py extract-readme \
  --repo-id "your-username/hybrid-model"

# 步骤 2：从 AA 导入（将与现有评估合并）
python3 scripts/evaluation_manager.py import-aa \
  --creator-slug "your-org" \
  --model-name "hybrid-model" \
  --repo-id "your-username/hybrid-model"

# 步骤 3：查看合并结果
python3 scripts/evaluation_manager.py show \
  --repo-id "your-username/hybrid-model"
```

### 工作流 4：为社区模型做贡献

通过添加缺失的评估来帮助改进社区模型。

```bash
# 找到在 README 中有评估但没有 model-index 的模型
# 示例：community/awesome-7b

# 创建带有提取的评估的 PR
python3 scripts/evaluation_manager.py extract-readme \
  --repo-id "community/awesome-7b" \
  --create-pr

# GitHub 将通知仓库所有者
# 他们可以审查并合并您的 PR
```

### 工作流 5：批量处理

一次更新多个模型。

```bash
# 创建仓库列表
cat > models.txt << EOF
your-org/model-1-7b
your-org/model-2-13b
your-org/model-3-70b
EOF

# 处理每个仓库
while read repo_id; do
  echo "Processing $repo_id..."
  python3 scripts/evaluation_manager.py extract-readme \
    --repo-id "$repo_id"
done < models.txt
```

### 工作流 6：自动更新（CI/CD）

使用 GitHub Actions 设置自动评估更新。

```yaml
# .github/workflows/update-evals.yml
name: 每周更新评估
on:
  schedule:
    - cron: '0 0 * * 0'  # 每周日
  workflow_dispatch:  # 手动触发

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: 设置 Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.13'

      - name: 安装依赖
        run: |
          pip install huggingface-hub python-dotenv pyyaml requests

      - name: 从 Artificial Analysis 更新
        env:
          AA_API_KEY: ${{ secrets.AA_API_KEY }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python scripts/evaluation_manager.py import-aa \
            --creator-slug "${{ vars.AA_CREATOR_SLUG }}" \
            --model-name "${{ vars.AA_MODEL_NAME }}" \
            --repo-id "${{ github.repository }}" \
            --create-pr
```


## 验证和确认

### 检查当前评估

```bash
python3 scripts/evaluation_manager.py show \
  --repo-id "your-username/your-model"
```

### 验证格式

```bash
python3 scripts/evaluation_manager.py validate \
  --repo-id "your-username/your-model"
```

### 在 HuggingFace UI 中查看

更新后，访问：
```
https://huggingface.co/your-username/your-model
```

评估小部件应自动显示您的分数。


## 故障排除示例

### 问题：未找到表格

```bash
# 检查您的 README 中存在哪些表格
python3 scripts/evaluation_manager.py extract-readme \
  --repo-id "your-username/your-model" \
  --dry-run

# 如果没有输出，请确保您的 README 中有带数字分数的 markdown 表格
```

### 问题：AA 模型未找到

```bash
# 验证创建者和模型 slugs
# 直接检查 AA 网站 URL 或 API
curl -H "x-api-key: $AA_API_KEY" \
  https://artificialanalysis.ai/api/v2/data/llms/models | jq
```

### 问题：令牌权限错误

```bash
# 验证您的令牌具有写入权限
# 在以下位置生成新令牌：https://huggingface.co/settings/tokens
# 确保启用了 "Write" 范围
```


## 提示和最佳实践

1. **始终先干运行**：使用 `--dry-run` 预览更改
2. **对他人的仓库使用 PR**：对于您不拥有的仓库，始终使用 `--create-pr`
3. **更新后验证**：运行 `validate` 确保格式正确
4. **保持评估最新**：为 AA 分数设置自动更新
5. **记录来源**：工具会自动添加来源归因
6. **检查 UI**：始终验证评估小部件显示正确

## 获取帮助

```bash
# 一般帮助
python3 scripts/evaluation_manager.py --help

# 命令特定帮助
python3 scripts/evaluation_manager.py extract-readme --help
python3 scripts/evaluation_manager.py import-aa --help
```

有关问题或疑问，请参考：
- `../SKILL.md` - 完整文档
- `../README.md` - 故障排除指南
- `../QUICKSTART.md` - 快速入门指南

