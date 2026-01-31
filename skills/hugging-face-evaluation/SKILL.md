---
name: hugging-face-evaluation
description: 在Hugging Face模型卡片中添加和管理评估结果。支持从README内容提取评估表格、从Artificial Analysis API导入分数，以及使用vLLM/lighteval运行自定义模型评估。与model-index元数据格式配合使用。
---

# 概述
本技能提供工具，用于向Hugging Face模型卡片添加结构化评估结果。它支持多种添加评估数据的方法：
- 从README内容中提取现有的评估表格
- 从Artificial Analysis导入基准测试分数
- 使用vLLM或accelerate后端（lighteval/inspect-ai）运行自定义模型评估

## 与HF生态系统的集成
- **模型卡片**：更新model-index元数据以集成排行榜
- **Artificial Analysis**：直接API集成，用于基准测试导入
- **Papers with Code**：兼容其model-index规范
- **Jobs**：在Hugging Face Jobs上直接运行评估，集成`uv`
- **vLLM**：用于自定义模型评估的高效GPU推理
- **lighteval**：HuggingFace的评估库，支持vLLM/accelerate后端
- **inspect-ai**：英国AI安全研究所的评估框架

# 版本
1.3.0

# 依赖项

## 核心依赖项
- huggingface_hub>=0.26.0
- markdown-it-py>=3.0.0
- python-dotenv>=1.2.1
- pyyaml>=6.0.3
- requests>=2.32.5
- re (内置)

## 推理提供商评估
- inspect-ai>=0.3.0
- inspect-evals
- openai

## vLLM自定义模型评估（需要GPU）
- lighteval[accelerate,vllm]>=0.6.0
- vllm>=0.4.0
- torch>=2.0.0
- transformers>=4.40.0
- accelerate>=0.30.0

注意：使用`uv run`时，vLLM依赖项会通过PEP 723脚本头部自动安装。

# 重要：使用本技能

## ⚠️ 关键：在创建新PR之前检查现有PR

**在使用`--create-pr`创建任何拉取请求之前，您必须检查是否存在现有的开放PR：**

```bash
uv run scripts/evaluation_manager.py get-prs --repo-id "username/model-name"
```

**如果存在开放PR：**
1. **不要创建新PR** - 这会为维护者创造重复工作
2. **警告用户** 已经存在开放PR
3. **向用户展示** 现有PR的URL，以便他们查看
4. 只有在用户明确确认他们想要创建另一个PR时才继续

这可以防止使用重复的评估PR垃圾信息淹没模型仓库。

---

> **所有路径都是相对于包含此SKILL.md文件的目录**
> 在运行任何脚本之前，首先`cd`到该目录或使用完整路径。


**使用`--help`获取最新的工作流指导。** 适用于普通Python或`uv run`：
```bash
uv run scripts/evaluation_manager.py --help
uv run scripts/evaluation_manager.py inspect-tables --help
uv run scripts/evaluation_manager.py extract-readme --help
```
关键工作流（与CLI帮助匹配）：

1) `get-prs` → 首先检查现有的开放PR
2) `inspect-tables` → 查找表格编号/列
3) `extract-readme --table N` → 默认打印YAML
4) 添加`--apply`（推送）或`--create-pr`写入更改

# 核心功能

## 1. 检查和从README提取评估表格
- **检查表格**：使用`inspect-tables`查看README中的所有表格，包括结构、列和样本行
- **解析Markdown表格**：使用markdown-it-py进行准确解析（忽略代码块和示例）
- **表格选择**：使用`--table N`从特定表格中提取（当存在多个表格时需要）
- **格式检测**：识别常见格式（作为行、列的基准测试，或包含多个模型的比较表格）
- **列匹配**：自动识别模型列/行；优先使用`--model-column-index`（来自inspect输出的索引）。仅当无法使用索引时，才使用`--model-name-override`并提供确切的列标题文本。
- **YAML生成**：将所选表格转换为model-index YAML格式
- **任务类型**：`--task-type`设置model-index输出中的`task.type`字段（例如，`text-generation`，`summarization`）

## 2. 从Artificial Analysis导入
- **API集成**：直接从Artificial Analysis获取基准测试分数
- **自动格式化**：将API响应转换为model-index格式
- **元数据保留**：维护源归属和URL
- **PR创建**：自动创建带有评估更新的拉取请求

## 3. Model-Index管理
- **YAML生成**：创建格式正确的model-index条目
- **合并支持**：向现有模型卡片添加评估，而不覆盖
- **验证**：确保符合Papers with Code规范
- **批处理**：高效处理多个模型

## 4. 在HF Jobs上运行评估（推理提供商）
- **Inspect-AI集成**：使用`inspect-ai`库运行标准评估
- **UV集成**：在HF基础设施上使用临时依赖无缝运行Python脚本
- **零配置**：不需要Dockerfile或Space管理
- **硬件选择**：为评估作业配置CPU或GPU硬件
- **安全执行**：通过CLI传递的密钥安全处理API令牌

## 5. 使用vLLM运行自定义模型评估（新增）

⚠️ **重要：** 这种方法仅适用于安装了`uv`且具有足够GPU内存的设备。
**优势：** 无需使用`hf_jobs()` MCP工具，可以直接在终端中运行脚本
**何时使用：** 用户直接在本地设备上工作且GPU可用时

### 运行脚本前

- 检查脚本路径
- 检查是否安装了uv
- 使用`nvidia-smi`检查GPU是否可用

### 运行脚本

```bash
uv run scripts/train_sft_example.py
```
### 特性

- **vLLM后端**：高性能GPU推理（比标准HF方法快5-10倍）
- **lighteval框架**：HuggingFace的评估库，支持Open LLM Leaderboard任务
- **inspect-ai框架**：英国AI安全研究所的评估库
- **独立或Jobs**：本地运行或提交到HF Jobs基础设施

# 使用说明

本技能包含`scripts/`中的Python脚本以执行操作。

### 前提条件
- 首选：使用`uv run`（PEP 723头部自动安装依赖）
- 或手动安装：`pip install huggingface-hub markdown-it-py python-dotenv pyyaml requests`
- 设置`HF_TOKEN`环境变量，包含写入访问令牌
- 对于Artificial Analysis：设置`AA_API_KEY`环境变量
- 如果安装了`python-dotenv`，会自动加载`.env`文件

### 方法1：从README提取（CLI工作流）

推荐流程（与`--help`匹配）：
```bash
# 1) 检查表格以获取表格编号和列提示
uv run scripts/evaluation_manager.py inspect-tables --repo-id "username/model"

# 2) 提取特定表格（默认打印YAML）
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model" \
  --table 1 \
  [--model-column-index <inspect-tables显示的列索引>] \
  [--model-name-override "<列标题/模型名称>"]  # 如果无法使用索引，请使用确切的标题文本

# 3) 应用更改（推送或PR）
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model" \
  --table 1 \
  --apply       # 直接推送
# 或
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model" \
  --table 1 \
  --create-pr   # 打开PR
```

验证清单：
- 默认打印YAML；应用前与README表格比较。
- 首选`--model-column-index`；如果使用`--model-name-override`，列标题文本必须完全一致。
- 对于转置表格（模型作为行），确保只提取一行。

### 方法2：从Artificial Analysis导入

从Artificial Analysis API获取基准测试分数并将其添加到模型卡片。

**基本用法：**
```bash
AA_API_KEY="your-api-key" uv run scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "username/model-name"
```

**使用环境文件：**
```bash
# 创建.env文件
echo "AA_API_KEY=your-api-key" >> .env
echo "HF_TOKEN=your-hf-token" >> .env

# 运行导入
uv run scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "username/model-name"
```

**创建拉取请求：**
```bash
uv run scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "username/model-name" \
  --create-pr
```

### 方法3：运行评估作业

使用`hf jobs uv run` CLI在Hugging Face基础设施上提交评估作业。

**直接CLI使用：**
```bash
HF_TOKEN=$HF_TOKEN \
hf jobs uv run hf-evaluation/scripts/inspect_eval_uv.py \
  --flavor cpu-basic \
  --secret HF_TOKEN=$HF_TOKEN \
  -- --model "meta-llama/Llama-2-7b-hf" \
     --task "mmlu"
```

**GPU示例（A10G）：**
```bash
HF_TOKEN=$HF_TOKEN \
hf jobs uv run hf-evaluation/scripts/inspect_eval_uv.py \
  --flavor a10g-small \
  --secret HF_TOKEN=$HF_TOKEN \
  -- --model "meta-llama/Llama-2-7b-hf" \
     --task "gsm8k"
```

**Python助手（可选）：**
```bash
uv run scripts/run_eval_job.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --task "mmlu" \
  --hardware "t4-small"
```

### 方法4：使用vLLM运行自定义模型评估

使用vLLM或accelerate后端直接在GPU上评估自定义HuggingFace模型。这些脚本**与推理提供商脚本分开**，在作业的硬件上本地运行模型。

#### 何时使用vLLM评估（vs推理提供商）

| 特性 | vLLM脚本 | 推理提供商脚本 |
|---------|-------------|---------------------------|
| 模型访问 | 任何HF模型 | 具有API端点的模型 |
| 硬件 | 您的GPU（或HF Jobs GPU） | 提供商的基础设施 |
| 成本 | HF Jobs计算成本 | API使用费用 |
| 速度 | vLLM优化 | 取决于提供商 |
| 离线 | 是（下载后） | 否 |

#### 选项A：使用vLLM后端的lighteval

lighteval是HuggingFace的评估库，支持Open LLM Leaderboard任务。

**独立（本地GPU）：**
```bash
# 使用vLLM运行MMLU 5-shot
uv run scripts/lighteval_vllm_uv.py \
  --model meta-llama/Llama-3.2-1B \
  --tasks "leaderboard|mmlu|5"

# 运行多个任务
uv run scripts/lighteval_vllm_uv.py \
  --model meta-llama/Llama-3.2-1B \
  --tasks "leaderboard|mmlu|5,leaderboard|gsm8k|5"

# 使用accelerate后端而不是vLLM
uv run scripts/lighteval_vllm_uv.py \
  --model meta-llama/Llama-3.2-1B \
  --tasks "leaderboard|mmlu|5" \
  --backend accelerate

# 聊天/指令调优模型
uv run scripts/lighteval_vllm_uv.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --tasks "leaderboard|mmlu|5" \
  --use-chat-template
```

**通过HF Jobs：**
```bash
hf jobs uv run scripts/lighteval_vllm_uv.py \
  --flavor a10g-small \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- --model meta-llama/Llama-3.2-1B \
     --tasks "leaderboard|mmlu|5"
```

**lighteval任务格式：**
任务使用格式`suite|task|num_fewshot`：
- `leaderboard|mmlu|5` - 5-shot的MMLU
- `leaderboard|gsm8k|5` - 5-shot的GSM8K
- `lighteval|hellaswag|0` - 零-shot的HellaSwag
- `leaderboard|arc_challenge|25` - 25-shot的ARC-Challenge

**查找可用任务：**
可用lighteval任务的完整列表可在以下位置找到：
https://github.com/huggingface/lighteval/blob/main/examples/tasks/all_tasks.txt

此文件包含所有支持的任务，格式为`suite|task|num_fewshot|0`（末尾的`0`是版本标志，可以忽略）。常见套件包括：
- `leaderboard` - Open LLM Leaderboard任务（MMLU、GSM8K、ARC、HellaSwag等）
- `lighteval` - 其他lighteval任务
- `bigbench` - BigBench任务
- `original` - 原始基准测试任务

要使用列表中的任务，提取`suite|task|num_fewshot`部分（不包含末尾的`0`）并将其传递给`--tasks`参数。例如：
- 从文件：`leaderboard|mmlu|0` → 使用：`leaderboard|mmlu|0`（或更改为`5`表示5-shot）
- 从文件：`bigbench|abstract_narrative_understanding|0` → 使用：`bigbench|abstract_narrative_understanding|0`
- 从文件：`lighteval|wmt14:hi-en|0` → 使用：`lighteval|wmt14:hi-en|0`

可以使用逗号分隔值指定多个任务：`--tasks "leaderboard|mmlu|5,leaderboard|gsm8k|5"`

#### 选项B：使用vLLM后端的inspect-ai

inspect-ai是英国AI安全研究所的评估框架。

**独立（本地GPU）：**
```bash
# 使用vLLM运行MMLU
uv run scripts/inspect_vllm_uv.py \
  --model meta-llama/Llama-3.2-1B \
  --task mmlu

# 使用HuggingFace Transformers后端
uv run scripts/inspect_vllm_uv.py \
  --model meta-llama/Llama-3.2-1B \
  --task mmlu \
  --backend hf

# 多GPU与张量并行
uv run scripts/inspect_vllm_uv.py \
  --model meta-llama/Llama-3.2-70B \
  --task mmlu \
  --tensor-parallel-size 4
```

**通过HF Jobs：**
```bash
hf jobs uv run scripts/inspect_vllm_uv.py \
  --flavor a10g-small \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- --model meta-llama/Llama-3.2-1B \
     --task mmlu
```

**可用的inspect-ai任务：**
- `mmlu` - 大规模多任务语言理解
- `gsm8k` - 小学数学
- `hellaswag` - 常识推理
- `arc_challenge` - AI2推理挑战
- `truthfulqa` - TruthfulQA基准测试
- `winogrande` - Winograd Schema Challenge
- `humaneval` - 代码生成

#### 选项C：Python助手脚本

助手脚本自动选择硬件并简化作业提交：

```bash
# 基于模型大小自动检测硬件
uv run scripts/run_vllm_eval_job.py \
  --model meta-llama/Llama-3.2-1B \
  --task "leaderboard|mmlu|5" \
  --framework lighteval

# 显式硬件选择
uv run scripts/run_vllm_eval_job.py \
  --model meta-llama/Llama-3.2-70B \
  --task mmlu \
  --framework inspect \
  --hardware a100-large \
  --tensor-parallel-size 4

# 使用HF Transformers后端
uv run scripts/run_vllm_eval_job.py \
  --model microsoft/phi-2 \
  --task mmlu \
  --framework inspect \
  --backend hf
```

**硬件推荐：**
| 模型大小 | 推荐硬件 |
|------------|---------------------|
| < 3B参数 | `t4-small` |
| 3B - 13B | `a10g-small` |
| 13B - 34B | `a10g-large` |
| 34B+ | `a100-large` |

### 命令参考

**顶级帮助和版本：**
```bash
uv run scripts/evaluation_manager.py --help
uv run scripts/evaluation_manager.py --version
```

**检查表格（从这里开始）：**
```bash
uv run scripts/evaluation_manager.py inspect-tables --repo-id "username/model-name"
```

**从README提取：**
```bash
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model-name" \
  --table N \
  [--model-column-index N] \
  [--model-name-override "Exact Column Header or Model Name"] \
  [--task-type "text-generation"] \
  [--dataset-name "Custom Benchmarks"] \
  [--apply | --create-pr]
```

**从Artificial Analysis导入：**
```bash
AA_API_KEY=... uv run scripts/evaluation_manager.py import-aa \
  --creator-slug "creator-name" \
  --model-name "model-slug" \
  --repo-id "username/model-name" \
  [--create-pr]
```

**查看/验证：**
```bash
uv run scripts/evaluation_manager.py show --repo-id "username/model-name"
uv run scripts/evaluation_manager.py validate --repo-id "username/model-name"
```

**检查开放PR（在--create-pr之前始终运行）：**
```bash
uv run scripts/evaluation_manager.py get-prs --repo-id "username/model-name"
```
列出模型仓库的所有开放拉取请求。显示PR编号、标题、作者、日期和URL。

**运行评估作业（推理提供商）：**
```bash
hf jobs uv run scripts/inspect_eval_uv.py \
  --flavor "cpu-basic|t4-small|..." \
  --secret HF_TOKEN=$HF_TOKEN \
  -- --model "model-id" \
     --task "task-name"
```

或使用Python助手：

```bash
uv run scripts/run_eval_job.py \
  --model "model-id" \
  --task "task-name" \
  --hardware "cpu-basic|t4-small|..."
```

**运行vLLM评估（自定义模型）：**
```bash
# 使用vLLM的lighteval
hf jobs uv run scripts/lighteval_vllm_uv.py \
  --flavor "a10g-small" \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- --model "model-id" \
     --tasks "leaderboard|mmlu|5"

# 使用vLLM的inspect-ai
hf jobs uv run scripts/inspect_vllm_uv.py \
  --flavor "a10g-small" \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- --model "model-id" \
     --task "mmlu"

# 助手脚本（自动硬件选择）
uv run scripts/run_vllm_eval_job.py \
  --model "model-id" \
  --task "leaderboard|mmlu|5" \
  --framework lighteval
```

### Model-Index格式

生成的model-index遵循以下结构：

```yaml
model-index:
  - name: Model Name
    results:
      - task:
          type: text-generation
        dataset:
          name: Benchmark Dataset
          type: benchmark_type
        metrics:
          - name: MMLU
            type: mmlu
            value: 85.2
          - name: HumanEval
            type: humaneval
            value: 72.5
        source:
          name: Source Name
          url: https://source-url.com
```

警告：不要在模型名称中使用markdown格式。使用表格中的确切名称。仅在source.url字段中使用URL。

### 错误处理
- **找不到表格**：如果未检测到评估表格，脚本将报告
- **无效格式**：对于格式错误的表格，提供明确的错误消息
- **API错误**：对瞬时Artificial Analysis API故障的重试逻辑
- **令牌问题**：尝试更新前的验证
- **合并冲突**：添加新条目时保留现有的model-index条目
- **空间创建**：优雅处理命名冲突和硬件请求失败

### 最佳实践

1. **首先检查现有PR**：在创建任何新PR之前运行`get-prs`，以避免重复
2. **始终从`inspect-tables`开始**：查看表格结构并获取正确的提取命令
3. **使用`--help`获取指导**：运行`inspect-tables --help`查看完整工作流
4. **先预览**：默认行为打印YAML；在使用`--apply`或`--create-pr`之前查看
5. **验证提取的值**：手动比较YAML输出与README表格
6. **对多表格README使用`--table N`**：当存在多个评估表格时需要
7. **对比较表格使用`--model-name-override`**：从`inspect-tables`输出复制确切的列标题
8. **为他人创建PR**：更新您不拥有的模型时使用`--create-pr`
9. **每个仓库一个模型**：只将主模型的结果添加到model-index
10. **YAML名称中无markdown**：YAML中的模型名字段应为纯文本

### 模型名称匹配

从包含多个模型（作为列或行）的评估表格中提取时，脚本使用**精确规范化令牌匹配**：

- 删除markdown格式（粗体`**`，链接`[]()` ）
- 规范化名称（小写，将`-`和`_`替换为空格）
- 比较令牌集：`"OLMo-3-32B"` → `{"olmo", "3", "32b"}`匹配`"**Olmo 3 32B**"`或`"[Olmo-3-32B](...)`
- 仅当令牌完全匹配时才提取（处理不同的词序和分隔符）
- 如果未找到精确匹配，则失败（而不是从相似名称中猜测）

**对于基于列的表格**（基准测试作为行，模型作为列）：
- 找到与模型名称匹配的列标题
- 仅从该列提取分数

**对于转置表格**（模型作为行，基准测试作为列）：
- 找到第一列中与模型名称匹配的行
- 仅从该行提取所有基准测试分数

这确保只提取正确模型的分数，从不提取不相关的模型或训练检查点。

### 常见模式

**更新您自己的模型：**
```bash
# 从README提取并直接推送
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "your-username/your-model" \
  --task-type "text-generation"
```

**更新他人的模型（完整工作流）：**
```bash
# 步骤1：始终首先检查现有PR
uv run scripts/evaluation_manager.py get-prs \
  --repo-id "other-username/their-model"

# 步骤2：如果不存在开放PR，则继续创建一个
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "other-username/their-model" \
  --create-pr

# 如果存在开放PR：
# - 警告用户有关现有PR
# - 向他们显示PR URL
# - 除非用户明确确认，否则不要创建新PR
```

**导入新的基准测试：**
```bash
# 步骤1：检查现有PR
uv run scripts/evaluation_manager.py get-prs \
  --repo-id "anthropic/claude-sonnet-4"

# 步骤2：如果没有PR，从Artificial Analysis导入
AA_API_KEY=... uv run scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "anthropic/claude-sonnet-4" \
  --create-pr
```

### 故障排除

**问题**："在README中未找到评估表格"
- **解决方案**：检查README是否包含带有数字分数的markdown表格

**问题**："在转置表格中找不到模型'X'"
- **解决方案**：脚本将显示可用模型。使用`--model-name-override`，并从列表中使用确切的名称
- **示例**：`--model-name-override "**Olmo 3-32B**"`

**问题**："AA_API_KEY未设置"
- **解决方案**：设置环境变量或添加到.env文件

**问题**："令牌没有写入访问权限"
- **解决方案**：确保HF_TOKEN对仓库具有写入权限

**问题**："在Artificial Analysis中找不到模型"
- **解决方案**：验证creator-slug和model-name与API值匹配

**问题**："硬件需要付款"
- **解决方案**：在您的Hugging Face账户中添加付款方式，以使用非CPU硬件

**问题**："vLLM内存不足"或CUDA OOM
- **解决方案**：使用更大的硬件类型，减少`--gpu-memory-utilization`，或对多GPU使用`--tensor-parallel-size`

**问题**："模型架构不支持vLLM"
- **解决方案**：对HuggingFace Transformers使用`--backend hf`（inspect-ai）或`--backend accelerate`（lighteval）

**问题**："需要信任远程代码"
- **解决方案**：为具有自定义代码的模型添加`--trust-remote-code`标志（例如，Phi-2、Qwen）

**问题**："未找到聊天模板"
- **解决方案**：仅对包含聊天模板的指令调优模型使用`--use-chat-template`

### 集成示例

**Python脚本集成：**
```python
import subprocess
import os

def update_model_evaluations(repo_id, readme_content):
    """使用README中的评估更新模型卡片。"""
    result = subprocess.run([
        "python", "scripts/evaluation_manager.py",
        "extract-readme",
        "--repo-id", repo_id,
        "--create-pr"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"成功更新 {repo_id}")
    else:
        print(f"错误: {result.stderr}")
```