# Hugging Face 技能

Hugging Face 技能是用于定义 AI/ML 任务的标准，如数据集创建、模型训练和评估。它们与所有主要的编码代理工具兼容，如 OpenAI Codex、Anthropic 的 Claude Code、Google DeepMind 的 Gemini CLI 和 Cursor。

本仓库中的技能遵循标准化的 [Agent Skill](https://agentskills.io/home) 格式。

## 技能如何工作？

实际上，技能是自包含的文件夹，它们将指令、脚本和资源打包在一起，供 AI 代理在特定用例中使用。每个文件夹都包含一个 `SKILL.md` 文件，该文件具有 YAML 前置内容（名称和描述），后面是编码代理在技能激活时遵循的指导。

> [!NOTE]
> "技能"（Skills）实际上是 Anthropic 在 Claude AI 和 Claude Code 中使用的术语，并未被其他代理工具采用，但我们喜欢这个术语！OpenAI Codex 使用 `AGENTS.md` 文件来定义编码代理的指令。Google Gemini 使用 "扩展"（extensions）来定义编码代理的指令，存储在 `gemini-extension.json` 文件中。**本仓库与所有这些工具兼容，甚至更多！**

> [!TIP]
> 如果您的代理不支持技能，您可以直接使用 [`agents/AGENTS.md`](agents/AGENTS.md) 作为备用方案。

## 安装

Hugging Face 技能与 Claude Code、Codex 和 Gemini CLI 兼容。正在开发与 Cursor、Windsurf 和 Continue 的集成。

### Claude Code

1. 将仓库注册为插件市场：
   
```
/plugin marketplace add huggingface/skills
```

2. 要安装技能，请运行：
   
```
/plugin install <skill-name>@huggingface/skills
```

例如：

```
/plugin install hugging-face-cli@huggingface/skills
```

### Codex

1. Codex 将通过 `AGENTS.md` 文件识别技能。您可以使用以下命令验证指令是否已加载：

```
codex --ask-for-approval never "Summarize the current instructions."
```

2. 有关更多详细信息，请参阅 [Codex AGENTS 指南](https://developers.openai.com/codex/guides/agents-md)。

### Gemini CLI

1. 本仓库包含 `gemini-extension.json` 以与 Gemini CLI 集成。

2. 本地安装：

```
gemini extensions install . --consent
```

或使用 GitHub URL：

```
gemini extensions install https://github.com/huggingface/skills.git --consent
```

4. 有关更多帮助，请参阅 [Gemini CLI 扩展文档](https://geminicli.com/docs/extensions/#installing-an-extension)。

## 技能

本仓库包含一些入门技能。您也可以向仓库贡献自己的技能。

### 可用技能

<!-- 此表格由 scripts/generate_agents.py 自动生成。请勿手动编辑。 -->
<!-- BEGIN_SKILLS_TABLE -->
| 名称 | 描述 | 文档 |
|------|-------------|---------------|
| `hugging-face-cli` | 使用 hf CLI 执行 Hugging Face Hub 操作。下载模型/数据集、上传文件、管理仓库和运行云计算作业。 | [SKILL.md](skills/hugging-face-cli/SKILL.md) |
| `hugging-face-datasets` | 在 Hugging Face Hub 上创建和管理数据集。支持初始化仓库、定义配置/系统提示、流式行更新和基于 SQL 的数据集查询/转换。 | [SKILL.md](skills/hugging-face-datasets/SKILL.md) |
| `hugging-face-evaluation` | 在 Hugging Face 模型卡片中添加和管理评估结果。支持从 README 内容提取评估表格、从 Artificial Analysis API 导入分数，以及使用 vLLM/lighteval 运行自定义评估。 | [SKILL.md](skills/hugging-face-evaluation/SKILL.md) |
| `hugging-face-jobs` | 在 Hugging Face 基础设施上运行计算作业。执行 Python 脚本、管理计划作业和监控作业状态。 | [SKILL.md](skills/hugging-face-jobs/SKILL.md) |
| `hugging-face-model-trainer` | 使用 TRL 在 Hugging Face Jobs 基础设施上训练或微调语言模型。涵盖 SFT、DPO、GRPO 和奖励建模训练方法，以及用于本地部署的 GGUF 转换。包括硬件选择、成本估算、Trackio 监控和 Hub 持久化。 | [SKILL.md](skills/hugging-face-model-trainer/SKILL.md) |
| `hugging-face-paper-publisher` | 在 Hugging Face Hub 上发布和管理研究论文。支持创建论文页面、将论文链接到模型/数据集、声明作者身份，以及生成基于 Markdown 的专业研究文章。 | [SKILL.md](skills/hugging-face-paper-publisher/SKILL.md) |
| `hugging-face-tool-builder` | 为 Hugging Face API 操作构建可重用脚本。适用于链接 API 调用或自动化重复任务。 | [SKILL.md](skills/hugging-face-tool-builder/SKILL.md) |
| `hugging-face-trackio` | 使用 Trackio 跟踪和可视化 ML 训练实验。通过 Python API 记录指标并通过 CLI 检索它们。支持与 HF Spaces 同步的实时仪表板。 | [SKILL.md](skills/hugging-face-trackio/SKILL.md) |
<!-- END_SKILLS_TABLE -->

### 在编码代理中使用技能

技能安装后，在向编码代理提供指令时直接提及它：

- "使用 HF LLM 训练器技能来估算 70B 模型运行所需的 GPU 内存。"
- "使用 HF 模型评估技能在最新检查点上启动 `run_eval_job.py`。"
- "使用 HF 数据集创建器技能来起草新的少样本分类模板。"
- "使用 HF 论文发布技能来索引我的 arXiv 论文并将其链接到我的模型。"

编码代理在完成任务时会自动加载相应的 `SKILL.md` 指令和辅助脚本。

### 贡献或自定义技能

1. 复制现有的技能文件夹之一（例如，`hf-datasets/`）并将其重命名。
2. 更新新文件夹的 `SKILL.md` 前置内容：
   ```markdown
   ---
   name: my-skill-name
   description: 描述技能的功能和使用场景
   ---

   # 技能标题
   指导 + 示例 + 约束
   ```
3. 添加或编辑指令引用的支持脚本、模板和文档。
4. 向 `.claude-plugin/marketplace.json` 添加一个条目，包含简洁、人类可读的描述。
5. 运行 `python scripts/generate_agents.py` 来验证结构。
6. 在编码代理中重新安装或重新加载技能包，以便更新的文件夹可用。

### 市场

`.claude-plugin/marketplace.json` 文件列出了技能及其人类可读的描述，用于插件市场。CI 会验证技能名称和路径是否在 `SKILL.md` 文件和 `marketplace.json` 之间匹配，但描述是单独维护的：`SKILL.md` 描述指导 Claude 何时激活技能，而市场描述是为浏览可用技能的人类编写的。

### 其他参考
- 直接在 [huggingface/skills](https://github.com/huggingface/skills) 浏览最新的指令、脚本和模板。
- 查看 Hugging Face 文档，了解您在每个技能中引用的特定库或工作流程。