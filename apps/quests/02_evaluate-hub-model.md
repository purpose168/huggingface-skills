# 第1周：评估Hub模型

📣 任务：向Hub上的模型卡片添加评估结果。我们将共同构建一个开源模型性能的分布式排行榜。

>[!NOTE]
> 为排行榜应用做出贡献可获得额外XP。在[hub](https://huggingface.co/spaces/hf-skills/distributed-leaderboard/discussions)或[GitHub](https://github.com/huggingface/skills/blob/main/apps/evals-leaderboard/app.py)上打开PR以获得你的（额外）XP。

## 为什么这很重要

没有评估数据的模型卡片很难比较。通过向元数据添加结构化评估结果，我们使模型更容易比较和审查。你的贡献为排行榜提供动力，并帮助社区找到最适合其需求的模型。此外，通过分布式方式执行此操作，我们可以与社区分享我们的评估结果。

## 目标

- 向Hub上的100个热门模型添加评估分数
- 在热门模型上包含AIME 2025、BigBenchHard、LiveCodeBench、MMLU、ARC等评估。
- 可以只包含模型可用的部分基准测试。
- 构建一个排行榜应用，显示热门模型的评估结果。

## XP等级

参与很简单。我们需要让模型作者在他们的模型卡片中显示评估结果。这是一项清理工作！

| 等级 | XP | 描述 | 计算标准 |
|------|-----|--------|----------|
| 🐢 贡献者 | 1 XP | 从一个基准测试中提取评估结果并更新其模型卡片。 | 任何包含评估数据的仓库PR。 |
| 🐕 评估者 | 5 XP | 从第三方基准测试（如Artificial Analysis）导入分数。 | 未定义的基准测试分数和已合并的PR。 |
| 🦁 高级 | 20 XP | 使用inspect-ai运行自己的评估并发布结果。 | 原始评估运行和已合并的PR。 |
| 🐉 额外 | 20 XP | 为排行榜应用做出贡献。 | 在hub或GitHub上的任何已合并PR。 |
| 🤢 垃圾 | -20 XP | 打开无用的PR。 | 重复PR、不正确的评估分数、不正确的基准测试分数 |

> [!WARNING]
> 这次黑客马拉松是关于推进开源AI的发展。我们需要的是帮助所有人的有用PR，而不仅仅是指标。

## 技能

使用`hf-evaluation/`完成此任务。关键功能：

- 从模型作者发布的现有README内容中提取评估表格。
- 从[Artificial Analysis](https://artificial.com/)导入基准测试分数。
- 使用[inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai)在[HF Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)上运行自己的评估。
- 更新模型卡片中的model-index元数据。

>[!NOTE]
> 有关更多详细信息，请查看[SKILL.md](https://github.com/huggingface/skills/blob/main/hf-evaluation/SKILL.md)。

### 从README中提取评估表格

1. 从hub上的*热门模型*中选择一个没有评估数据的Hub模型
2. 使用技能提取或添加基准测试分数
3. 创建PR（如果是你自己的模型，则直接推送）

代理将使用此脚本从模型的README中提取评估表格。

```bash
python hf-evaluation/scripts/evaluation_manager.py extract-readme \
  --repo-id "model-author/model-name" --dry-run
```

### 从Artificial Analysis导入分数

1. 找到在外部网站上有基准测试数据的模型
2. 使用`import-aa`从Artificial Analysis API获取分数
3. 创建包含正确归因的评估数据的PR

代理将使用此脚本从Artificial Analysis API获取分数并将其添加到模型卡片中。

```bash
python hf-evaluation/scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" --model-name "claude-sonnet-4" \
  --repo-id "target/model" --create-pr
```

### 使用inspect-ai运行自己的评估并发布结果

1. 选择一个评估任务（MMLU、GSM8K、HumanEval等）
2. 在HF Jobs基础设施上运行评估
3. 使用你的结果和方法更新模型卡片

代理将使用此脚本在HF Jobs基础设施上运行评估并使用结果更新模型卡片。

```bash
HF_TOKEN=$HF_TOKEN hf jobs uv run hf-evaluation/scripts/inspect_eval_uv.py \
  --flavor a10g-small --secret HF_TOKEN=$HF_TOKEN \
  -- --model "meta-llama/Llama-2-7b-hf" --task "mmlu"
```

## 提示

- 始终先使用`--dry-run`预览更改，然后再推送
- 检查模型为行、基准测试为列的转置表格
- 对不属于你的模型提交PR时要小心 — 大多数维护者会欣赏评估贡献，但要尊重他们。
- 手动验证提取的分数，并在需要时关闭PR。

## 资源

- [SKILL.md](../../hf-evaluation/SKILL.md) — 完整技能文档
- [Example Usage](../../hf-evaluation/examples/USAGE_EXAMPLES.md) — 工作示例
- [Metric Mapping](../../hf-evaluation/examples/metric_mapping.json) — 标准指标类型